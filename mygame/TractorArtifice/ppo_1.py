import argparse
import os
import random

import numpy as np
import torch
from DNQ.mygame.TractorArtifice.game_env import tractor_game
from tractor_utils import getNowTimePath, drawBrokenLine
from DNQ.mygame.TractorArtifice.game_env.baselinePolicy import baselineColdeck
from DNQ.mygame.TractorArtifice.game_env.tractor_game import Action,Player,dfsPrintActList
from encoder import handTo01code, getActionFeature, getBaseFea, actTo01code
from encoder import cardsTo01code
from tractor_network import AgentNet, AgentCriNet

# start to define the workers...
from DNQ.mygame.TractorArtifice.game_env.tractor_game import CC


class Dppoagent:
    def __init__(self,id,args,critic_net,worker):#代表这个智能体
        self.id=id
        self.args=args
        self.worker=worker
        in_fea=345#有待输入
        self.net = AgentNet().cuda(self.worker.cudaID)
        self.critic_net=critic_net
        # print(self.worker.cudaID,next(self.net.lstm1.parameters()).device)
        self.baseFea=None
        self.lock=False
        self.lanta = args.lanta
        self.gamaPow = np.zeros((110))
        self.galanPow = np.zeros((110))#gama*lanta
        self.gamaPow[0]=self.galanPow[0]=1
        for i in range(1,110):
            self.gamaPow[i]=self.gamaPow[i-1]*args.gama
            self.galanPow[i]=self.galanPow[i-1]*args.gama*self.lanta
        self.initMenory()
        self.p=worker.env.players[id]
        self.CLIP_EPSL = 0.2
        self.initNet()
    def initNet(self):
        baseNet=list(self.net.mlp_base.parameters())+list(self.net.lstm1.parameters())
        self.optimizer = [torch.optim.Adam(self.critic_net.parameters(), lr=self.args.value_lr),
                          torch.optim.Adam(list(self.net.mlp_act1.parameters())+baseNet, lr=self.args.value_lr),
                          torch.optim.Adam(list(self.net.mlp_act2.parameters())+baseNet, lr=self.args.value_lr)]
    def initMenory(self):#初始化memory,来记录本局游戏的数据
        self.memoryAllAct = [[] for i in range(25)]
        self.memoryProb = [[] for i in range(25)]#储存当时神经网络决策的动作和概率
        self.memoryBinProb = [[] for i in range(25)]
        self.memoryBase = [[] for i in range(25)]
        self.memoryOtherAct = [[] for i in range(25)]
        self.memoryCriticVal = [[] for i in range(25)]
        self.memoryReward = [0 for i in range(25)]
    def printMemory(self):
        dfsPrintActList(self.memoryAllAct)
        print(self.memoryProb)
        print(self.memoryBinProb)

    def pushMemory(self,i,allAct:list,baseFea,base_out,othertActFea,act,binact=None):#allAct内部是多个list
        self.memoryAllAct[i].append(allAct)
        self.memoryProb[i].append(act)#id,prob
        # print(otherAct)
        self.memoryBase[i].append(baseFea)
        self.memoryOtherAct[i].append(othertActFea)
        x=self.critic_net.forward(baseFea.cuda(self.worker.cudaID),othertActFea.cuda(self.worker.cudaID))
        self.memoryCriticVal[i].append(x.cpu())
        self.memoryBinProb[i].append(binact)
    def pushReward(self,i,reward):
        self.memoryReward[i] =reward
    def update_network(self, critic_shared_grad_buffer, actor_shared_grad_buffer,
                       shared_critic_model, shared_actor_model, critic_counter, actor_counter, traffic_signal):
        if self.lock:#如果该agent被锁定，则不更新参数。
            return
    def chooseFirstActFromMax2(self,roundId,env, otherAct,allActionList:list):#无脑出最大的
        return env.getActListMax(allActionList)
    def chooseFirstActFromMax1(self,roundId,env, otherAct,allActionList:list):#无脑出个数最多的
        maxLen = 0
        maxLenActList = []
        for a in allActionList:  # 找出长度最长的动作
            maxLen = max(maxLen, a.len)
        for i in range(len(allActionList)):  # 找出长度最长的动作的列表
            if maxLen == allActionList[i].len:
                maxLenActList.append(allActionList[i])
        act_i = random.randint(0, len(maxLenActList) - 1)
        return maxLenActList[act_i]
    def chooseFirstActFromRandom(self,roundId,env, otherAct,allActionList:list):#随机出牌
        act_i = random.randint(0, len(allActionList) - 1)
        return allActionList[act_i]
    def checkProb(self,probList):
        # print(len(probList))
        for i in range(len(probList)):
            if probList[i]<0:
                print(probList)
                probList[i]=0
    def chooseFirstActFromNet(self,roundId,env, otherAct,allActionList:list):#从神经网络获取
        baseFea = getBaseFea(env,0,self.id,self.p.isDealer())  # self.id
        otherActFea = getActionFeature(otherAct)
        # print(self.worker.cudaID,next(self.net.parameters()).device)
        base_out = self.net.forward_base(baseFea.cuda(self.worker.cudaID), otherActFea.cuda(self.worker.cudaID))  # base的特征向量
        # if np.random.uniform(0, 1) < self.epsl:
        #     ansAct = allActionList[np.random.randint(0, len(allActionList))]
        probList = torch.zeros((len(allActionList)))
        for i in range(len(allActionList)):  # 遍历所有动作
            nowActTensor = actTo01code(allActionList[i]).unsqueeze(dim=0)
            probList[i] = self.net.forward_act(base_out, nowActTensor.cuda(self.worker.cudaID)).view(-1)[0].cpu()
        # print(probList)
        probList = torch.softmax(probList, dim=0).cpu().detach()
        self.checkProb(probList)
        try:
            actid = torch.multinomial(probList, num_samples=1, replacement=True)[0].cpu().detach()
        except RuntimeError as e:
            actid = 0
            print(probList)
            # print(actid)
        # print(baseFea)
        self.pushMemory( roundId, [a.tolist() for a in allActionList],baseFea,base_out,otherActFea, (actid,probList[actid]) )
        return allActionList[actid]
    def updateINF(self,roundId, p:Player, act: list, kind):#kind是第一个出牌的玩家的花色。返回种类和在cards_decorList中位置的编号
        actList = []
        for j in range(p.cards_decorLen[kind]):  # 先看本花色有木有
            if p.cards_decorList[kind][j] != 0:
                actList.append((kind, j))
        if len(actList) == 0:  # 本花色没有，去其它花色找
            for i in range(5):
                if i == kind or p.cards_decorLen[i] == 0:
                    continue
                for j in range(p.cards_decorLen[i]):
                    if p.cards_decorList[i][j] != 0:
                        actList.append((i, j))
        # ans = actList[random.randint(0, len(actList) - 1)]
        baseFea = getBaseFea(self.worker.env, self.useSeq, self.id, self.p.isDealer())  # self.id
        otherActFea=getActionFeature(self.otherAct)
        base_out = self.net.forward_base(baseFea.cuda(self.worker.cudaID), otherActFea.cuda(self.worker.cudaID))  # base的特征向量
        probList = torch.zeros((len(actList)))
        allActionList=[]
        for i in range(len(actList)):  # 遍历所有动作
            a=actList[i]
            nowAct=Action([p.cards_decorList[a[0]][a[1]]])
            allActionList.append([p.cards_decorList[a[0]][a[1]]])
            nowActTensor = actTo01code(nowAct).unsqueeze(dim=0)
            # print(self.baseFea.shape,nowActTensor.shape)
            probList[i] = self.net.forward_act(base_out, nowActTensor.cuda(self.worker.cudaID)).view(-1)[0].cpu()
        probList = torch.softmax(probList, dim=0).cpu().detach()
        self.checkProb(probList)
        try:
            actid = torch.multinomial(probList, num_samples=1, replacement=True)[0].cpu().detach()
        except RuntimeError as e:
            actid = 0
            print(probList)
        # self.otherAct[p.id].
        self.pushMemory(roundId, allActionList,baseFea,base_out, otherActFea,(actid, probList[actid]))
        return actList[actid][0], actList[actid][1]
    def chooseOtherAct(self,roundId,useSeq,env, otherAct, ansUp, ansDown, firstKind, sortCardList2):#ansUp代表较大的动作，ansDown代表较小的动作
        nu = len(ansUp)
        nd = len(ansDown)
        baseFea = getBaseFea(env, useSeq, self.id, self.p.isDealer()).cuda(self.worker.cudaID)#self.id
        otherActFea = getActionFeature(otherAct).cuda(self.worker.cudaID)
        # print(actFea)
        base_out=self.net.forward_base(baseFea,otherActFea)#base的特征向量
        # print(base_out.shape)
        allActionList=ansUp
        fpTuple=None
        if nu>0 and nd>0:
            fpprob=self.net.forward_bin(base_out).squeeze(dim=0).cpu().detach()#选择出大的还是小的,1是大，0是小
            torch.softmax(fpprob, dim=0).cpu().detach()
            # print(fpprob.shape)
            self.checkProb(fpprob)
            try:
                fp=torch.multinomial(fpprob, num_samples=1, replacement=True)[0].cpu().detach()
            except RuntimeError as e:
                fp = 0
                print(fpprob)
            # print(fp)
            fpprob=fpprob[fp]
            fpTuple=(fp,fpprob)
            # print(fpprob)
            if fp==0:
                allActionList=ansDown
        elif nd>0:
            allActionList = ansDown
        # print(allActionList)
        # [[53, 53], [38, 38], [32, 32], [1000, 1000]]
        haveINF=False
        for a in allActionList[0]:#寻找是否有INF
            if a== tractor_game.INF:
                haveINF=True
                break
        if haveINF:#把动作拆散
            ansAct= allActionList[0]
            self.baseFea = base_out
            self.otherAct = otherAct#这两个变量用于传递到updateINF里面
            self.useSeq=useSeq
            env._useCardsContainINF(roundId,env.players[self.id], ansAct, firstKind, self.updateINF, sortCardList2[self.id])
        else:

            probList=torch.zeros((len(allActionList)))
            for i in range(len(allActionList)):#遍历所有动作
                nowActTensor=cardsTo01code(allActionList[i]).unsqueeze(dim=0)
                # print(nowActTensor.shape)
                probList[i]=self.net.forward_act(base_out,nowActTensor.cuda(self.worker.cudaID)).view(-1)[0].cpu()
            probList=torch.softmax(probList, dim=0).cpu().detach()
            self.checkProb(probList)
            try:
                actid = torch.multinomial(probList, num_samples=1, replacement=True)[0].cpu().detach()
            except RuntimeError as e:
                actid=0
                print(probList)
            # actid = torch.multinomial(probList, num_samples=1, replacement=True)[0].cpu().detach()
            ansAct=allActionList[actid]
            self.pushMemory(roundId,allActionList,baseFea,base_out, otherActFea,(actid,probList[actid]),fpTuple)
            env._useCardsContainINF(roundId,env.players[self.id], ansAct, firstKind, None, sortCardList2[self.id])
        return ansAct

    def chooseOtherAct_fix(self,roundId,useSeq,env, otherAct, ansUp, ansDown, firstKind, sortCardList2):#固定策略，固定选取第0个
        nu = len(ansUp)
        nd = len(ansDown)
        allActionList = ansUp
        fpprob = 1
        if nu > 0 and nd > 0:
            allActionList = ansUp
        elif nd > 0:
            fp = 0
            allActionList = ansDown
        else:
            fp = 1
        ansAct = allActionList[0]
        haveINF = False
        for a in ansAct:  # 寻找是否有INF
            if a == tractor_game.INF:
                haveINF = True
                break
        if haveINF:  # 把动作拆散
            env._useCardsContainINF(roundId, env.players[self.id], ansAct, firstKind, tractor_game.randomUpdateINF,
                                    sortCardList2[self.id])
        else:
            env._useCardsContainINF(roundId, env.players[self.id], ansAct, firstKind, None, sortCardList2[self.id])
        return allActionList[0]
    def initLearn(self,roundUp,rewardList):#使用GAE计算优势函数,要计算每一步的Gt-critic(state)
        self.roundInd=np.zeros((roundUp+1),dtype=np.int32)#roundId转换advantages的id
        self.detlaList = []
        id=0
        for i in range(roundUp):##计算deala
            self.roundInd[i] = id
            id +=len(self.memoryCriticVal[i])
        self.roundInd[roundUp] = id
        # print(G0,self.args.gama)
        for i in range(roundUp):##计算deala
            m=len(self.memoryCriticVal[i])
            for j in range(0,m):
                criTensor=self.memoryCriticVal[i][j]
                v=criTensor[0][0].detach()
                # print(G0,q)
                if j == m - 1:
                    reward= rewardList[i]
                    if i==roundUp-1:
                        next_v=0
                    else:
                        next_v = self.memoryCriticVal[i+1][0][0][0].detach()
                else:
                    reward=0
                    next_v = self.memoryCriticVal[i][j+1][0][0].detach()
                self.detlaList.append((reward+self.gamaPow[1]*next_v - v).clone())
                # print(G0)

        # print(rewardList)
        self.advantages=[]#计算GAE优势函数
        adv0 = 0
        for i in range(len(self.detlaList)):
            adv0 +=self.galanPow[i]*self.detlaList[i]
        for i in range(len(self.detlaList)):
            self.advantages.append(adv0.clone())
            adv0 -= self.detlaList[i]
            adv0 /= self.galanPow[1]
        # print(self.advantages)
    def selflearn(self,roundId,roundUp,rewardList,worker):
        allActList=self.memoryAllAct[roundId]
        actPolicy=self.memoryProb[roundId]#储存当时神经网络决策的动作和概率
        actPolicy_bin=self.memoryBinProb[roundId]#当时玩家采取的二分类动作,有可能是None
        baseTensor=self.memoryBase[roundId]
        otherActTensor=self.memoryOtherAct[roundId]
        n=len(baseTensor)

        for i in range(n):
            statei = baseTensor[i].cuda(self.worker.cudaID)
            otherActi=otherActTensor[i].cuda(self.worker.cudaID)
            if i == n - 1:
                reward = self.memoryReward[self.roundInd[roundId]]
                if roundId==roundUp-1:
                    next_statei = None
                    next_otherActi = None
                else:
                    next_statei = self.memoryBase[roundId+1][0].cuda(self.worker.cudaID)
                    next_otherActi = self.memoryOtherAct[roundId+1][0].cuda(self.worker.cudaID)
            else:
                reward=0
                next_statei=baseTensor[i+1].cuda(self.worker.cudaID)
                next_otherActi = otherActTensor[i + 1].cuda(self.worker.cudaID)
            # base_out = self.net.forward_base(statei,otherActi)
            # tderro = Gt - critic_q  # tderro也是优势函数,根据具体问题可以在is_ter==True时阻断
            # advantage=tderro.detach()#普通优势函数
            advantage=self.advantages[self.roundInd[roundId]+i]#得到GAE优势
            # print(advantage)
            for _ in range(self.args.dataUseCnt):#训练critic
                self.net.zero_grad()
                self.critic_net.zero_grad()
                # base_out = self.net.forward_base(statei, otherActi)
                v = self.critic_net.forward(statei,otherActi)
                if next_statei!=None:
                    next_v=self.critic_net.forward(next_statei,next_otherActi)
                    tderro = reward+self.gamaPow[1]*next_v - v#因为自动作没有奖励，所以Q(s,a)都是Gt
                else:
                    tderro= reward - v
                critic_loss = tderro.pow(2).mean()
                critic_loss.backward()
                self.optimizer[0].step()
                # 下面是更新决策二分类网络
                if actPolicy_bin[i]!=None:
                    # print(actPolicy_bin[i])
                    self.net.zero_grad()
                    base_out = self.net.forward_base(statei, otherActi)
                    nowRate = self.net.forward_bin(base_out)
                    act=torch.Tensor([[actPolicy_bin[i][0]]]).cuda(self.worker.cudaID).long()
                    nowRate=nowRate.gather(1, act)
                    preRate =torch.Tensor([[actPolicy_bin[i][1]]]).cuda(self.worker.cudaID)#旧策略
                    # print(nowRate.shape,preRate.shape)
                    ratio = torch.exp(torch.log(nowRate) - torch.log(preRate))  # 计算p1/p2防止精度问题
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.CLIP_EPSL, 1 + self.CLIP_EPSL) * advantage
                    actor_loss_bin = -torch.max(torch.min(surr1, surr2),0.8*advantage).mean()
                    actor_loss_bin.backward()
                    self.optimizer[1].step()

                #下面是更新act网络
                self.net.zero_grad()
                base_out = self.net.forward_base(statei, otherActi)
                cnt=len(allActList[i])
                probList = torch.zeros((cnt)).cuda(self.worker.cudaID)
                for j in range(cnt):#遍历所有可能
                    # print(allActList[i][j])
                    actTensor=cardsTo01code(allActList[i][j]).unsqueeze(dim=0).cuda(self.worker.cudaID)
                    x=self.net.forward_act(base_out, actTensor)
                    probList[j] = x.view(-1)[0]
                    # print(probList)
                probList = torch.softmax(probList, dim=0).unsqueeze(dim=0)
                # print(probList)
                act = torch.Tensor([[actPolicy[i][0]]]).cuda(self.worker.cudaID).long()
                nowRate = probList.gather(1, act)
                preRate = torch.Tensor([[actPolicy[i][1]]]).cuda(self.worker.cudaID)  # 旧策略
                # print(nowRate.shape,preRate.shape)
                ratio = torch.exp(torch.log(nowRate) - torch.log(preRate))  # 计算p1/p2防止精度问题
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.CLIP_EPSL, 1 + self.CLIP_EPSL) * advantage
                actor_loss = -torch.max(torch.min(surr1, surr2),0.8*advantage).mean()
                actor_loss.backward()
                self.optimizer[2].step()
class DppoWorkers:
    def __init__(self, id, shared_obs_state, args):
        self.id = id
        self.args = args
        self.cudaID=args.cudaIDList[id]
        self.env = CC()
        self.gama = args.gama
        self.CLIP_EPSL = 0.2
        self.lanta = 0.2
        self.shared_obs_state = shared_obs_state
        critic_net=AgentCriNet().cuda(self.cudaID)
        self.agents=(Dppoagent(0,args,critic_net,self),Dppoagent(1,args,critic_net,self),
                     Dppoagent(2,args,critic_net,self),Dppoagent(3,args,critic_net,self))
        self.trainlist=self.args.trainlist.copy()
        self.savePath = getNowTimePath()
        # print(path)

        self.memory_batch = 50
        MEMORY_CAPACITY = 26
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        OVERLAY_CNT = 1  # 针对此游戏的叠帧操作
        self.OVERLAY_CNT = OVERLAY_CNT
        self.rewardFun=getattr(self, self.args.rewardFun)
        # beginShape = self.resetPic(self.env.reset()).shape


    def initAllState(self):#初始化游戏状态，返回开始时的状态
        self.env.reset()
        return
    def getGameState(self):
        li=[handTo01code(self.env.players[i]) for i in range(4)]

        return li,self.env.getNowUsedCards()

    def saveAgentNet(self):
        path=self.savePath
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.agents[0].critic_net, path + "agentCriticNet.pt")  #
        for i in range(4):
            torch.save(self.agents[i].net, path+"agentNet"+str(i)+".pt")  #
    def readAgentNet(self,path):
        if not os.path.exists(path):
            print("模型不存在")
            return False
        p0 = path + "agentCriticNet.pt"
        critic = torch.load(p0).cuda(self.cudaID)
        for i in range(4):
            p=path + "agentNet" + str(i) + ".pt"
            if not os.path.exists(p):
                print("模型不存在")
                return False
            self.agents[i].critic_net=critic
            self.agents[i].net=torch.load(p).cuda(self.cudaID)
            self.agents[i].initNet()
        print("read success")
        return True
    def firstPlayerPolicy(self,roundId,act,firstPlayerID,sortCardList1):  # 第一个人的出牌策略,需要神经网络决策,act是其他玩家出的牌，这里一定是空
        selfAllAct = self.env.getAllFirstAct(sortCardList1[firstPlayerID], self.env.players[firstPlayerID])
        # dfsPrintActList(selfAllAct)  # 输出先手动作集合
        if self.args.trainlist[firstPlayerID]==2:
            ans=self.agents[firstPlayerID].chooseFirstActFromMax1(roundId,self.env, act,selfAllAct)
        elif self.args.trainlist[firstPlayerID]<2:
            # print(firstPlayerID)
            ans=self.agents[firstPlayerID].chooseFirstActFromNet(roundId,self.env, act,selfAllAct)
        return ans
    def otherPlayerPolicy(self,roundId,useSeq, act, nextID, firstKind, sortCardList2, cards):  # 使用的cards只能是单牌，对子，连对
        if len(cards) == 0:
            return None
        ansUp, ansDown, isHave = self.env.getAllAct(sortCardList2[nextID], self.env.players[nextID], cards)
        if self.args.trainlist[nextID]==2:
            ans=self.agents[nextID].chooseOtherAct_fix(roundId,useSeq,self.env, act, ansUp, ansDown, firstKind, sortCardList2)
        elif self.args.trainlist[nextID]<2:
            # print(nextID)
            ans=self.agents[nextID].chooseOtherAct(roundId,useSeq,self.env, act, ansUp, ansDown, firstKind, sortCardList2)
        return ans
    def setReward(self,env,playerId,firstPlayerID, sc, isTer, info,seqCardReward):#设置奖励函数，在闲家视角的奖励
        reward = sc
        if isTer:
            reward+=info['fen']
            if env.players[firstPlayerID].isDealer():  #
                reward = -reward
            if env.sumSc >= 80:
                reward += (env.sumSc // 40 - 1) * 80
            elif env.sumSc > 0:
                reward += (env.sumSc // 40 - 2) * 80
            else:
                reward += -3 * 80
            reward += seqCardReward
        else:
            if env.players[firstPlayerID].isDealer():  #
                reward = -reward
            reward+=seqCardReward
        reward=reward/100
        if env.players[playerId].isDealer():
            reward = -reward
        return reward
    def setReward_Sparse(self,env,playerId,firstPlayerID, sc, isTer, info,seqCardReward):#系数奖励
        reward = sc
        if isTer:
            if env.sumSc>=80:
                reward=env.sumSc//40-1
            elif env.sumSc>0:
                reward=env.sumSc//40-2
            else:
                reward=-3
        else:
            reward=0
        # reward=reward/50
        if env.players[playerId].isDealer():
            reward=-reward
        return reward
    def playAGame(self):#玩一局游戏，同时要收集数据
        env=self.env
        env.dealCards()
        # env.coldeck(baselineColdeck)  # 换底牌，baselineColdeck是最基本的换底牌策略
        env.coldeck(baselineColdeck)  # 换底牌，使用神经网络k
        isTer = False
        roundId = 0
        firstPlayerID = env.dealer
        for i in range(4):
            self.agents[i].initMenory()
        while (not isTer):  # 开始出牌
            # env.printAllCards()
            # print("轮次：", epoch, "  先出牌玩家：", firstPlayerID)
            act = [None, None, None, None]
            allAct = [[], [], [], [], []]
            sortCardList1 = [[], [], [], [], []]
            sortCardList2 = [[], [], [], [], []]
            for i in range(4):
                sortCardList2[i] = env.players[i].toSortCardsList2(env)  # 会重叠
                sortCardList1[i] = env.players[i].toSortCardsList1(sortCardList2[i], env)  # 去重
            act[firstPlayerID] = self.firstPlayerPolicy(roundId,act,firstPlayerID,sortCardList1)  # 获取动作
            isSeq, canSeq = env.judgeSeqUse(act[firstPlayerID], firstPlayerID, sortCardList2)
            seqCardReward=0
            if isSeq and canSeq == False:  # 如果不能甩
                # print("不能甩！！！")
                if env.players[firstPlayerID].dealerTag < 2:  # 是庄家甩牌失败,闲家得10分
                    seqCardReward = 10
                    env.sumSc += 10
                else:
                    seqCardReward = -10
                    env.sumSc = max(0, env.sumSc - 10)
                act[firstPlayerID] = act[firstPlayerID].getMinCard(env)  # 强制出最小的组子合

            firstKind = env.getActKind(act[firstPlayerID])
            env.useCardsContainINF(roundId,env.players[firstPlayerID], act[firstPlayerID], firstKind, None,
                                   sortCardList2[firstPlayerID])
            # act[firstPlayerID].println()

            for i in range(1, 4):
                nextID = (firstPlayerID + i) % 4
                act[nextID] = Action()
                for a in act[firstPlayerID].one:
                    li = self.otherPlayerPolicy(roundId,i, act, nextID, firstKind, sortCardList2, [a])
                    act[nextID].add(li)
                for dou in act[firstPlayerID].double:
                    act[nextID].addDou(self.otherPlayerPolicy(roundId,i, act, nextID, firstKind, sortCardList2, dou))
                # act[nextID].println()
            firstPlayerID, sc, isTer, info = env.game_step(act, firstPlayerID)
            # 评价谁赢，返回赢者id,本轮分数(双方都会得分)，isTer是游戏有木有结束,不论谁赢，info都会包含底牌分数乘以倍数
            for i in range(4):
                self.agents[i].pushReward(roundId,self.rewardFun(env,i,firstPlayerID, sc, isTer, info,seqCardReward))
            # reset
            # env.printAllInfo(act)
            roundId += 1
            if isTer:
                # env.printUnderCards()
                winPlayer = env.getWinPlayer()  # 赢了的人
                # print(env.dealer,winPlayer, str(env.sumSc))
                playerId,grade = env.reset(env.sumSc)  # 重置游戏，-1代表继续,否则代表先达到A的玩家。grade大于等于0是庄家赢，否则是闲家赢
                # print(winPlayer)
                # print(playerId, "\n")
                return roundId,playerId,winPlayer
    def train_agent(self,  readName=None):
        # try:  # 加载保存的参数
        #     # print(shared_actor_model.state_dict())
        #     if shared_model!=None:
        #         for agent in self.agents:
        #             agent.net.load_state_dict(shared_model.state_dict())
        #     print("初始化完成", self.id)
        # except ZeroDivisionError:  # 处理异常
        #     print("共享参数异常")  # 输出错误原因
        # print(self.env.ini_borad)
        if readName!=None and len(readName)>0:
            self.readAgentNet("mod/"+readName+"/")#2023-03-24-15-44
        print("begin train!!!")

        pltDataLen=80
        xList=[i for i in range(pltDataLen)]
        yList=[0 for i in range(pltDataLen)]
        self.__trainCnt = 0
        self.env = CC()
        self.winPlayerSum=[0,0]#0是0和2赢，1是1和3赢。
        while True:
            # randomPlayGame(env)
            # while(True):
            self.env.reset_game()
            self.winPlayerSum = [0, 0]  # 0是0和2赢，1是1和3赢。
            while(True):
                roundUp,playerId,winPlayer = self.playAGame()#！=-1代表到A，本次游戏结束，返回赢得那个人。

                for i in range(4):
                    # self.agents[i].printMemory()
                    if self.trainlist[i]==1 and winPlayer%2==i%2:#如果这个智能体可以训练，就训练
                        rewardList=self.agents[i].memoryReward
                        self.agents[i].initLearn(roundUp,rewardList)
                        for j in range(0,roundUp,1):
                            self.agents[i].selflearn(j,roundUp,rewardList,self)
                        # self.agents[i].selflearn(roundUp-1, roundUp, rewardList,self)
                        # print(i)
                # break
                self.winPlayerSum[winPlayer % 2] += 1
                if (playerId != -1):
                    # print("先到到A的是:" + str(playerId % 2) + "," + str(playerId % 2 + 2))
                    break
            print("Number of games "+str(self.__trainCnt)+" : ",self.winPlayerSum)#谁赢了多少次
            yList[self.__trainCnt%pltDataLen]=self.winPlayerSum[0]-self.winPlayerSum[1]
            self.__trainCnt += 1
            if self.__trainCnt%pltDataLen==0:
                self.saveAgentNet()
                print("save net "+self.savePath)
            if self.__trainCnt%pltDataLen==0:
                drawBrokenLine(xList,yList,self.args.picName)
                print("draw pic "+self.args.picName+".jpg")
            # break

parser = argparse.ArgumentParser()
parser.add_argument('--isInitPar', type=bool, default=False)
parser.add_argument('--cudaIDList', type=list, default=[0])
parser.add_argument('--value_lr', type=float, default=0.0001)
parser.add_argument('--policy_lr', type=float, default=0.0001)
parser.add_argument('--learn_step', type=int, default=50)
parser.add_argument('--collection_length', type=int, default=1)
parser.add_argument('--dataUseCnt', type=int, default=5)
parser.add_argument('--env_name', type=str, default="tractor")
parser.add_argument('--trainlist', type=list, default=[2,1,2,1])#初始时训练那个人
parser.add_argument('--shape', type=tuple, default=(1,350))#初始时训练那个人
parser.add_argument('--gama', type=float, default=1)
parser.add_argument('--lanta', type=float, default=1)
parser.add_argument('--rewardFun', type=str, default="setReward")#setReward_Sparse
parser.add_argument('--picName', type=str, default="tdPic")

args = parser.parse_args()
print("cuda"+str(args.cudaIDList[0]))
worker=DppoWorkers(0,None,args)
worker.train_agent("")#2023-03-24-23-30

#trainlist[i]=2代表固定策略，0代表初始神经网络，1代表训练的神经网络。
#共用critic，但base分开
