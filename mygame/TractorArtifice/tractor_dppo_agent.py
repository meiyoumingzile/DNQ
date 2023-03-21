import argparse
import datetime
import os
import random

import numpy as np
import torch
import tractor_game
from baselinePolicy import baselineColdeck
from tractor_game import CC,Action,Player,dfsPrintActList
from encoder import handTo01code, getActionFeature, getBaseFea, actTo01code
from encoder import cardsTo01code
from tractor_network import AgentNet
import pyro
import pyro.distributions as dist


# start to define the workers...
from tractor_game import CC


class Dppoagent:
    def __init__(self,id,args,p:Player,worker):#代表这个智能体
        self.id=id
        self.args=args
        self.worker=worker
        in_fea=345#有待输入
        self.net = AgentNet().cuda(self.worker.cudaID)
        self.baseFea=None
        self.lock=False
        self.lanta = 0.5
        self.gamaPow = np.zeros((110))
        self.galanPow = np.zeros((110))#gama*lanta
        self.gamaPow[0]=self.galanPow[0]=1
        for i in range(1,110):
            self.gamaPow[i]=self.gamaPow[i-1]*args.gama
            self.galanPow[i]=self.galanPow[i-1]*args.gama*self.lanta
        self.initMenory()
        self.p=p
        self.CLIP_EPSL = 0.1
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.value_lr)
    def initMenory(self):#初始化memory,来记录本局游戏的数据
        self.memoryAllAct = [[] for i in range(25)]
        self.memoryProb = [[] for i in range(25)]#储存当时神经网络决策的动作和概率
        self.memoryBinProb = [[] for i in range(25)]
        self.memoryBase = [[] for i in range(25)]
        self.memoryOtherAct = [[] for i in range(25)]
        self.memoryCriticVal = [[] for i in range(25)]
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
        x=self.net.forward_q(base_out.cuda(self.worker.cudaID))
        self.memoryCriticVal[i].append(x.cpu())
        self.memoryBinProb[i].append(binact)

    def update_network(self, critic_shared_grad_buffer, actor_shared_grad_buffer,
                       shared_critic_model, shared_actor_model, critic_counter, actor_counter, traffic_signal):
        if self.lock:#如果该agent被锁定，则不更新参数。
            return
    def chooseFirstActFromMax2(self,env, otherAct,allActionList:list):#无脑出最大的
        return env.getActListMax(allActionList)
    def chooseFirstActFromMax1(self,env, otherAct,allActionList:list):#无脑出个数最多的
        maxLen = 0
        maxLenActList = []
        for a in allActionList:  # 找出长度最长的动作
            maxLen = max(maxLen, a.len)
        for i in range(len(allActionList)):  # 找出长度最长的动作的列表
            if maxLen == allActionList[i].len:
                maxLenActList.append(allActionList[i])
        act_i = random.randint(0, len(maxLenActList) - 1)
        return maxLenActList[act_i]
    def chooseFirstActFromRandom(self,env, otherAct,allActionList:list):#随机出牌
        act_i = random.randint(0, len(allActionList) - 1)
        return allActionList[act_i]

    def chooseFirstActFromNet(self,roundId,env, otherAct,allActionList:list):#从神经网络获取
        baseFea = getBaseFea(env,useSeq,selfId,isDealer)  # self.id
        otherActFea = getActionFeature(otherAct)
        base_out = self.net.forward_base(baseFea.cuda(self.worker.cudaID), otherActFea.cuda(self.worker.cudaID))  # base的特征向量
        # if np.random.uniform(0, 1) < self.epsl:
        #     ansAct = allActionList[np.random.randint(0, len(allActionList))]
        probList = torch.zeros((len(allActionList)))
        for i in range(len(allActionList)):  # 遍历所有动作
            nowActTensor = actTo01code(allActionList[i]).unsqueeze(dim=0)
            probList[i] = self.net.forward_act(base_out, nowActTensor.cuda(self.worker.cudaID)).view(-1)[0].cpu()
        probList = torch.softmax(probList, dim=0).cpu().detach()
        # print(probList)
        actid = torch.multinomial(probList, num_samples=1, replacement=True)[0].cpu().detach()
        self.pushMemory(roundId, [a.tolist() for a in allActionList],baseFea,base_out,otherActFea, (actid,probList[actid]) )
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
        baseFea = getBaseFea(self.worker.env)  # self.id
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
        actid = torch.multinomial(probList, num_samples=1, replacement=True)[0].cpu().detach()
        # self.otherAct[p.id].
        self.pushMemory(roundId, allActionList,baseFea,base_out, otherActFea,(actid, probList[actid]),(0,1))
        return actList[actid][0], actList[actid][1]
    def chooseOtherAct(self,roundId,env, otherAct, ansUp, ansDown, firstKind, sortCardList2):#ansUp代表较大的动作，ansDown代表较小的动作
        nu = len(ansUp)
        nd = len(ansDown)
        baseFea = getBaseFea(env).cuda(self.worker.cudaID)#self.id
        otherActFea = getActionFeature(otherAct).cuda(self.worker.cudaID)
        # print(actFea)
        base_out=self.net.forward_base(baseFea,otherActFea)#base的特征向量
        # print(base_out.shape)
        allActionList=ansUp
        fpprob=1
        if nu>0 and nd>0:
            fpprob=self.net.forward_bin(base_out).cpu().detach()#选择出大的还是小的,1是大，0是小
            # print(fpprob.shape)
            fp=torch.multinomial(fpprob, num_samples=1, replacement=True)[0].cpu().detach()
            fpprob=fpprob[0][fp]
            # print(fpprob)
            if fp==0:
                allActionList=ansDown
        elif nd>0:
            fp=0
            allActionList = ansDown
        else:
            fp=1
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
            env._useCardsContainINF(roundId,env.players[self.id], ansAct, firstKind, self.updateINF, sortCardList2[self.id])
        else:

            probList=torch.zeros((len(allActionList)))
            for i in range(len(allActionList)):#遍历所有动作
                nowActTensor=cardsTo01code(allActionList[i]).unsqueeze(dim=0)
                # print(nowActTensor.shape)
                probList[i]=self.net.forward_act(base_out,nowActTensor.cuda(self.worker.cudaID)).view(-1)[0].cpu()
            probList=torch.softmax(probList, dim=0).cpu().detach()
            try:
                actid = torch.multinomial(probList, num_samples=1, replacement=True)[0].cpu().detach()
            except ZeroDivisionError as e:
                print(probList)
            # actid = torch.multinomial(probList, num_samples=1, replacement=True)[0].cpu().detach()
            ansAct=allActionList[actid]
            self.pushMemory(roundId,allActionList,baseFea,base_out, otherActFea,(actid,probList[actid]),(fp,fpprob))
            env._useCardsContainINF(roundId,env.players[self.id], ansAct, firstKind, None, sortCardList2[self.id])
        return ansAct
    def calcGt(self,roundId,roundUp,rewardList):#用蒙特卡洛方法计算Gt
        ans=0
        for i in range(roundId,roundUp):
            ans+=self.gamaPow[i-roundId]*rewardList[i]
        return ans
    def initLearn(self,roundUp,rewardList):#使用GAE计算优势函数,要计算每一步的Gt-critic(state)
        self.roundInd=np.zeros((roundUp),dtype=np.int32)#roundId转换advantages的id
        self.detlaList = []
        G0 = self.calcGt(0, roundUp, rewardList)
        id=0
        for i in range(roundUp):##计算deala
            self.roundInd[i] = id
            id +=len(self.memoryCriticVal[i])
            for criTensor in self.memoryCriticVal[i]:
                q=criTensor[0][0].detach()
                # print(G0,q)
                self.detlaList.append(G0 - q)
            G0 -= rewardList[i]
            G0 /= self.args.gama

        self.advantages=[]#计算GAE优势函数
        adv0 = 0
        for i in range(len(self.detlaList)):
            adv0 +=self.galanPow[i]*self.detlaList[i]
        for i in range(len(self.detlaList)):
            self.advantages.append(adv0)
            adv0 -= self.detlaList[i]
            adv0 /= self.galanPow[1]
    def selflearn(self,roundId,roundUp,rewardList,worker):
        allActList=self.memoryAllAct[roundId]
        actPolicy=self.memoryProb[roundId]#储存当时神经网络决策的动作和概率
        actPolicy_bin=self.memoryBinProb[roundId]#当时玩家采取的二分类动作,有可能是None
        baseTensor=self.memoryBase[roundId]
        otherActTensor=self.memoryOtherAct[roundId]
        n=len(baseTensor)
        Gt=self.calcGt(roundId,roundUp,rewardList)

        for i in range(n):
            statei = baseTensor[i].cuda(self.worker.cudaID)
            otherActi=otherActTensor[i].cuda(self.worker.cudaID)
            # base_out = self.net.forward_base(statei,otherActi)
            # critic_q = self.net.forward_q(base_out)
            # tderro = Gt - critic_q  # tderro也是优势函数,根据具体问题可以在is_ter==True时阻断
            # advantage=tderro.detach()#普通优势函数
            advantage=self.advantages[self.roundInd[roundId]+i]#得到GAE优势
            # print(advantage)
            for _ in range(self.args.value_step):#训练critic
                self.net.zero_grad()
                base_out = self.net.forward_base(statei, otherActi)
                critic_q = self.net.forward_q(base_out)
                tderro = Gt - critic_q#因为自动作没有奖励，所以Q(s,a)都是Gt
                critic_loss = tderro.pow(2).mean()
                critic_loss.backward()#保存梯度
                self.optimizer.step()
            for _ in range(self.args.policy_step):#训练2个actor
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
                    actor_loss = -torch.min(surr1, surr2).mean()
                    actor_loss.backward()
                    self.optimizer.step()

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
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss.backward()
                self.optimizer.step()
class DppoWorkers:
    def __init__(self, id, shared_obs_state, args):
        self.id = id
        self.args = args
        self.cudaID=args.cudaIDList[0]
        self.env = CC()
        self.gama = 0.99
        self.CLIP_EPSL = 0.2
        self.lanta = 0.2
        self.shared_obs_state = shared_obs_state
        self.agents=(Dppoagent(0,args,self.env.players[0],self),Dppoagent(1,args,self.env.players[1],self),
                     Dppoagent(2,args,self.env.players[2],self),Dppoagent(3,args,self.env.players[3],self))
        self.trainlist=self.args.trainlist.copy()


        self.memory_batch = 50
        MEMORY_CAPACITY = 26
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        OVERLAY_CNT = 1  # 针对此游戏的叠帧操作
        self.OVERLAY_CNT = OVERLAY_CNT
        # beginShape = self.resetPic(self.env.reset()).shape

        self.memory_i = 0  # counter used for experience replay buffer//初始化记忆池
        # self.memoryQue = [#状态,BaseFeature,torch.zeros((MEMORY_CAPACITY,) + self.args.shape),
        #                   # torch.zeros((MEMORY_CAPACITY,4,54)),#动作编码，ActionFeature
        #                   torch.zeros((MEMORY_CAPACITY, 1)), #reward
        #
        #                     ]
        self.memoryReward=torch.zeros((MEMORY_CAPACITY)) #reward


    def initAllState(self):#初始化游戏状态，返回开始时的状态
        self.env.reset()
        return
    def getGameState(self):
        li=[handTo01code(self.env.players[i]) for i in range(4)]

        return li,self.env.getNowUsedCards()

    def pushMemory(self,reward):
        # baseFea = getBaseFea(self.env)  # self.id
        # actFea = getActionFeature(act)
        self.memoryReward[self.memory_i] =reward
        # self.memoryQue[1][self.memory_i]= actFea
        # self.memoryQue[2][self.memory_i]= reward
        self.memory_i+=1

    def saveAgentNet(self):
        now = datetime.datetime.now()
        now = str(now.strftime("%Y-%m-%d-%H"))
        now = now.replace(":", "-")
        path = "mod/" + now + "/"
        os.mkdir(path)
        for i in range(4):
            torch.save(self.agents[i].net, path+"agentNet"+str(i)+".pt")  #

    def firstPlayerPolicy(self,roundId,act,firstPlayerID,sortCardList1):  # 第一个人的出牌策略,需要神经网络决策,act是其他玩家出的牌，这里一定是空
        selfAllAct = self.env.getAllFirstAct(sortCardList1[firstPlayerID], self.env.players[firstPlayerID])
        # dfsPrintActList(selfAllAct)  # 输出先手动作集合
        ans=self.agents[firstPlayerID].chooseFirstActFromNet(roundId,self.env, act,selfAllAct)
        return ans
    def otherPlayerPolicy(self,roundId, act, nextID, firstKind, sortCardList2, cards):  # 使用的cards只能是单牌，对子，连对
        if len(cards) == 0:
            return None
        ansUp, ansDown, isHave = self.env.getAllAct(sortCardList2[nextID], self.env.players[nextID], cards)

        ans=self.agents[nextID].chooseOtherAct(roundId,self.env, act, ansUp, ansDown, firstKind, sortCardList2)
        return ans
    def setReward(self,env,firstPlayerID, sc, isTer, info,seqCardReward):#设置奖励函数，在闲家视角的奖励
        reward = sc
        if isTer:
            reward+=info['fen']
            if env.players[firstPlayerID].dealerTag < 2:  #
                reward = -reward
            reward += seqCardReward

            if env.sumSc < 80:  # 庄家获得胜利
                reward -= 80
            else:
                reward += 80
        else:
            if env.players[firstPlayerID].dealerTag < 2:  #
                reward = -reward
            reward+=seqCardReward
        reward=reward/10
        return reward
    def playAGame(self):#玩一局游戏，同时要收集数据
        env=self.env
        env.dealCards()
        # env.coldeck(baselineColdeck)  # 换底牌，baselineColdeck是最基本的换底牌策略
        env.coldeck(baselineColdeck)  # 换底牌，使用神经网络k
        isTer = False
        roundId = 0
        firstPlayerID = env.dealer
        self.memory_i=0
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
                    li = self.otherPlayerPolicy(roundId, act, nextID, firstKind, sortCardList2, [a])
                    act[nextID].add(li)
                for dou in act[firstPlayerID].double:
                    act[nextID].addDou(self.otherPlayerPolicy(roundId, act, nextID, firstKind, sortCardList2, dou))
                # act[nextID].println()
            firstPlayerID, sc, isTer, info = env.game_step(act, firstPlayerID)
            # 评价谁赢，返回赢者id,本轮分数(双方都会得分)，isTer是游戏有木有结束,不论谁赢，info都会包含底牌分数乘以倍数
            self.pushMemory(self.setReward(env,firstPlayerID, sc, isTer, info,seqCardReward))#奖励函数是根据闲家来的，在闲家视角，
            # reset
            # env.printAllInfo(act)
            roundId += 1
            if isTer:
                # env.printUnderCards()
                # print(firstPlayerID, str(env.sumSc))
                winPlayer = env.getWinPlayer()  # 赢了的人
                playerId,grade = env.reset(env.sumSc)  # 重置游戏，-1代表继续,否则代表先达到A的玩家。grade大于等于0是庄家赢，否则是闲家赢
                print(winPlayer)
                # print(playerId, "\n")
                return roundId,playerId,winPlayer
    def train_agent(self,  shared_model):
        try:  # 加载保存的参数


            # print(shared_actor_model.state_dict())
            if shared_model!=None:
                for agent in self.agents:
                    agent.net.load_state_dict(shared_model.state_dict())
            print("初始化完成", self.id)
        except ZeroDivisionError:  # 处理异常
            print("共享参数异常")  # 输出错误原因
        # print(self.env.ini_borad)
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
                    if self.trainlist[i]!=0:#如果这个智能体可以训练，就训练
                        rewardList=self.memoryReward.clone()
                        if self.env.players[i].isDealer():#是庄家
                            rewardList=-rewardList
                        self.agents[i].initLearn(roundUp,rewardList)
                        for j in range(0,roundUp,1):
                            self.agents[i].selflearn(j,roundUp,rewardList,self)
                        # self.agents[i].selflearn(roundUp-1, roundUp, rewardList,self)
                        # print(i)
                # break
                self.winPlayerSum[winPlayer % 2] += 1
                if (playerId != -1):
                    print("先到到A的是:" + str(playerId % 2) + "," + str(playerId % 2 + 2))
                    break
            print(self.winPlayerSum)#谁赢了多少次
            # break


parser = argparse.ArgumentParser()
parser.add_argument('--isInitPar', type=bool, default=False)
parser.add_argument('--cudaIDList', type=list, default=[1])
parser.add_argument('--value_lr', type=float, default=0.0001)
parser.add_argument('--policy_lr', type=float, default=0.0001)
parser.add_argument('--learn_step', type=int, default=50)
parser.add_argument('--collection_length', type=int, default=1)
parser.add_argument('--value_step', type=int, default=3)
parser.add_argument('--policy_step', type=int, default=3)
parser.add_argument('--env_name', type=str, default="tractor")
parser.add_argument('--trainlist', type=list, default=[1,0,1,0])#初始时训练那个人
parser.add_argument('--shape', type=tuple, default=(1,345))#初始时训练那个人
parser.add_argument('--gama', type=float, default=1)
args = parser.parse_args()
worker=DppoWorkers(0,None,args)
worker.train_agent(None)