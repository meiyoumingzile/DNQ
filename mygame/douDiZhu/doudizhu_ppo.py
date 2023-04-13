import argparse
import datetime
import math
import os
import random
import time
from collections import deque
from functools import cmp_to_key

import numpy as np
import torch
from torch import nn

import doudizhu_game
from doudizhu_utils import getNowTimePath, drawBrokenLine
from doudizhu_game import Doudizhu,Action,Player,dfsPrintActList
from doudizhu_encoder import handTo01code, getAllActionFea, getBaseFea, actTo01code, actZeros
from doudizhu_network import AgentNet, AgentCriNet
import pyro
import pyro.distributions as dist


INFEA=330

class Dppoagent:
    def __init__(self,id,args,worker):#代表这个智能体
        self.id=id
        self.args=args
        self.worker=worker
        self.env=worker.env
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
    def setNet(self,actor_net,critic_net):#设置网络
        self.net = actor_net
        self.critic_net = critic_net

    def initPar(self,roundId,maxact):
        self.maxact=maxact
        self.roundId=roundId
        if roundId==0:
            self.hisActTensor =torch.zeros((1,1,57)).cuda(self.worker.cudaID)
        else:
            self.hisActTensor=torch.stack(self.worker.historyActionList,dim=0).unsqueeze(dim=0).cuda(self.worker.cudaID)
    def initMenory(self):#初始化memory,来记录本局游戏的数据
        self.memoryRound = []
        self.memoryAllAct=[]
        self.memoryBaseFea = []
        self.memoryReward = []
        self.memoryAct1 = []
        self.memoryAct2 = []
        self.memoryHisFea=[]
    def printMemory(self):
        dfsPrintActList(self.memoryAllAct)
        print(self.memoryRound)
        print(self.memoryReward)
        print(self.memoryAct1)
        print(self.memoryAct2)
    def pushMemory(self,roundId,allActList,baseFea,hisFea,reward,act1,act2):#allAct内部是多个list
        self.memoryRound.append(roundId)
        self.memoryAllAct.append(allActList)
        self.memoryBaseFea.append(baseFea)
        self.memoryReward.append(reward)
        self.memoryHisFea.append(hisFea)
        self.memoryAct1.append(act1)
        self.memoryAct2.append(act2)

    def update_network(self, critic_shared_grad_buffer, actor_shared_grad_buffer,
                       shared_critic_model, shared_actor_model, critic_counter, actor_counter, traffic_signal):
        if self.lock:#如果该agent被锁定，则不更新参数。
            return
    def chooseFirstActFromMax2(self,roundId,env, otherAct,allActionList:list):#无脑出最大的
        return env.getActListMax(allActionList)
    def chooseFirstActFromMax1(self,roundId,env, nowPlayerId,maxPlayerid,maxact,allActionList:list):#无脑出个数最多的
        n = len(allActionList)
        allActionList.sort(key=cmp_to_key(doudizhu_game._sortAction_cmp1))
        newList = []
        prelen = -1
        for a in allActionList:
            if a.len != prelen:
                newList.append(a)
                prelen = a.len
        id = random.randint(0, len(newList) - 1)
        env.players[nowPlayerId].useAction(newList[id], doudizhu_game.baselinePolicy_INFfun)
        return newList[id]
    def chooseFirstActFromRandom(self,roundId,env, otherAct,allActionList:list,firstPlayerID):#随机出牌
        act_i = random.randint(0, len(allActionList) - 1)
        return allActionList[act_i]
    def checkProb(self,probList):
        # print(len(probList))
        for i in range(len(probList)):
            if probList[i]<0:
                print(probList)
                probList[i]=0
    def chooseFirstActFromNet(self,roundId,env, nowPlayerId,maxPlayerid,maxact,allActionList:list):#从神经网络获取动作
        baseFea = getBaseFea(env,nowPlayerId,maxact).cuda(self.worker.cudaID)  # self.id
        allAct_kind = env.classActKind(allActionList)

        # print(self.worker.cudaID,next(self.net.parameters()).device)
        hisActFea=self.hisActTensor
        probList1= self.net.forward_fp(baseFea, hisActFea).squeeze(dim=0)  # base的特征向量
        for i in range(len(allAct_kind)):
            if len(allAct_kind[i])==0:
                probList1[i]=-math.inf
        probList1 = torch.softmax(probList1, dim=0)
        actid_1 = torch.multinomial(probList1, num_samples=1, replacement=True)[0].item()
        actid_1_prob=probList1[actid_1].item()

        if len(allAct_kind[actid_1])<=1:
            actid_2 = 0
            actid_2_prob =1
        else:
            # dfsPrintActList(allAct_kind[actid_1])
            allActFea = getAllActionFea(allAct_kind[actid_1]).cuda(self.worker.cudaID)
            probList2 = self.net.forward_act(baseFea, hisActFea, allActFea).squeeze(dim=1)
            # dfsPrintActList(allAct_kind[actid_1])
            # print(probList2.shape)
            actid_2 = torch.multinomial(probList2, num_samples=1, replacement=True)[0].item()
            actid_2_prob = probList2[actid_2].item()
        act=allAct_kind[actid_1][actid_2]
        self.pushMemory(roundId,allActionList,baseFea,hisActFea,0,(actid_1,actid_1_prob),(actid_2,actid_2_prob))
        self.env.players[nowPlayerId].useAction(act, self.updateINF)
        return act

    def updateINF(self, cardi, appendix, allActList):
        baseFea = getBaseFea(self.env, self.id, self.maxact).cuda(self.worker.cudaID)  # self.id
        hisActFea = self.hisActTensor

        allActFea = getAllActionFea(allActList).cuda(self.worker.cudaID)
        probList2 = self.net.forward_act(baseFea, hisActFea, allActFea).squeeze(dim=1).cpu().detach()
        actid_2 = torch.multinomial(probList2, num_samples=1, replacement=True)[0].item()
        actid_2_prob = probList2[actid_2].item()

        # dfsPrintActList(allActList)
        self.pushMemory(self.roundId,allActList, baseFea,hisActFea, 0, None, (actid_2, actid_2_prob))
        return actid_2

    def chooseOtherActFromNet(self,roundId,env, nowPlayerid,maxPlayerid,maxact,allActionList:list):#ansUp代表较大的动作，ansDown代表较小的动作
        baseFea = getBaseFea(env,nowPlayerid,maxact).cuda(self.worker.cudaID)  # self.id
        allAct_kind = env.classActKind(allActionList)

        # print(self.worker.cudaID,next(self.net.parameters()).device)
        hisActFea=self.hisActTensor
        probList1= self.net.forward_fp(baseFea, hisActFea).squeeze(dim=0)  # base的特征向量
        for i in range(len(allAct_kind)):
            if len(allAct_kind[i])==0:
                probList1[i]=-math.inf
        probList1 = torch.softmax(probList1, dim=0)
        actid_1 = torch.multinomial(probList1, num_samples=1, replacement=True)[0].item()
        actid_1_prob=probList1[actid_1].item()

        if len(allAct_kind[actid_1])<=1:
            actid_2 = 0
            actid_2_prob =1
        else:
            allActFea = getAllActionFea(allAct_kind[actid_1]).cuda(self.worker.cudaID)
            probList2 = self.net.forward_act(baseFea, hisActFea, allActFea).squeeze(dim=1).cpu().detach()
            actid_2 = torch.multinomial(probList2, num_samples=1, replacement=True)[0].item()
            actid_2_prob = probList2[actid_2].item()
        act=allAct_kind[actid_1][actid_2]
        self.pushMemory(roundId,allActionList,baseFea,hisActFea,0,(actid_1,actid_1_prob),(actid_2,actid_2_prob))
        self.env.players[nowPlayerid].useAction(act, self.updateINF)
        return act

    def chooseOtherAct_fix(self,roundId,env, nowPlayerid,maxPlayerid,maxact,allActionList:list):#固定策略，固定选取第0个
        id = 0
        if (env.players[maxPlayerid].dealerTag * env.players[nowPlayerid].dealerTag) == 0 and len(allActionList) > 1:
            id = random.randint(1, len(allActionList) - 1)
        env.players[nowPlayerid].useAction(allActionList[id], doudizhu_game.baselinePolicy_INFfun)
        return allActionList[id]
    def initLearn(self,roundUp):#
        statei = self.memoryBaseFea[0]
        hisActi = self.memoryHisFea[0]
        self.detlaList=[]
        prev = self.critic_net.forward(statei, hisActi)
        up=len(self.memoryBaseFea)
        for i in range(1, up):#calc detla
            statei = self.memoryBaseFea[i]
            hisActi = self.memoryHisFea[i]
            reward = self.memoryReward[i-1]
            v = self.critic_net.forward(statei, hisActi)
            detla=(reward + v * self.gamaPow[1] - prev).view(-1).detach()
            self.detlaList.append(detla.clone())
            prev = v
        reward = self.memoryReward[up- 1]
        detla = (reward - prev).view(-1).detach()
        self.detlaList.append(detla.clone())

        adv0 = 0
        self.advantages =torch.zeros((up,1)).cuda(self.worker.cudaID)
        for i in range(up):
            adv0 += self.galanPow[i] * self.detlaList[i]
        for i in range(up):
            self.advantages[i]=adv0.clone()
            adv0 -= self.detlaList[i]
            adv0 /= self.galanPow[1]
        mean = self.advantages.mean()
        std = self.advantages.std()
        self.advantages = (self.advantages - mean) / std
        # print()
        # print(self.advantages)
        # print(self.memoryReward)
        # exit()
    def selflearn(self,roundUp,worker):
        up = len(self.memoryBaseFea)
        for _ in range(self.args.dataUseCnt):  # 训练critic
            sumLoss_critic = []
            sumLoss_actor1 = []
            sumLoss_actor2 = []
            # self.memoryRound.append(roundId)
            # self.memoryAllAct.append(allActList)
            # self.memoryBaseFea.append(baseFea)
            # self.memoryReward.append(reward)
            # self.memoryHisFea.append(hisFea)
            # self.memoryAct1.append(act1)
            # self.memoryAct2.append(act2)
            for i in range(up):
                statei = self.memoryBaseFea[i]
                reward=self.memoryReward[i]
                actPolicy1= self.memoryAct1[i]#actid_1,actid_1_prob
                actPolicy2=self.memoryAct2[i]#actid_2,actid_2_prob
                hisActi = self.memoryHisFea[i]
                allAct_kind=self.env.classActKind(self.memoryAllAct[i])
                if i==up-1:
                    next_statei = None
                else:
                    next_statei = self.memoryBaseFea[i+1]
                    next_hisActi = self.memoryHisFea[i+1]
                advantage=self.advantages[i].clone().detach()
                v = self.critic_net.forward(statei,hisActi)
                if next_statei!=None:
                    next_v=self.critic_net.forward(next_statei,next_hisActi)
                    tderro = reward+self.gamaPow[1]*next_v - v#因为自动作没有奖励，所以Q(s,a)都是Gt
                else:
                    tderro= reward - v
                critic_loss = tderro.pow(2).mean()
                sumLoss_critic.append(critic_loss)
                # critic_loss.backward()
                # self.optimizer[0].step()
                # 下面是更新决策二分类网络
                if actPolicy1!=None:
                    nowRate = self.net.forward_fp(statei,hisActi)
                    actid=torch.Tensor([[actPolicy1[0]]]).cuda(self.worker.cudaID).long()
                    for i in range(len(allAct_kind)):#遮罩
                        if len(allAct_kind[i]) == 0:
                            nowRate[0][i] = -math.inf
                    # print(nowRate)
                    nowRate=torch.softmax(nowRate,dim=1)
                    # print(nowRate)
                    nowRate=nowRate.gather(1, actid)
                    # print(actPolicy1[0],nowRate)
                    preRate =torch.Tensor([[actPolicy1[1]]]).cuda(self.worker.cudaID)#旧策略
                    # print(nowRate.shape,preRate.shape)
                    ratio = torch.exp(torch.log(nowRate) - torch.log(preRate))  # 计算p1/p2防止精度问题
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.CLIP_EPSL, 1 + self.CLIP_EPSL) * advantage
                    actor_loss_bin = (-torch.max(torch.min(surr1, surr2),( 1 - self.CLIP_EPSL)*advantage)-self.args.entropy_coef*torch.log(nowRate)).mean()
                    sumLoss_actor1.append(actor_loss_bin)
                    allActFea = getAllActionFea(allAct_kind[actPolicy1[0]]).cuda(self.worker.cudaID)
                else:
                    allActFea=getAllActionFea(self.memoryAllAct[i]).cuda(self.worker.cudaID)
                #下面是更新act网络

                probList = self.net.forward_act(statei,hisActi,allActFea)
                act = torch.Tensor([[actPolicy2[0]]]).cuda(self.worker.cudaID).long()
                # print(probList)
                nowRate = probList.gather(0, act)
                # print(act,nowRate)
                preRate = torch.Tensor([[actPolicy2[1]]]).cuda(self.worker.cudaID)  # 旧策略
                # print(nowRate.shape,preRate.shape)
                ratio = torch.exp(torch.log(nowRate) - torch.log(preRate))  # 计算p1/p2防止精度问题
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.CLIP_EPSL, 1 + self.CLIP_EPSL) * advantage
                actor_loss = (-torch.max(torch.min(surr1, surr2),( 1 - self.CLIP_EPSL)*advantage)-self.args.entropy_coef*torch.log(nowRate)).mean()
                sumLoss_actor2.append(actor_loss)
                # print(loss_actor,actor_loss)
                # actor_loss.backward()
                # self.optimizer[2].step()
            self.updateNetwork(sumLoss_critic,sumLoss_actor1,sumLoss_actor2)
    def updateNetwork(self,sumLoss_critic,sumLoss_actor1,sumLoss_actor2):
        self.critic_net.zero_grad()
        sumLoss_critic=torch.stack(sumLoss_critic).mean()
        sumLoss_critic.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 3)
        # self.critic_net.printGrad()
        self.opt_cri.step()

        loss=0
        self.net.zero_grad()
        if len(sumLoss_actor1)>0:
            loss = torch.stack(sumLoss_actor1).mean()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3)

        loss = loss+torch.stack(sumLoss_actor2).mean()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3)
        # self.net.printGrad()
        self.opt[0].step()
class DppoWorkers:
    def __init__(self, id, shared_obs_state, args):
        self.id = id
        self.args = args
        self.cudaID=args.cudaIDList[id]

        self.gama = args.gama
        self.CLIP_EPSL = 0.2
        self.shared_obs_state = shared_obs_state
        self.env = Doudizhu()
        self.agents = (Dppoagent(0, args, self), Dppoagent(1, args, self),Dppoagent(2, args, self))
        self.trainlist=self.args.trainlist
        self.savePath = getNowTimePath()

        self.memory_batch = 50
        MEMORY_CAPACITY = 26
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        OVERLAY_CNT = 1  # 针对此游戏的叠帧操作
        self.OVERLAY_CNT = OVERLAY_CNT
        self.rewardFun=getattr(self, self.args.rewardFun)
        # beginShape = self.resetPic(self.env.reset()).shape
    def initNet(self,path=""):#如果path存在,就是读取
        self.actor_net = [AgentNet(INFEA).cuda(self.cudaID) for i in range(3)]#0是地主，1是地主下家，2是地主上家

        self.critic_net_landlord = AgentCriNet(INFEA).cuda(self.cudaID)
        self.critic_net_peasant  = AgentCriNet(INFEA).cuda(self.cudaID)

        if path=="" or not os.path.exists(path):
            print("模型不存在")
        else:
            p0 = path + "critic_net_landlord.pt"
            self.critic_net_landlord = torch.load(p0).cuda(self.cudaID)
            p0 = path + "critic_net_peasant.pt"
            self.critic_net_peasant=torch.load(p0).cuda(self.cudaID)
            for i in range(3):
                p = path + "agentNet" + str(i) + ".pt"
                if not os.path.exists(p):
                    print("模型不存在")
                self.actor_net[i]= torch.load(p).cuda(self.cudaID)
            print("read success")
        self.critic_net_landlord.train()
        self.critic_net_peasant.train()
        for i in range(3):
            self.actor_net[i].train()

        self.optimizer_critic_landlord = torch.optim.Adam(self.critic_net_landlord.parameters(),lr=self.args.value_lr)
        self.optimizer_critic_peasant = torch.optim.Adam(self.critic_net_peasant.parameters(), lr=self.args.value_lr)
        self.optimizer=[]
        for i in range(3):
            # baseNet = list(self.actor_net[i].mlp_base.parameters()) + list(self.actor_net[i].lstm1.parameters())
            self.optimizer.append([torch.optim.Adam(self.actor_net[i].parameters(), lr=self.args.policy_lr)])
    def setNet(self,pid):#把网络给对应智能体
        self.agents[pid].setNet(self.actor_net[0],self.critic_net_landlord)
        self.agents[(pid+1)%3].setNet(self.actor_net[1],self.critic_net_peasant)
        self.agents[(pid+2)%3].setNet(self.actor_net[2],self.critic_net_peasant)

        self.agents[pid].opt_cri=self.optimizer_critic_landlord
        self.agents[pid].opt = self.optimizer[0]
        self.agents[(pid+1)%3].opt_cri = self.optimizer_critic_peasant
        self.agents[(pid+1)%3].opt = self.optimizer[1]
        self.agents[(pid+2)%3].opt_cri = self.optimizer_critic_peasant
        self.agents[(pid+2)%3].opt = self.optimizer[2]


    def saveAgentNet(self):
        path=self.savePath
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.critic_net_landlord, path + "critic_net_landlord.pt")  #
        torch.save(self.critic_net_peasant, path + "critic_net_peasant.pt")  #
        for i in range(3):
            torch.save(self.actor_net[i], path+"agentNet"+str(i)+".pt")  #


    def playerPolicy(self,roundId,scList,nowPlayerId,maxPlayerid,maxact,allActionList:list):  # 第一个人的出牌策略,需要神经网络决策,act是其他玩家出的牌，这里一定是空
        # maxact是0张量代表首先出牌
        self.agents[nowPlayerId].initPar(roundId,maxact)
        isDealer=(nowPlayerId!=self.env.dealer)
        if maxPlayerid==-1:#先出的人
            if self.args.trainlist[isDealer]==2:#固定策略
                # dfsPrintActList(allActionList)
                act:Action=self.agents[nowPlayerId].chooseFirstActFromMax1(roundId,self.env, nowPlayerId,maxPlayerid,maxact,allActionList)
                # dfsPrintActList(self.env.players[nowPlayerid].cards)
                # act.print()
            elif self.args.trainlist[isDealer]<2:
                # print(firstPlayerID)
                act=self.agents[nowPlayerId].chooseFirstActFromNet(roundId, self.env, nowPlayerId, maxPlayerid, maxact, allActionList)
            else:
                exit()
        else:
            if self.args.trainlist[isDealer]==2:#固定策略
                act=self.agents[nowPlayerId].chooseOtherAct_fix(roundId, self.env, nowPlayerId, maxPlayerid, maxact, allActionList)

            elif self.args.trainlist[isDealer]<2:
                # print(firstPlayerID)
                act=self.agents[nowPlayerId].chooseOtherActFromNet(roundId, self.env, nowPlayerId, maxPlayerid, maxact, allActionList)
            else:
                exit()
        if act.isBomb():
            for i in range(3):
                scList[i]*=2
        return act

    def setReward_Sparse(self,pid,maxPlayerId, scList,winPlayer):#系数奖励
        if winPlayer!=-1:
            # self.env.printPlayerHand()
            # print(winPlayer,self.env.dealer)
            for i in range(3):
                nowid=(winPlayer+i)%3
                k=len(self.agents[nowid].memoryReward)-1
                if k>0:
                    self.agents[nowid].memoryReward[k]=scList[nowid]
                # print(self.agents[nowid].memoryReward)
    def playAGame(self):#玩一局游戏，同时要收集数据
        env=self.env
        env.reset()
        # deck,dealer=mkDeck(cheat1)
        # env.dealCards(deck, dealer)
        env.dealCards(None,0)
        roundId = 0
        self.setNet(env.dealer)
        scList = [1, 1, 1]  # 玩家倍数
        scList[env.dealer] = 2
        maxPlayerId = env.dealer
        for i in range(3):
            self.agents[i].initMenory()
        self.historyActionList = []
        while(True):  # 开始出牌
            # env.printAllCards()
            # print("轮次：", epoch, "  先出牌玩家：", maxPlayerID)
            allAct = env.players[maxPlayerId].getAllFirstAction()
            maxact = self.playerPolicy(roundId,scList,maxPlayerId,-1,None,allAct)  #在这里面存入buffer
            roundId += 1
            self.historyActionList.append(actTo01code(maxact))
            pid = maxPlayerId
            winPlayer = -1
            que = deque(maxlen=2)
            while(True):
                # for i in range(3):
                #     ans = env.players[i].getAllFirstAction()
                #     dfsPrintActList(ans)
                pid = (pid + 1) % 3
                allAct = env.players[pid].getAllAction(maxact)
                act = self.playerPolicy(roundId,scList,pid,maxPlayerId,maxact,allAct)
                roundId += 1
                self.historyActionList.append(actTo01code(act))
                maxPlayerId, maxact, winPlayer = env.step(maxact, act, maxPlayerId, pid)
                if winPlayer != -1:
                    if env.dealer == winPlayer:
                        scList[(env.dealer + 1) % 3] = -scList[(env.dealer + 1) % 3]
                        scList[(env.dealer + 2) % 3] = -scList[(env.dealer + 2) % 3]
                    else:
                        scList[env.dealer] = -scList[env.dealer]
                self.setReward_Sparse(pid,maxPlayerId, scList,winPlayer)
                que.append(act)
                if winPlayer != -1 or len(que) == 2 and que[0].isPass() and que[-1].isPass():
                    break
            # reset
            # env.printAllInfo(act)
            isTer= winPlayer != -1
            if isTer:
                # print(scList)
                # print(winPlayer)
                # exit()
                return roundId,winPlayer,scList
    def train_agent(self,  readName=None):
        if readName!=None and len(readName)>0:
            self.initNet("mod/"+readName+"/")
        else:
            self.initNet()
        print("begin train!!!")
        epoch_batchsize=100
        pltDataMaxLen=10000
        pltDataLen=self.args.pltDataLen
        intv=self.args.rewardInterval
        xList = []
        y1List = []
        y2List = []
        self.__trainCnt = 0
        env=self.env
        self.winPlayerSum=[0,0]#0是0和2赢，1是1和3赢。
        self.playerGradeSum=[0,0]
        start_time = time.time()
        ma30=deque(maxlen=30)
        while True:
            winsum=[0,0]
            for _ in range(epoch_batchsize):
                roundUp,winPlayer,scList = self.playAGame()#！=-1代表到A，本次游戏结束，返回赢得那个人。

                if self.trainlist[0]==1:#训练地主
                    if self.trainlist[self.env.dealer]==1:#如果这个智能体可以训练，就训练
                        self.agents[self.env.dealer].initLearn(roundUp)
                        self.agents[self.env.dealer].selflearn(roundUp,self)
                if  self.trainlist[1]==1:#训练农民
                    for i in range(1,3):
                        pid=(self.env.dealer+i)%3
                        self.agents[pid].initLearn(roundUp)
                        self.agents[pid].selflearn(roundUp, self)
                # break
                self.winPlayerSum[winPlayer!=env.dealer] += 1
                winsum[winPlayer!=env.dealer] += 1
                self.playerGradeSum[0]+=scList[env.dealer]#z
                self.playerGradeSum[1]+=scList[(env.dealer+1)%3]*2

            print("Number of games "+str(self.__trainCnt)+" : ",winsum)#谁赢了多少次
            self.__trainCnt += 1
            if self.__trainCnt%pltDataLen==0:
                self.saveAgentNet()
                print("save net "+self.savePath)
            if self.__trainCnt % pltDataLen == 0:
                drawBrokenLine(xList, y1List, self.args.picName + "_wp", "epoch",
                               "The proportion of landowners winning in one epoch")
                drawBrokenLine(xList, y2List, self.args.picName + "_adp", "epoch",
                               "Addition the score of the landlord from the score of the farmer")
                end_time = time.time()
                print("draw pic " + self.args.picName + "_wp.jpg")
                print("Program execution time: " + str(end_time-start_time) )
                start_time=end_time
            if self.__trainCnt % (pltDataMaxLen*intv) == 0:
                xList ,y1List,y2List = [],[],[]
                print("reset xlist!!")
            if self.__trainCnt%intv==0:
                xList.append(self.__trainCnt//intv)
                rate=self.winPlayerSum[0] / (self.winPlayerSum[0] + self.winPlayerSum[1])
                ma30.append(rate)
                y1List.append(rate)
                y2List.append((self.playerGradeSum[0] + self.playerGradeSum[1])/intv)
                self.winPlayerSum = [0, 0]  # 0是0和2赢，1是1和3赢。
                self.playerGradeSum = [0, 0]
            if len(ma30)==30 and (self.trainlist[0]==1 and ma30[0]>0.65 or self.trainlist[1]==1 and ma30[0]<0.35):#ma30大于0.6
                xList, y1List, y2List = [], [], []
                ma30 = deque(maxlen=30)
                self.trainlist[0],self.trainlist[1]=self.trainlist[1],self.trainlist[0]
                print("swap train!!")
            # break

parser = argparse.ArgumentParser()
parser.add_argument('--isInitPar', type=bool, default=False)
parser.add_argument('--cudaIDList', type=list, default=[1])
parser.add_argument('--value_lr', type=float, default=0.0001)
parser.add_argument('--policy_lr', type=float, default=0.0001)
parser.add_argument('--entropy_coef', type=float, default=0.0001)
parser.add_argument('--learn_step', type=int, default=50)
parser.add_argument('--collection_length', type=int, default=1)
parser.add_argument('--dataUseCnt', type=int, default=1)
parser.add_argument('--env_name', type=str, default="tractor")
parser.add_argument('--trainlist', type=list, default=[0,1])#初始时训练哪个智能体，0代表地主，1代表农民
parser.add_argument('--shape', type=tuple, default=(1,350))#初始时训练那个人
parser.add_argument('--gama', type=float, default=1)
parser.add_argument('--lanta', type=float, default=1)
parser.add_argument('--rewardInterval', type=int, default=1)
parser.add_argument('--rewardFun', type=str, default="setReward_Sparse")#setReward_Sparse
parser.add_argument('--picName', type=str, default="tdPic")
parser.add_argument('--pltDataLen', type=int, default=200)

args = parser.parse_args()
print("cuda:"+str(args.cudaIDList[0]))
worker=DppoWorkers(0,None,args)
worker.train_agent("2023-04-13-18-47")#2023-04-13-18-47

#trainlist[i]=2代表固定策略，0代表初始神经网络，1代表训练的神经网络。
#共用critic，但base分开
