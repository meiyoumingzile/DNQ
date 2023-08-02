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

from otherPolicyBaseline import getModel
from graphDou.graphDou_codeParameter import INFEA, actionSpaceType
from graphDou.graphDou_ppo_agent import Dppoagent
from doudizhu_cheat import mkDeck, cheat1
from doudizhu_utils import getNowTimePath, drawBrokenLine, MACD, getNowTime
from doudizhu_game import Doudizhu,Action,Player,dfsPrintActList
from graphDou.graphDou_encoder import actTo01code, cardsTo01code
from graphDou.graphDou_network import AgentNet, AgentCriNet
import pyro
import pyro.distributions as dist



def updateNetwork_critic(agent,worker, sumLoss_critic):
    if len(sumLoss_critic)!=0:
        agent.critic_net.zero_grad()
        sumLoss_critic = torch.stack(sumLoss_critic).mean()
        sumLoss_critic.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_net.parameters(), 1)
        # if worker.epoch_i == 0:
        # agent.critic_net.printGrad()
        agent.opt_cri.step()

def updateNetwork_actor(agent,worker, sumLoss_actor1, sumLoss_actor2):
    if len(sumLoss_actor2)!=0:
        print("trainNet step " + str(worker._playCnt), len(sumLoss_actor2))
        agent.net.zero_grad()
        loss = torch.stack(sumLoss_actor2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.net.parameters(), 1)
        # if worker._trainCnt%100==0 and worker.isPrint:#定期打印梯度
        #     worker.isPrint=False
        # agent.net.printGrad()
        agent.opt.step()
class DppoWorkers:
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.cudaID=args.cudaIDList[id]
        self._trainCnt = 0
        self._playCnt = 0
        self.gama = args.gama
        self.env = Doudizhu()
        self.agents = (Dppoagent(0, args, self), Dppoagent(1, args, self),Dppoagent(2, args, self))
        self.policykind=self.args.policykind
        self.istrain=self.args.istrain
        self.savePath = getNowTimePath()
        self.saveTime=getNowTime()
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
            # baseNet1 = list(self.actor_net[i].mlp_base.parameters()) + list(self.actor_net[i].lstm1.parameters())
            # baseNet2 = list(self.actor_net[i].mlp_base.parameters()) + list(self.actor_net[i].lstm2.parameters())\
            #           + list(self.actor_net[i].mlp_act2.parameters())+ list(self.actor_net[i].lstm1.parameters())
            self.optimizer.append(torch.optim.Adam(self.actor_net[i].parameters(), lr=self.args.policy_lr))

        #下面加载baseline的模型
        douzero_models = getModel("douzero", self.cudaID)
        douzeroResnet_models=getModel("douzeroresnet", self.cudaID)
        perfectdou_models = getModel("perfectdou", self.cudaID)
        for i in range(3):
            self.agents[i].douzero_models=douzero_models
            self.agents[i].douzeroResnet_models = douzeroResnet_models
            self.agents[i].perfectdou_models = perfectdou_models

    def setNet(self,pid=0):#把网络给对应智能体
        self.agents[pid].setNet(self.actor_net[0],self.critic_net_landlord)
        self.agents[(pid+1)%3].setNet(self.actor_net[1],self.critic_net_peasant)
        self.agents[(pid+2)%3].setNet(self.actor_net[2],self.critic_net_peasant)

        self.agents[pid].opt_cri=self.optimizer_critic_landlord
        self.agents[pid].opt = self.optimizer[0]
        self.agents[(pid+1)%3].opt_cri = self.optimizer_critic_peasant
        self.agents[(pid+1)%3].opt = self.optimizer[1]
        self.agents[(pid+2)%3].opt_cri = self.optimizer_critic_peasant
        self.agents[(pid+2)%3].opt = self.optimizer[2]

        for i in range(3):
            agent=self.agents[(pid+i)%3]
            agent.setPlayer(self.env.players[(self.env.dealer+i)%3])#0是庄家
            agent.beginCardsFea=cardsTo01code(agent.player.beginCards)

    def saveAgentNet(self):
        path=self.savePath
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.critic_net_landlord, path + "critic_net_landlord.pt")  #
        torch.save(self.critic_net_peasant, path + "critic_net_peasant.pt")  #
        for i in range(3):
            torch.save(self.actor_net[i], path+"agentNet"+str(i)+".pt")  #


    def playerPolicy(self,roundId,nowAgent,maxAgent,preActQue,allActionList:list):  # 第一个人的出牌策略,需要神经网络决策,act是其他玩家出的牌，这里一定是空
        # maxact是0张量代表首先出牌
        nowAgent.initPar(roundId,preActQue)
        # print(roundId)
        isDealer=(nowAgent.player.id!=self.env.dealer)
        k=self.args.policykind[isDealer]
        act=None
        if k in nowAgent.actPolicyFun:  # 选择策略
            # dfsPrintActList(allActionList)
            act: Action = nowAgent.actPolicyFun[k][self.istrain[isDealer]](roundId, self.env, maxAgent, preActQue, allActionList)
            # act.print()
        return act

    def setReward_Sparse(self,roundEnd, scList,winAgentId):#稀疏奖励
        # print(scList,winAgentId)
        if winAgentId!=-1:
            # self.env.printPlayerHand()
            # print(winPlayer,self.env.dealer)
            for i in range(3):

                nowid=(winAgentId+i)%3
                k=len(self.agents[nowid].memoryReward)-1
                sc = scList[self.agents[nowid].player.id]
                if k>=0 and sc!=0 and self.istrain[self.agents[i].player.id!=self.env.dealer]==1:
                    self.agents[nowid].memoryReward[k]=(math.log2(abs(sc))+1)*(np.sign(sc))/2.3*3
                    # self.agents[nowid].memoryReward[k] =sc/50
                # print(self.agents[nowid].player.id,self.agents[nowid].memoryReward)

        # if roundEnd:#一轮结束
        #     roundCnt = np.array([len(self.agents[i].player.cards) for i in range(3)])
        #     sct=roundCnt[0] - min(roundCnt[1], roundCnt[2])
        #     sct_pre=self.roundBeginCnt[0] - min(self.roundBeginCnt[1], self.roundBeginCnt[2])
        #     sc=sct-sct_pre
        #     for i in range(3):
        #         k = len(self.agents[i].memoryReward) - 1
        #         if i==0:
        #             sc=-sc*2
        #         if k >= 0 and sc != 0:
        #             self.agents[i].memoryReward[k] += sc/50
            # print(self.env.dealer,winAgentId,self.agents[winAgentId].player.id)
            # print(scList)
            # print(self.agents[0].memoryReward)
            # print(self.agents[1].memoryReward)
            # print(self.agents[2].memoryReward)
    def getActionSpaceType(self,nowAgent):
        isDealer = (nowAgent.player.id != self.env.dealer)
        k = self.args.policykind[isDealer]
        return k in actionSpaceType
    def bidFun(self,i,pid, already, actList):  # 叫牌网络
        act_id = random.randint(0, len(actList) - 1)
        if already[0]==0 and already[1]==0 and already[2]==0:
            act_id=1
        return actList[act_id]
    def updateDone(self):
        for i in range(3):
            k = len(self.agents[i].memoryIsDone) - 1
            if k>=0 and self.istrain[self.agents[i].player.id!=self.env.dealer]==1:
                self.agents[i].memoryIsDone[k] = False
    def learnNet(self,playCnt):
        # print(playCnt)
        if self.istrain[0] == 1 :
            if playCnt%self.args.learn_cstep==0:
                up=len(self.agents[0].memoryReward)
                self.agents[0].selflearn_critic(self.agents[0].preup,up,updateNetwork_critic)
            if playCnt%self.args.learn_astep==0:  # 训练地主
                self.agents[0].initLearn()
                self.agents[0].selflearn_actor(updateNetwork_actor)
                self.agents[0].initMenory()

        if self.istrain[1] == 1:
            if playCnt % self.args.learn_cstep == 0:
                for i in range(1, 3):
                    up = len(self.agents[i].memoryReward)
                    self.agents[i].selflearn_critic(self.agents[i].preup, up, updateNetwork_critic)
            if playCnt%self.args.learn_astep==0:  # 训练农民
                for i in range(1, 3):
                    self.agents[i].initLearn()
                    self.agents[i].selflearn_actor(updateNetwork_actor)
                for j in range(1,3):
                    self.agents[j].initMenory()

    def playAGame(self,trainCnt,mini_epoch,trainDataset=True):#玩一局游戏，同时要收集数据
        env=self.env
        env.reset()
        # deck=[12, 16, 17, 26, 3, 43, 36, 42, 30, 24, 29, 44, 22, 4, 31, 33, 49, 5, 1, 50, 19, 27, 46, 32, 40, 20, 45, 14, 39,
        #  7, 54, 10, 21, 15, 13, 8, 48, 52, 47, 38, 37, 34, 35, 2, 25, 23, 28, 51, 18, 41, 53, 11, 9, 6, ]
        deck=None
        if self.args.deckDataNpy_len!=0 and trainDataset:
            self.dataset_i=(self.dataset_i+random.randint(0,10))%self.args.deckDataNpy_len
            deck=self.args.deckDataNpy[self.dataset_i].tolist()
        # if mini_epoch[1]<mini_epoch[0]*0.1:
        #     deck,dealer=mkDeck(cheat1)
        # env.dealCards(deck, dealer)
        env.dealCards(deck, 0)
        roundId = 0
        self.setNet(0)

        maxAgentId = 0#0永远是庄家
        env.historyAct=[]
        self.historyActionList = []
        for i in range(3):
            self.agents[i].initHisGameActFea(self.hisGameActList)
        while(True):  # 开始出牌
            # env.printPlayerHand()
            # print("轮次：", epoch, "  先出牌玩家：", maxPlayerID)

            self.roundBeginCnt=np.array([len(self.agents[i].player.cards) for i in range(3)])
            maxAgent = self.agents[maxAgentId]
            allAct = self.agents[maxAgentId].player.getAllFirstAction(all=self.getActionSpaceType(maxAgent))
            que = deque(maxlen=2)
            maxact = self.playerPolicy(roundId,maxAgent,None,que,allAct)  #在这里面存入buffer
            que.append(maxact)
            # maxact.println(0)
            roundId += 1
            self.historyActionList.append(actTo01code(maxact,env.players[maxact.playerId]))
            pid = maxAgentId
            winPlayer=winAgent=-1
            if len(env.players[maxact.playerId].cards)==0:
                winPlayer = maxact.playerId
                winAgent=pid
            act_i=0
            while(winPlayer==-1):

                pid = (pid + 1) % 3
                maxAgent=self.agents[maxAgentId]
                nowAgent = self.agents[pid]
                allAct = self.agents[pid].player.getAllAction(maxact,all=self.getActionSpaceType(nowAgent))
                act = self.playerPolicy(roundId,self.agents[pid],maxAgent,que,allAct)
                roundId += 1
                self.historyActionList.append(actTo01code(act,env.players[act.playerId]))
                maxAgentId, maxact, winAgent = env.step(maxact, act, maxAgentId, pid)
                act_i += 1
                # act.println(act_i)
                que.append(act)
                roundEnd=(len(que) == 2 and que[0].isPass() and que[-1].isPass())
                if winAgent != -1 or roundEnd:
                    if winAgent != -1:
                        winPlayer = self.agents[winAgent].player.id
                    break
            self.setReward_Sparse(True, env.settleScList(winPlayer),winAgent)
            # print("asdsda")
            if winPlayer!=-1:
                # env.printPlayerHand()
                self.updateDone()
                self.hisGameActList=self.historyActionList
                # print(winPlayer)
                # exit()
                # print(winPlayer,env.settleScList(winPlayer))
                return roundId,winPlayer,env.settleScList(winPlayer)
    def testModel(self,epoch_size):
        tmp=self.istrain.copy()
        self.istrain=[0,0]
        winsum=[0,0]
        playerGradeSum=[0,0]
        for _ in range(epoch_size):
            roundUp, winPlayer, scList = self.playAGame(self._trainCnt, (epoch_size, _),trainDataset=False)  # ！=-1代表到A，本次游戏结束，返回赢得那个人。
            winsum[winPlayer != self.env.dealer] += 1
            playerGradeSum[0] += scList[self.env.dealer]  # z
            playerGradeSum[1] += scList[(self.env.dealer + 1) % 3] + scList[(self.env.dealer + 2) % 3]
        rate = winsum[0] / (winsum[0] + winsum[1])
        avgAdp = (playerGradeSum[0] - playerGradeSum[1]) / (2 * epoch_size)
        self.istrain = tmp
        return rate,avgAdp
    def updatactEpsl(self):
        self.actepsl = max(0.2,self.actepsl-0.01)
        # if self.actepsl==0:
        #     self.initactEpsl()
    def initactEpsl(self):
        self.actepsl = 0.2
    def reachThre(self,adp):
        return self.istrain[0]==1 and  adp>self.args.scthreshold or self.istrain[1]==1 and adp<-self.args.scthreshold
    def train_agent(self,  readName=None):
        if readName!=None and len(readName)>0:
            self.initNet("mod/"+readName+"/")
        else:
            self.initNet()
        print("begin train!!!")
        print(self.policykind,self.istrain)
        epoch_size=self.args.epoch_size
        pltDataMaxLen=10000
        pltDataLen=self.args.pltDataLen
        xList = []
        y1List = []
        y2List = []
        pointList=[]
        self._trainCnt = 0
        self._playCnt = 0
        env=self.env
        self.winPlayerSum=[0,0]#0是0和2赢，1是1和3赢。
        self.playerGradeSum=[0,0]
        start_time = time.time()
        ma5=MACD(10)
        self.initactEpsl()
        self.dataset_i=0
        self.hisGameActList=[]
        while True:
            winsum=[0,0]
            self.isPrint=True
            for _ in range(epoch_size):
                roundUp,winPlayer,scList = self.playAGame(self._trainCnt,(epoch_size,_))#！=-1代表到A，本次游戏结束，返回赢得那个人。
                # print(winPlayer, scList)
                self._playCnt+=1
                self.learnNet(self._playCnt)
                # break
                self.winPlayerSum[winPlayer!=env.dealer]+= 1
                winsum[winPlayer!=env.dealer] += 1
                self.playerGradeSum[0]+=scList[env.dealer]#z
                self.playerGradeSum[1]+=scList[(env.dealer+1)%3]+scList[(env.dealer+2)%3]

            print("Number of games "+str(self._trainCnt)+" : ",winsum,self.playerGradeSum)#谁赢了多少次
            self.updatactEpsl()
            # if abs(winsum[0]-winsum[1])>=epoch_size-2:
            #     self.actepsl=1
            # else:
            #     self.actepsl=0.2
            # self.agents[1].net.printGrad()
            self._trainCnt += 1
            if self._trainCnt%pltDataLen==0:
                self.saveAgentNet()
                print("save net "+self.savePath)
            if self._trainCnt % pltDataLen == 0:
                drawBrokenLine(xList, y1List,None, self.args.picName + "_WP_"+self.saveTime, "epoch",
                               "The proportion of landowners winning in one epoch")
                drawBrokenLine(xList, y2List,pointList, self.args.picName + "_ADP_"+self.saveTime, "epoch",
                               "Addition the score of the landlord from the score of the farmer")
                end_time = time.time()
                print("draw pic " + self.args.picName + "_wp.jpg")
                print("Program execution time: " + str(end_time-start_time) )
                start_time=end_time
            if self._trainCnt % (pltDataMaxLen) == 0:
                xList ,y1List,y2Listm,pointList = [],[],[],[]
                print("reset xlist!!")

            xList.append(self._trainCnt)
            rate=self.winPlayerSum[0] / (self.winPlayerSum[0] + self.winPlayerSum[1])
            avgAdp=(self.playerGradeSum[0] - self.playerGradeSum[1])/(2*epoch_size)
            ma5.add(avgAdp)
            y1List.append(rate)
            y2List.append(avgAdp)
            self.winPlayerSum = [0, 0]
            self.playerGradeSum = [0, 0]
            adp=ma5.getAvg()
            if self._trainCnt%100==0 or (self._trainCnt%10==0 and self.reachThre(adp)):#ma30大于0.6
                wp,adp=self.testModel(1000)
                print("test wp "+str(wp)+", test adp:"+str(adp))
                if self.reachThre(adp):
                    self.initactEpsl()
                    if self.istrain[1]==1:
                        self.args.scthreshold=max(0.4,self.args.scthreshold-0.1)
                    # ma5 =MACD(10)
                    self._playCnt=0
                    pointList.append((self._trainCnt,avgAdp))
                    self.istrain[0],self.istrain[1]=self.istrain[1],self.istrain[0]
                    print("swap train!!",self.istrain,self.policykind)
            # break