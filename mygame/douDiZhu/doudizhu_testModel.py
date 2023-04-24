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
from otherPolicyBaseline import getModel
from doudizhu_codeParameter import INFEA
from doudizhu_ppo_agent import Dppoagent
from doudizhu_cheat import mkDeck, cheat1
from doudizhu_utils import getNowTimePath, drawBrokenLine, MACD
from doudizhu_game import Doudizhu,Action,Player,dfsPrintActList
from doudizhu_encoder import actTo01code, cardsTo01code
from doudizhu_network import AgentNet, AgentCriNet
import pyro
import pyro.distributions as dist



def updateNetwork(agent,worker, sumLoss_critic, sumLoss_actor1, sumLoss_actor2):
    agent.critic_net.zero_grad()
    sumLoss_critic = torch.stack(sumLoss_critic).mean()
    sumLoss_critic.backward()
    torch.nn.utils.clip_grad_norm_(agent.critic_net.parameters(), 3)
    # if worker.epoch_i == 0:
    # agent.critic_net.printGrad()
    agent.opt_cri.step()

    if len(sumLoss_actor1) > 0:
        agent.net.zero_grad()
        loss = torch.stack(sumLoss_actor1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.net.parameters(), 3)
        # if worker.epoch_i == 0:
        #     agent.net.printGrad()
        agent.opt[0].step()

    agent.net.zero_grad()
    loss = torch.stack(sumLoss_actor2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.net.parameters(), 3)
    # if worker.epoch_i==0:
    # agent.net.printGrad()
    agent.opt[1].step()
class DppoWorkers:
    def __init__(self, id, args):
        self.id = id
        self.args = args
        self.cudaID=args.cudaIDList[id]

        self.gama = args.gama
        self.env = Doudizhu()
        self.agents = (Dppoagent(0, args, self), Dppoagent(1, args, self),Dppoagent(2, args, self))
        self.policykind=self.args.policykind
        self.istrain=self.args.istrain
        self.savePath = getNowTimePath()

        self.memory_batch = 50
        MEMORY_CAPACITY = 26
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        OVERLAY_CNT = 1  # 针对此游戏的叠帧操作
        self.OVERLAY_CNT = OVERLAY_CNT
        self.rewardFun=getattr(self, self.args.rewardFun)
        # beginShape = self.resetPic(self.env.reset()).shape
    def initNet(self,path1="",path2=""):#如果path存在,就是读取
        self.actor_net = [AgentNet(INFEA).cuda(self.cudaID) for i in range(3)]#0是地主，1是地主下家，2是地主上家

        self.critic_net_landlord = AgentCriNet(INFEA).cuda(self.cudaID)
        self.critic_net_peasant  = AgentCriNet(INFEA).cuda(self.cudaID)

        if path1=="" or not os.path.exists(path1):
            print("模型不存在1")
        else:
            p0 = path1 + "critic_net_landlord.pt"
            self.critic_net_landlord = torch.load(p0).cuda(self.cudaID)
            p = path1 + "agentNet0.pt"
            if not os.path.exists(p):
                print("模型不存在1")
            self.actor_net[0]= torch.load(p).cuda(self.cudaID)
            print("read success1")
        if path2=="" or not os.path.exists(path2):
            print("模型不存在2")
        else:
            p0 = path2 + "critic_net_peasant.pt"
            self.critic_net_peasant=torch.load(p0).cuda(self.cudaID)
            for i in range(1,3):
                p = path2 + "agentNet" + str(i) + ".pt"
                if not os.path.exists(p):
                    print("模型不存在2")
                self.actor_net[i]= torch.load(p).cuda(self.cudaID)
            print("read success2")
        self.critic_net_landlord.train()
        self.critic_net_peasant.train()
        for i in range(3):
            self.actor_net[i].train()

        self.optimizer_critic_landlord = torch.optim.Adam(self.critic_net_landlord.parameters(),lr=self.args.value_lr)
        self.optimizer_critic_peasant = torch.optim.Adam(self.critic_net_peasant.parameters(), lr=self.args.value_lr)
        self.optimizer=[]
        for i in range(3):
            baseNet1 = list(self.actor_net[i].mlp_base.parameters()) + list(self.actor_net[i].lstm1.parameters())\
                      + list(self.actor_net[i].mlp_act1.parameters())
            baseNet2 = list(self.actor_net[i].mlp_base.parameters()) + list(self.actor_net[i].lstm2.parameters())\
                      + list(self.actor_net[i].mlp_act2.parameters())
            self.optimizer.append([torch.optim.Adam(baseNet1, lr=self.args.policy_lr),torch.optim.Adam(baseNet2, lr=self.args.policy_lr)])

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


    def playerPolicy(self,roundId,scList,nowAgent,maxAgent,preActQue,allActionList:list):  # 第一个人的出牌策略,需要神经网络决策,act是其他玩家出的牌，这里一定是空
        # maxact是0张量代表首先出牌
        nowAgent.initPar(roundId,preActQue)
        # print(roundId)
        isDealer=(nowAgent.player.id!=self.env.dealer)
        k=self.args.policykind[isDealer]
        act=None
        if k in nowAgent.actPolicyFun:  # 选择策略
            # dfsPrintActList(allActionList)
            act: Action = nowAgent.actPolicyFun[k](roundId, self.env, maxAgent, preActQue, allActionList)
            # act.print()
        else:
            print("策略不存在")
            exit()
        return act

    def setReward_Sparse(self,pid,maxAgentId,act,que, scList,winAgentId):#稀疏奖励
        if winAgentId!=-1:
            # self.env.printPlayerHand()
            # print(winPlayer,self.env.dealer)
            for i in range(3):
                nowid=(winAgentId+i)%3
                k=len(self.agents[nowid].memoryReward)-1
                sc = scList[self.agents[nowid].player.id]
                if k>=0 and sc!=0:
                    self.agents[nowid].memoryReward[k]=(math.log2(abs(sc))+1)*(np.sign(sc))/2.3306

            # print(self.env.dealer,winAgentId,self.agents[winAgentId].player.id)
            # print(scList)
            # print(self.agents[0].memoryReward)
            # print(self.agents[1].memoryReward)
            # print(self.agents[2].memoryReward)
    def getActionSpaceType(self,nowAgent):
        isDealer = (nowAgent.player.id != self.env.dealer)
        k = self.args.policykind[isDealer]
        return (k==-3 or k==-2 or k==3 or k==11 or k==-4)
    def bidFun(self,i,pid, already, actList):  # 叫牌网络
        act_id = random.randint(0, len(actList) - 1)
        if already[0]==0 and already[1]==0 and already[2]==0:
            act_id=1
        return actList[act_id]
    def playAGame(self,trainCnt,mini_epoch):#玩一局游戏，同时要收集数据
        env=self.env
        env.reset()
        # deck=[12, 16, 17, 26, 3, 43, 36, 42, 30, 24, 29, 44, 22, 4, 31, 33, 49, 5, 1, 50, 19, 27, 46, 32, 40, 20, 45, 14, 39,
        #  7, 54, 10, 21, 15, 13, 8, 48, 52, 47, 38, 37, 34, 35, 2, 25, 23, 28, 51, 18, 41, 53, 11, 9, 6, ]
        deck=None
        if self.args.deckDataNpy_len!=0:
            self.dataset_i=(self.dataset_i+random.randint(0,10))%self.args.deckDataNpy_len
            deck=self.args.deckDataNpy[self.dataset_i].tolist()
        # if mini_epoch[1]<mini_epoch[0]*0.1:
        #     deck,dealer=mkDeck(cheat1)
        # env.dealCards(deck, dealer)
        env.dealCards(deck, 0)

        scList=env.scList
        # print(scList)
        roundId = 0
        self.setNet(0)

        maxAgentId = 0#0永远是庄家
        for i in range(3):
            self.agents[i].initMenory()
        env.historyAct=[]
        self.historyActionList = []
        while(True):  # 开始出牌
            # env.printAllCards()
            # print("轮次：", epoch, "  先出牌玩家：", maxPlayerID)
            maxAgent = self.agents[maxAgentId]
            allAct = self.agents[maxAgentId].player.getAllFirstAction(all=self.getActionSpaceType(maxAgent))
            que = deque(maxlen=2)
            maxact = self.playerPolicy(roundId,scList,maxAgent,None,que,allAct)  #在这里面存入buffer
            roundId += 1
            self.historyActionList.append(actTo01code(maxact,env.players[maxact.playerId]))
            pid = maxAgentId
            winPlayer = -1
            while(True):

                pid = (pid + 1) % 3
                maxAgent=self.agents[maxAgentId]
                allAct = self.agents[pid].player.getAllAction(maxact,all=self.getActionSpaceType(maxAgent))
                act = self.playerPolicy(roundId,scList,self.agents[pid],maxAgent,que,allAct)
                roundId += 1
                self.historyActionList.append(actTo01code(act,env.players[act.playerId]))
                maxAgentId, maxact, winAgent = env.step(maxact, act, maxAgentId, pid)
                if winAgent != -1:
                    winPlayer = self.agents[winAgent].player.id
                    if env.dealer == winPlayer:
                        scList[(env.dealer + 1) % 3] = -scList[(env.dealer + 1) % 3]
                        scList[(env.dealer + 2) % 3] = -scList[(env.dealer + 2) % 3]
                    else:
                        scList[env.dealer] = -scList[env.dealer]
                self.setReward_Sparse(pid,maxAgentId,act,que, scList,winAgent)
                que.append(act)
                if winAgent != -1 or len(que) == 2 and que[0].isPass() and que[-1].isPass():
                    break
            # reset
            # env.printAllInfo(act)
            if winPlayer!=-1:
                # print(scList)
                # print(winPlayer)
                # exit()
                return roundId,winPlayer,scList
    def updatactEpsl(self):
        self.actepsl = max(0.1,self.actepsl-0.1)
    def initactEpsl(self):
        self.actepsl = 0.1
    def train_agent(self,  readName1,readName2):
        self.initNet("mod/"+readName1+"/","mod/"+readName2+"/")
        print("begin train!!!")
        print(self.policykind,self.istrain)
        epoch_size=self.args.epoch_size
        pltDataMaxLen=10000
        pltDataLen=self.args.pltDataLen
        xList = []
        y1List = []
        y2List = []
        pointList=[]
        self.__trainCnt = 0
        env=self.env
        self.winPlayerSum=[0,0]#0是0和2赢，1是1和3赢。
        self.playerGradeSum=[0,0]
        start_time = time.time()
        ma5=MACD(5)
        self.initactEpsl()
        self.dataset_i=0
        while True:
            winsum=[0,0]
            self.epoch_i=epoch_size
            for _ in range(epoch_size):
                roundUp,winPlayer,scList = self.playAGame(self.__trainCnt,(epoch_size,_))#！=-1代表到A，本次游戏结束，返回赢得那个人。
                self.epoch_i -=1
                # break
                self.winPlayerSum[winPlayer!=env.dealer]+= 1
                winsum[winPlayer!=env.dealer] += 1
                self.playerGradeSum[0]+=scList[env.dealer]#z
                self.playerGradeSum[1]+=scList[(env.dealer+1)%3]+scList[(env.dealer+2)%3]

            print("Number of games "+str(self.__trainCnt)+" : ",winsum,self.playerGradeSum)#谁赢了多少次
            self.updatactEpsl()
            # if abs(winsum[0]-winsum[1])>=epoch_size-2:
            #     self.actepsl=1
            # else:
            #     self.actepsl=0.2
            # self.agents[1].net.printGrad()
            self.__trainCnt += 1
            if self.__trainCnt%pltDataLen==0:
                self.saveAgentNet()
                print("save net "+self.savePath)
            if self.__trainCnt % pltDataLen == 0:
                drawBrokenLine(xList, y1List,None, self.args.picName + "_wp", "epoch",
                               "The proportion of landowners winning in one epoch")
                drawBrokenLine(xList, y2List,pointList, self.args.picName + "_adp", "epoch",
                               "Addition the score of the landlord from the score of the farmer")
                end_time = time.time()
                print("draw pic " + self.args.picName + "_wp.jpg")
                print("Program execution time: " + str(end_time-start_time) )
                start_time=end_time
            if self.__trainCnt % (pltDataMaxLen) == 0:
                xList ,y1List,y2Listm,pointList = [],[],[],[]
                print("reset xlist!!")

            xList.append(self.__trainCnt)
            rate=self.winPlayerSum[0] / (self.winPlayerSum[0] + self.winPlayerSum[1])
            avgAdp=(self.playerGradeSum[0] - self.playerGradeSum[1])/(2*epoch_size)
            ma5.add(avgAdp)
            y1List.append(rate)
            y2List.append(avgAdp)
            self.winPlayerSum = [0, 0]  # 0是0和2赢，1是1和3赢。
            self.playerGradeSum = [0, 0]
def loadnpy(args,path):
    if os.path.exists(path):
        args.deckDataNpy = np.load(path)
        args.deckDataNpy_len=args.deckDataNpy.shape[0]
    else:
        args.deckDataNpy=None
        args.deckDataNpy_len=0
        print("没有数据集文件")
parser = argparse.ArgumentParser()
parser.add_argument('--isInitPar', type=bool, default=False)
parser.add_argument('--cudaIDList', type=list, default=[1])
parser.add_argument('--value_lr', type=float, default=0.0002)
parser.add_argument('--policy_lr', type=float, default=0.0002)
parser.add_argument('--entropy_coef', type=float, default=0)
parser.add_argument('--learn_step', type=int, default=50)
parser.add_argument('--collection_length', type=int, default=1)
parser.add_argument('--dataUseCnt', type=int, default=1)
parser.add_argument('--istrain', type=list, default=[0,0])#1代表训练
parser.add_argument('--policykind', type=list, default=[-4,3])
parser.add_argument('--shape', type=tuple, default=(1,350))#初始时训练那个人
parser.add_argument('--gama', type=float, default=1)
parser.add_argument('--lanta', type=float, default=1)
parser.add_argument('--rewardFun', type=str, default="setReward_Sparse")#setReward_Sparse
parser.add_argument('--picName', type=str, default="ppogae")
parser.add_argument('--epoch_size', type=int, default=99)
parser.add_argument('--pltDataLen', type=int, default=100)
parser.add_argument('--scthreshold', type=float, default=1)
args = parser.parse_args()
loadnpy(args,'')#eval_data.npy
print("cuda:"+str(args.cudaIDList[0]))
worker=DppoWorkers(0,args)
worker.train_agent("","")#2023-04-21-19-47