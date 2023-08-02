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
from doudizhu_codeParameter import INFEA, actionSpaceType
from doudizhu_ppo_agent import Dppoagent
from doudizhu_cheat import mkDeck, cheat1, setHandCards, setAllPlayerHandCards
from doudizhu_utils import getNowTimePath, drawBrokenLine, MACD, getNowTime
from doudizhu_game import Doudizhu,Action,Player,dfsPrintActList
from doudizhu_encoder import actTo01code, cardsTo01code
from doudizhu_network import AgentNet, AgentCriNet
import pyro
import pyro.distributions as dist

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
        self.memory_batch = 50
        MEMORY_CAPACITY = 26
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        OVERLAY_CNT = 1  # 针对此游戏的叠帧操作
        self.OVERLAY_CNT = OVERLAY_CNT
        # beginShape = self.resetPic(self.env.reset()).shape
    def initNet(self,path1="",path2=""):#如果path存在,就是读取
        self.actor_net = [AgentNet(INFEA).cuda(self.cudaID) for i in range(3)]#0是地主，1是地主下家，2是地主上家

        if path1=="" or not os.path.exists(path1):
            print("模型不存在1")
        else:
            p = path1 + "agentNet0.pt"
            if not os.path.exists(p):
                print("模型不存在1")
            self.actor_net[0]= torch.load(p).cuda(self.cudaID)
            print("read success1")
        if path2=="" or not os.path.exists(path2):
            print("模型不存在2")
        else:
            for i in range(1,3):
                p = path2 + "agentNet" + str(i) + ".pt"
                if not os.path.exists(p):
                    print("模型不存在2")
                self.actor_net[i]= torch.load(p).cuda(self.cudaID)
            print("read success2")
        for i in range(3):
            self.actor_net[i].train()

        self.optimizer=[]
        for i in range(3):
            self.optimizer.append([None,None])

        #下面加载baseline的模型
        douzero_models = getModel("douzero", self.cudaID)
        douzeroResnet_models=getModel("douzeroresnet", self.cudaID)
        perfectdou_models = getModel("perfectdou", self.cudaID)
        for i in range(3):
            self.agents[i].douzero_models=douzero_models
            self.agents[i].douzeroResnet_models = douzeroResnet_models
            self.agents[i].perfectdou_models = perfectdou_models


    def setNet(self,pid=0):#把网络给对应智能体
        self.agents[pid].setNet(self.actor_net[0],None)
        self.agents[(pid+1)%3].setNet(self.actor_net[1],None)
        self.agents[(pid+2)%3].setNet(self.actor_net[2],None)

        self.agents[pid].opt = self.optimizer[0]
        self.agents[(pid+1)%3].opt = self.optimizer[1]
        self.agents[(pid+2)%3].opt = self.optimizer[2]

        for i in range(3):
            agent=self.agents[(pid+i)%3]
            agent.setPlayer(self.env.players[(self.env.dealer+i)%3])#0是庄家
            agent.beginCardsFea=cardsTo01code(agent.player.beginCards)

    def saveAgentNet(self):
        path=self.savePath
        if not os.path.exists(path):
            os.mkdir(path)
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
            act: Action = nowAgent.actPolicyFun[k][self.args.istrain[isDealer]](roundId, self.env, maxAgent, preActQue, allActionList)
            # act.print()
        return act

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
            if k>=0:
                self.agents[i].memoryIsDone[k] = False
    def playAGame(self,trainCnt,mini_epoch):#玩一局游戏，同时要收集数据
        env=self.env
        env.reset()
        # deck=[12, 16, 17, 26, 3, 43, 36, 42, 30, 24, 29, 44, 22, 4, 31, 33, 49, 5, 1, 50, 19, 27, 46, 32, 40, 20, 45, 14, 39,
        #  7, 54, 10, 21, 15, 13, 8, 48, 52, 47, 38, 37, 34, 35, 2, 25, 23, 28, 51, 18, 41, 53, 11, 9, 6, ]
        deck=None
        if self.args.deckDataNpy_len!=0:
            self.dataset_i=(self.dataset_i+random.randint(0,10))%self.args.deckDataNpy_len
            deck=self.args.deckDataNpy[self.dataset_i].tolist()
        env.dealCards(deck, 0)
        roundId = 0
        self.setNet(0)

        env.historyAct=[]
        self.historyActionList = []
        for i in range(3):
            self.agents[i].initHisGameActFea(self.hisGameActList)
        maxAgentId = 0
        # maxAgentId = 1
        # for i in range(3):
        #     setAllPlayerHandCards(env.players, [["A", "A"], ["3", "3","6","6","K","K"], ["大王", "4","4"]])
        # env.printPlayerHand()
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
            # print("asdsda")
            # print(env.settleScList(winPlayer))
            if winPlayer!=-1:
                # env.printPlayerHand()
                self.updateDone()
                self.hisGameActList=self.historyActionList
                # print(winPlayer)
                # exit()
                return roundId,winPlayer,env.settleScList(winPlayer)
    def updatactEpsl(self):
        self.actepsl = max(0.3,self.actepsl-0.2)
    def initactEpsl(self):
        self.actepsl = 0.3
    def train_agent(self, readName1,readName2):
        self.initNet("mod/" + readName1 + "/", "mod/" + readName2 + "/")
        print("begin train!!!")
        print(self.policykind, self.istrain)
        epoch_size=self.args.epoch_size
        pltDataMaxLen=10000
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
        ma5=MACD(5)
        self.initactEpsl()
        self.dataset_i=0
        self.hisGameActList=[]
        while True:
            winsum=[0,0]
            self.isPrint=True
            for _ in range(epoch_size):
                roundUp,winPlayer,scList = self.playAGame(self._trainCnt,(epoch_size,_))#！=-1代表到A，本次游戏结束，返回赢得那个人。
                self._playCnt+=1
                # break
                self.winPlayerSum[winPlayer!=env.dealer]+= 1
                winsum[winPlayer!=env.dealer] += 1
                self.playerGradeSum[0]+=scList[env.dealer]#z
                self.playerGradeSum[1]+=scList[(env.dealer+1)%3]+scList[(env.dealer+2)%3]
            adp=(self.playerGradeSum[0]-self.playerGradeSum[1])/(epoch_size*2)
            print("Number of games "+str(self._trainCnt)+" : ",winsum,self.playerGradeSum,"adp:"+str(adp))#谁赢了多少次
            self.updatactEpsl()
            self.winPlayerSum = [0, 0]  # 0是0和2赢，1是1和3赢。
            self.playerGradeSum = [0, 0]
            break

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
parser.add_argument('--cudaIDList', type=list, default=[0])
parser.add_argument('--value_lr', type=float, default=0.0002)
parser.add_argument('--policy_lr', type=float, default=0.0002)
parser.add_argument('--entropy_coef', type=float, default=0)
parser.add_argument('--learn_step', type=int, default=50)
parser.add_argument('--collection_length', type=int, default=1)
parser.add_argument('--dataUseCnt', type=int, default=1)
parser.add_argument('--istrain', type=list, default=[0,0])#1代表训练
parser.add_argument('--policykind', type=list, default=[-2,1])
parser.add_argument('--shape', type=tuple, default=(1,350))#初始时训练那个人
parser.add_argument('--gama', type=float, default=1)
parser.add_argument('--lanta', type=float, default=1)
parser.add_argument('--picName', type=str, default="ppogae")
parser.add_argument('--epoch_size', type=int, default=1000)
parser.add_argument('--seeType', type=int, default=1)#1是不能看见
parser.add_argument('--partialRewards', type=float, default=0)
args = parser.parse_args()
loadnpy(args,'')#eval_data.npy
print("cuda:"+str(args.cudaIDList[0]))
worker=DppoWorkers(0,args)
worker.train_agent("2023-06-17-13-33","2023-06-17-13-33")#2023-04-21-19-47
