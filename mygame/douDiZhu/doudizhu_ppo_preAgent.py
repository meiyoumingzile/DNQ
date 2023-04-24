import math
import random
from functools import cmp_to_key

import numpy as np
import torch

import doudizhu_game
from doudizhu_game import Doudizhu,Action,Player,dfsPrintActList
from doudizhu_encoder import handTo01code, getAllActionFea, getBaseFea, actTo01code, actZeros, getBaseFea_withprob, \
    getOtherProbFea
import pyro
import pyro.distributions as dist



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
        self.beginCardsFea=None
        self.CLIP_EPSL = 0.2
        self.actPolicyFun={-1:self.chooseActFromNetMax,0:self.chooseActFromNet,1:self.chooseActFromNet,2:self.chooseAct_random2,3:self.chooseAct_random3}
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

    def chooseAct_random2(self,roundId,env,maxPlayerId,maxact,allActionList:list):  # 无脑出个数最多的
        n = len(allActionList)
        allActionList.sort(key=cmp_to_key(doudizhu_game._sortAction_cmp1))
        newList = []
        prelen = -1
        for a in allActionList:
            if a.len != prelen:
                newList.append(a)
                prelen = a.len
        id = random.randint(0, len(newList) - 1)
        env.players[self.id].useAction(newList[id], doudizhu_game.baselinePolicy_INFfun)
        return newList[id]
    def chooseAct_random3(self,roundId,env,maxPlayerId,maxact,allActionList:list):#随机出牌
        id = random.randint(0, len(allActionList) - 1)
        env.players[self.id].useAction(allActionList[id], doudizhu_game.baselinePolicy_INFfun)
        return allActionList[id]

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
    def chooseActFromNet(self,roundId,env,maxPlayerId,maxact,allActionList:list):#ansUp代表较大的动作，ansDown代表较小的动作
        baseFea_see = getBaseFea(env, self.id, maxact,kind=2).cuda(self.worker.cudaID)  #可见信息
        allAct_kind = env.classActKind(allActionList)

        # print(self.worker.cudaID,next(self.net.parameters()).device)
        hisActFea=self.hisActTensor
        predictFea=self.net.forward_pre(baseFea_see, hisActFea).squeeze(dim=0)
        predictFea1,predictFea2=getOtherProbFea(self.env,self.id,self.beginCardsFea,predictFea)
        baseFea=getBaseFea_withprob(self.env, self.id, self.maxact,predictFea1,predictFea2).cuda(self.worker.cudaID)  # self.id
        #。。。

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
        self.env.players[self.id].useAction(act, self.updateINF)
        return act
    def chooseActFromNetMax(self,roundId,env,maxPlayerId,maxact,allActionList:list):
        baseFea = getBaseFea(env, self.id, maxact,kind=1).cuda(self.worker.cudaID)  # self.id
        allAct_kind = env.classActKind(allActionList)

        # print(self.worker.cudaID,next(self.net.parameters()).device)
        hisActFea = self.hisActTensor
        probList1 = self.net.forward_fp(baseFea, hisActFea).squeeze(dim=0)  # base的特征向量
        for i in range(len(allAct_kind)):
            if len(allAct_kind[i]) == 0:
                probList1[i] = -math.inf
        probList1 = torch.softmax(probList1, dim=0)
        actid_1 = torch.argmax(probList1).item()

        actid_1_prob = probList1[actid_1].item()

        if len(allAct_kind[actid_1]) <= 1:
            actid_2 = 0
            actid_2_prob = 1
        else:
            allActFea = getAllActionFea(allAct_kind[actid_1]).cuda(self.worker.cudaID)
            probList2 = self.net.forward_act(baseFea, hisActFea, allActFea).squeeze(dim=1).cpu().detach()
            actid_2 = torch.argmax(probList2).item()
            actid_2_prob = probList2[actid_2].item()
        act = allAct_kind[actid_1][actid_2]
        self.pushMemory(roundId, allActionList, baseFea, hisActFea, 0, (actid_1, actid_1_prob), (actid_2, actid_2_prob))
        self.env.players[self.id].useAction(act, self.updateINF)
        return act

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
    def selflearn(self,roundUp,worker,updateNetwork):#updateNetwork为更新函数
        up = len(self.memoryBaseFea)
        for _ in range(self.args.dataUseCnt):  # 训练critic
            sumLoss_critic = []
            sumLoss_actor1 = []
            sumLoss_actor2 = []
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
            updateNetwork(self,sumLoss_critic,sumLoss_actor1,sumLoss_actor2)