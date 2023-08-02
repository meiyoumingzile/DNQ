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
from doudizhu_network import AgentNet, AgentCriNet, shuffleNet
LR=0.0001
class deckGenerator:
    def __init__(self, id, args):
        self.id = id
        self.cudaID=args.cudaIDList[0]
        # print(self.cudaID)
        self.args=args
        self.reset()
        self.net=shuffleNet().cuda(self.cudaID)
        self.opt_act = torch.optim.Adam(self.net.mlp_actor.parameters(), lr=LR)
        self.opt_val = torch.optim.Adam(self.net.mlp_critic.parameters(), lr=LR)
        self.gama ,self.lanta=1,1
        self.initMemory()
        sz=100
        self.CLIP_EPSL=0.2
        self.gamaPow = np.zeros((sz))
        self.galanPow = np.zeros((sz))#gama*lanta
        self.gamaPow[0]=self.galanPow[0]=1
        for i in range(1,sz):
            self.gamaPow[i]=self.gamaPow[i-1]*self.gama
            self.galanPow[i]=self.galanPow[i-1]*self.gama*self.lanta
    def reset(self):
        self.deck = torch.Tensor([[i for i in range(1,55)]])
    def pushMemory(self,baseFea,reward,act,done):#allAct内部是多个list
        self.memoryDeck.append(baseFea.cpu())
        self.memoryReward.append(reward)
        self.memoryAct.append(act)
        self.memoryIsDone.append(done)
    def initMemory(self):
        self.memoryDeck=[]
        self.memoryReward=[]
        self.memoryAct=[]
        self.memoryIsDone=[]
    def mkDeck(self):
        self.reset()
        for i in range(doudizhu_game.CARDS_CNT-1, 0, -1):
            # rnd = random.randint(0, i)  # 每次随机出0-i-1之间的下标
            baseFea = torch.cat((self.deck.clone(),torch.Tensor([[i]])),dim=1)
            x=self.net.forward_act(baseFea.cuda(self.cudaID))
            # print(x)
            # rnd=torch.argmax(x[:,0:i+1]).item()
            actid, x=self.epslBoltzmann(x)
            rnd=torch.multinomial(x, num_samples=1, replacement=True)[0].item()
            prob=x[0][rnd].item()
            # print(i,rnd)
            tmp=self.deck[0][i].clone()
            self.deck[0][i]= self.deck[0][rnd]
            self.deck[0][rnd]=tmp
            self.pushMemory(baseFea, 0, (rnd, prob), i>1)

        return self.deck.squeeze(dim=0).int().tolist()
    def setReward(self,reward):#根据adp设置奖励
        k=len(self.memoryReward)-1
        if k>-1:
            self.memoryReward[k]=reward
    def  epslBoltzmann(self, probList, epsl=0.05):  # 修正玻尔兹曼,这里epsl是随机选取的概率
        cnt = probList.shape[0]
        # if cnt<1:
        #     print(probList,mask,probList.shape)
        p = epsl / cnt
        ans = probList * (1 - epsl) + p
        actid = torch.multinomial(ans, num_samples=1, replacement=True)[0].item()
        return actid, ans
    def initLearn(self):#
        up=len(self.memoryDeck)
        memory_i=0
        # print(self.memoryIsDone,self.memoryReward)
        self.advantages = torch.zeros((up, 1)).cuda(self.cudaID)
        while memory_i<up:
            statei = self.memoryDeck[memory_i].cuda(self.cudaID)
            self.detlaList=[]
            prev = self.net.forward_val(statei)
            i=memory_i+1

            while(self.memoryIsDone[i]):#calc detla
                statei = self.memoryDeck[i].cuda(self.cudaID)
                reward = self.memoryReward[i-1]
                v = self.net.forward_val(statei)
                detla=(reward + v * self.gamaPow[1] - prev).view(-1).detach()
                self.detlaList.append(detla.clone())
                prev = v
                i+=1
            reward = self.memoryReward[i]
            detla = (reward - prev).view(-1).detach()
            self.detlaList.append(detla.clone())

            adv0 = 0
            n=i-memory_i
            # print(len(self.detlaList),)
            for i in range(n):
                adv0 += self.galanPow[i] * self.detlaList[i]
            for i in range(n):
                self.advantages[i+memory_i]=adv0.clone()
                adv0 -= self.detlaList[i]
                adv0 /= self.galanPow[1]
            memory_i+=n+1
            statei.cpu()
            # print(memory_i, n, up)
        mean = self.advantages.mean()
        std = self.advantages.std()
        self.advantages = (self.advantages - mean) / (std)
        # print(self.advantages )
    def learn(self):
        up = len(self.memoryDeck)
        if up==0:
            return
        self.initLearn()
        sumLoss_critic = []
        for i in range(0,up):
            statei = self.memoryDeck[i].cuda(self.cudaID)
            reward = self.memoryReward[i]
            done = self.memoryIsDone[i]
            v = self.net.forward_val(statei)

            if done:
                next_statei = self.memoryDeck[i + 1].cuda(self.cudaID)
                next_v = self.net.forward_val(next_statei)
                next_statei.cpu()
                tderro = reward + self.gamaPow[1] * next_v - v  # 因为自动作没有奖励，所以Q(s,a)都是Gt
            else:
                tderro = reward - v
            critic_loss = tderro.pow(2).mean()
            statei.cpu()
            # enemyv=self.enemyCriNet.forward(statei,hisGameActi, hisActi).detach()
            # critic_loss2=-(v-enemyv).pow(3).mean()

            sumLoss_critic.append(critic_loss)
        if len(sumLoss_critic) != 0:
            self.net.zero_grad()
            sumLoss_critic = torch.stack(sumLoss_critic).mean()
            # print(sumLoss_critic.device)
            sumLoss_critic.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.critic_net.parameters(), 3)
            # if worker.epoch_i == 0:
            # agent.critic_net.printGrad()
            self.opt_val.step()

        sumLoss_actor2 = []
        for i in range(up):
            statei = self.memoryDeck[i].cuda(self.cudaID)
            actPolicy2 = self.memoryAct[i]  # actid_2,actid_2_prob
            advantage = self.advantages[i].clone().detach()
            probList2 = self.net.forward_act(statei).squeeze(dim=0)
            actid, probList2 = self.epslBoltzmann(probList2)
            act = torch.Tensor([actPolicy2[0]]).cuda(self.cudaID).long()
            # print(probList2.shape,act)
            nowRate = probList2.gather(0, act)
            # print(probList2,act,nowRate)
            preRate = torch.Tensor([actPolicy2[1]]).cuda(self.cudaID)  # 旧策略
            # print(nowRate.shape,preRate.shape)
            ratio = torch.exp(torch.log(nowRate) - torch.log(preRate))  # 计算p1/p2防止精度问题
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.CLIP_EPSL, 1 + self.CLIP_EPSL) * advantage
            actor_loss = (-torch.max(torch.min(surr1, surr2),
                                     (1 - self.CLIP_EPSL) * advantage) - self.args.entropy_coef * torch.log(nowRate)).mean()
            sumLoss_actor2.append(actor_loss)
            statei.cpu()
            # print(loss_actor,actor_loss)
            # actor_loss.backward()
            # self.optimizer[2].step()
        if len(sumLoss_actor2) != 0:
            # print("trainNet step " + str(worker._playCnt), len(sumLoss_actor2))
            self.net.zero_grad()
            loss = torch.stack(sumLoss_actor2).mean()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            # if worker._trainCnt%100==0 and worker.isPrint:#定期打印梯度
            #     worker.isPrint=False
            #     agent.net.printGrad()
            self.opt_act.step()
        self.initMemory()
# net=deckGenerator(0)
# import time
#
# start_time = time.time()
# for i in range(100):
#
#     net.reset()
#     net.mkDeck()
# # print(net.mkDeck())
# end_time = time.time()
# elapsed_time = end_time - start_time
#
# print(f"程序运行时间为 {elapsed_time:.2f} 秒")