import math
import random
import time
import platform
# import pynvml
import torchvision
import cv2
import gym
import os
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.nn.functional as F#激活函数
import argparse

from tractor_game import env
from tractor_network import AgentNet

BATCH_SZ=50
EPOCHS_CNT=1000000
MEMORY_CAPACITY=200
CLIP_EPSL=0.1
OVERLAY_CNT=1#针对此游戏的叠帧操作
ENVMAX=[1,4]
γ=0.97#reward_decay
α=0.00001#学习率learning_rate,也叫lr
print(platform.system())
parser = argparse.ArgumentParser()
parser.add_argument('--cudaID', type=int, default=0)

args = parser.parse_args()
print("显卡" + str(args.cudaID))
def initInfo(s,f="info.txt"):
    if os.path.exists(f):
        os.remove(f)
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")
def wInfo(s,f="info.txt"):
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")

class Agent(object):#智能体，
    saveMask=[]
    def __init__(self,beginShape):
        self.critic_net = AgentNet().to(args.cudaID)
        self.critic_net.train()

        self.beginShape=beginShape
        self.optimizer=torch.optim.Adam(self.actor_net.parameters(), lr=α)
        self.episodeLi=[]
    def nextState(self,stList,state):
        for i in range(OVERLAY_CNT,0,-1):
            stList[i] = stList[i-1]
        stList[0] = state
        return stList
    def mask_action(self,x):
        infcnt = 0
        for i in range(len(env.fp)):
            act = env.fp[i]
            pos = (env.head.pos[0] + act[0], env.head.pos[1] + act[1])
            if env.board[pos[0]][pos[1]] > 0:
                x[0][i] = -math.inf
                infcnt += 1
        if infcnt == len(env.fp):
            for i in range(len(env.fp)):
                x[0][i] = 1
        return F.softmax(x, dim=1)

    def choose_action(self, x):  # 按照当前状态，选取概率最大的动作
        x = torch.FloatTensor(x).unsqueeze(0).cuda()  # 加一个维度
        x = self.actor_net.forward(x, 0)
        prob = self.mask_action(x).cpu()
        # print(x,prob[0].detach().numpy())
        act = np.random.choice(a=ENVMAX[1], p=prob[0].detach().numpy())
        return act, prob[0],

    def pushRemember(self,state,act,reward,probList,ister):#原本是DQN的方法的结构，但这里用它当作缓存数据
        t=self.memory_i%MEMORY_CAPACITY

        self.memoryQue[0][t] = state[1:OVERLAY_CNT+1]
        self.memoryQue[1][t] = state[0:OVERLAY_CNT]
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memoryQue[4][t] = np.array(probList.detach().numpy())
        self.memoryQue[5][t] = np.array(ister)
        self.memory_i=(self.memory_i+1)
    def learn(self):#学习,训练ac算法2个网络
        #          得TD_error = r + γV(s’)-V(s)
        #           顺便用TD_error的均方误差训练Q_Network
        #       ④ TD_error反馈给Actor，Policy
        # Gradient公式
        # 训练Actor
        # t=(self.memory_i-1+MEMORY_CAPACITY)%MEMORY_CAPACITY
        t = np.random.choice(MEMORY_CAPACITY, BATCH_SZ)
        t[0]=(self.memory_i-1)%MEMORY_CAPACITY
        state = torch.FloatTensor(self.memoryQue[0][t]).unsqueeze(dim=0)  # state
        next_state = torch.FloatTensor(self.memoryQue[1][t]).unsqueeze(dim=0)
        act = torch.LongTensor(self.memoryQue[2][t].astype(int)).cuda()
        reward = torch.FloatTensor(self.memoryQue[3][t]).cuda()
        probList = torch.FloatTensor(self.memoryQue[4][t]).cuda()
        ister = torch.FloatTensor(self.memoryQue[5][t]).cuda()
        state = state.squeeze(dim=0)
        next_state = next_state.squeeze(dim=0)#去掉一层

        v = (self.critic_net.forward(state,1), self.critic_net.forward(next_state,1))
        cri_loss = F.mse_loss(reward + γ * v[1], v[0])  # 用tderror的方差来算

        delta = reward + γ * v[1]-v[0] #tderro也是优势函数,根据具体问题可以在is_ter==True时阻断
        # delta = delta.cpu().detach().numpy()
        # advantage_list = []
        # advantage = 0.0
        # for delta_t in delta[::-1]:
        #     advantage = γ * advantage + delta_t
        #     advantage_list.append(advantage)
        # advantage_list.reverse()
        # # print(delta.shape,advantage_list.shape)
        # advantage = torch.Tensor(delta.detach()).cuda()
        advantage = delta

        # print(advantage.shape)
        nowRate = self.actor_net.forward(state,0)
        for i in range(len(probList)):
            for j in range(len(probList[i])):
                if probList[i][j]==0:
                    nowRate[i][j]=-math.inf
        nowRate=F.softmax(nowRate, dim=1)
        # print(nowRate, probList)
        # print(nowRate.shape)
        nowRate = nowRate.gather(1, act)
        probList= probList.gather(1, act)
        # print(nowRate,probList)
        # print(probList.shape)
        ratio = torch.exp(torch.log(nowRate) - torch.log(probList))#计算p1/p2防止精度问题
        # print(ratio)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - CLIP_EPSL, 1 + CLIP_EPSL) * advantage#clamp(a,l,r)代表把x限定在[l,r]之间
        actor_loss=-torch.min(surr1, surr2).mean()
        # self.optimizer.zero_grad()#0代表演员，1代表评论家
        # actor_loss.backward()
        # self.optimizer.step()
        cri_loss=cri_loss+actor_loss
        self.optimizer.zero_grad()  # 0代表演员，1代表评论家
        cri_loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.actor_net, PATH+"mod/actor_net.pt")  #
        torch.save(self.critic_net, PATH+"mod/critic_net.pt")  #
    def read(self):
        if os.path.exists(PATH+"mod/actor_net.pt") and os.path.exists(PATH+"mod/critic_net.pt") :
            self.actor_net=torch.load(PATH+"mod/actor_net.pt")  #
            self.critic_net = torch.load(PATH+"mod/critic_net.pt")  #
            self.optimizer=torch.optim.Adam(self.actor_net.parameters(), lr=α)
rewardInd={0:-0.05,10:0.5,50:0.6,100:0.7,200:0.8,300:0.9}
preDis=100000
def setReward(reward,is_terminal):#设置奖励函数函数
    global preDis
    if is_terminal :
        if env.emptyCnt>0:
            return -33
        return 33
    if reward==0:
        dis = env.getDisFromFood()
        ans=0
        if dis>preDis:
            ans=-1
        else:
            ans = 1
        preDis=dis
        return ans
    return 33

def train(beginShape, cartPole_util=None):
    agent=Agent(beginShape)
    # agent.read()
    sumreward = 0
    for episode_i in range(EPOCHS_CNT):#循环若干次
        is_terminal=False
        stList = agent.initState(env)
        frame_i=0
        while (not is_terminal):
            # env.render()
            action,prob = agent.choose_action(stList[0:OVERLAY_CNT])
            # print(action)
            next_state, reward, is_terminal = env.step(action)
            stList = agent.nextState(stList, resetPic(next_state))  # 用stList储存前几步
            sumreward += reward
            frame_i+=1
            reward = setReward(reward,is_terminal)
            agent.pushRemember(stList,action,reward,prob,1-is_terminal)
            if agent.memory_i%5==0 and agent.memory_i>MEMORY_CAPACITY:
                agent.learn()
        # if episode_i%10==0:
        #     # checkWD()
        if episode_i>0 and episode_i%100==0 :
            s=str(episode_i) + "reward:" + str(sumreward / 100)
            wInfo(s)
            sumreward = 0
            if episode_i%2000==0:
                wInfo("episode_i:"+str(episode_i))
                agent.save()
    return agent
initInfo("begin!!")

# agent=train(shape)
# agent.save()