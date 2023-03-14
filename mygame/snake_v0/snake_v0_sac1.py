import math
import random
import sys
import time
import platform
# import pynvml
import torchvision
# import cv2
# import gym
import os
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.nn.functional as F#激活函数
from snake_v0 import CC
import argparse
# import psutil
import os
from matplotlib import pyplot as plt


BATCH_SZ=256
EPOCHS_CNT=1000000
MEMORY_CAPACITY=4000
OVERLAY_CNT=1#针对此游戏的叠帧个数
env=CC()#吃豆人#MsPacman-ram-v0
ENVMAX=[1,4]

PATH=""
# PATH="/home/hz/Z_BING333/RL/DNQ/mygame/snake_v0/"
# PATH="/home/hw1/syb/rl/DNQ/mygame/snake_v0/"
# pynvml.nvmlInit()

parser = argparse.ArgumentParser()
parser.add_argument('--cudaID', type=int, default=-1)
class MyPar():#充当静态类
    log_std_range=[-20,2]
    gamma = 0.99  # reward_decay
    lr = 0.001  # 学习率learning_rate,也叫lr
    halpha=0.4
    tau = 0.01
    inf =1e-9
    CLIP_EPSL = 0.1
    state_dim=490
    cudaID = 3
    fileName=sys.argv[0].split("/")[-1][:-3]
    device=torch.device("cuda0" if torch.cuda.is_available() else "cpu",cudaID)
par=MyPar()

if platform.system() == "Linux":
    # if psutil.cpu_count()==128:
    #     PATH = "/work/home/ac39rl9wn5/project/DNQ/mygame/snake_v0/"
    #     par.device=torch.device("cpu")
    # if torch.cuda.is_available():
    #     par.device = torch.device('cuda', 0)
    args = parser.parse_args()
    if args.cudaID<0:
        args.cudaID=par.cudaID
    # print("显卡"+str(args.cudaID))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cudaID)#str(args.cudaID)
if platform.system() =="Windows":
    BATCH_SZ = 1
    MEMORY_CAPACITY =10
    PATH = "D:\code\python\ReinforcementLearning1\DNQ\mygame\snake_v0/"
def initInfo(s,f=PATH+"info.txt"):
    if os.path.exists(f):
        os.remove(f)
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")
def wInfo(s,f=PATH+"info.txt"):
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")

def resetPic(pic):
    # pic=np.array(pic).flatten()
    sz=env.SZ+2
    pic=np.zeros(((sz*sz)))
    headPos=env.head.pos
    k=0
    for i in range(0,sz):
        for j in range(0, sz):
            pic[k] = (env.board[i][j] == 0 or env.board[i][j] == -1)
            k += 1

    a=np.array([abs(headPos[0]-env.foodList[0][0]),abs(headPos[1]-env.foodList[0][1]),abs(headPos[0]-env.foodList[1][0]),abs(headPos[1]-env.foodList[1][1]),
                env.nowstepCnt,env.stepCntUp-env.nowstepCnt
                ])
    pic=np.concatenate((pic, a), axis=0)
    return pic
def resetPic1(pic):
    # pic=np.array(pic).flatten()
    pic=np.zeros((24))
    headPos=env.head.pos
    k=0
    for i in range(-2,3):
        for j in range(-2, 3):
            x = headPos[0] + i
            y = headPos[1] + j
            if (i!=0 or j!=0) and x>0 and y>0 and x<=env.SZ+1 and y<=env.SZ+1:
                pic[k]=(env.board[x][y]==0 or env.board[x][y]==-1)
                k+=1

    a=np.array([env.head.pos[0],env.head.pos[1],env.foodList[0][0],env.foodList[0][1],env.foodList[1][0],env.foodList[1][1],
                env.stepCnt
                ])
    pic=np.concatenate((pic, a), axis=0)
    return np.expand_dims(pic,axis=0)
class ReplayBuffer:
    def __init__(self, beginShape):
        self.memory_i = 0  # counter used for experience replay buffer//初始化记忆池
        self.rangeList=torch.IntTensor([i for i in range(MEMORY_CAPACITY)])
        # print(self.rangeList)
        self.memoryQue = [np.zeros((MEMORY_CAPACITY, OVERLAY_CNT) + beginShape),
                          np.zeros((MEMORY_CAPACITY, OVERLAY_CNT) + beginShape),
                          np.zeros((MEMORY_CAPACITY, 1)),
                          np.zeros((MEMORY_CAPACITY, 1)),
                          np.zeros((MEMORY_CAPACITY, ENVMAX[1])),
                          np.zeros((MEMORY_CAPACITY, ENVMAX[1])),
                          np.zeros((MEMORY_CAPACITY, 1))]

    def pushRemember(self, state, act, reward, probList,nextpList, ister):  # 原本是DQN的方法的结构，但这里用它当作缓存数据
        t = self.memory_i % MEMORY_CAPACITY
        self.memoryQue[0][t] = state[1:OVERLAY_CNT + 1]
        self.memoryQue[1][t] = state[0:OVERLAY_CNT]
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memoryQue[4][t] = np.array(probList.detach().numpy())
        self.memoryQue[5][t] = np.array(nextpList.detach().numpy())
        self.memoryQue[6][t] = np.array(ister)
        self.memory_i = (self.memory_i + 1)

    def __len__(self):
        return len(self.memoryQue)
beginShape = resetPic(env.reset()).shape# 重新开始游戏,(210, 160, 3)
buffer=ReplayBuffer(beginShape)
class ANetwork(nn.Module):#演员网络,actor
    def __init__(self,state_dim, action_dim):
        super().__init__()
        hdim = 128
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2, groups=1),
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=1, stride=1, padding=0, groups=1),
        #     nn.ReLU(),
        #     nn.Conv2d(),
        # )
        self.action_dim=action_dim
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=hdim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hdim, out_features=1, bias=True)
        )

    def forward(self,x):#重写前向传播算法，x是张量
        x=x.to(par.device)
        # print(x.shape)
        x = self.fc1(x)
        x=torch.abs(F.tanh(x))
        return x

class QNetwork(nn.Module):#相当于评论家网络,估计Q
    def __init__(self,state_dim, action_dim):
        super().__init__()
        hdim=128
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_dim+1, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=hdim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hdim, out_features=1, bias=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=state_dim+1, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=hdim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hdim, out_features=1, bias=True)
        )
        self.apply(self.initw)
    def initw(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    def forward(self,x,action):#
        x = torch.cat([x, action], 1)
        x=x.to(par.device)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1,x2

class Agent(object):#智能体，蒙特卡洛策略梯度算法
    saveMask=[]
    def __init__(self,beginShape):
        self.actor_net = ANetwork(par.state_dim, ENVMAX[1]).to(par.device)  # 初始化两个网络，target and training
        self.critic_net, self.critic_net_target = QNetwork(par.state_dim, ENVMAX[1]).to(par.device), QNetwork(par.state_dim,
                                                                                                  ENVMAX[1]).to(par.device)
        self.rangeList=np.array([i for i in range(BATCH_SZ)])
        # print(self.rangeList)
        self.net = (self.actor_net, self.critic_net)
        self.actor_net.train()
        self.critic_net.train()
        self.beginShape = beginShape
        self.learn_step_i = 0
        self.optimizer = (torch.optim.Adam(self.actor_net.parameters(), lr=par.lr),
                          torch.optim.Adam(self.critic_net.parameters(), lr=par.lr))

        # 动态
        self.target_entropy = -torch.prod(torch.Tensor([ENVMAX[1]]).to(par.device)).item()
        print('entropy：', self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=par.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=par.lr)  # 利用优化器动态调整log_alpha
        self.halpha = par.halpha


    def initState(self,env):#
        state = resetPic(env.reset())  # 重新开始游戏,(210, 160)
        stList=np.zeros((OVERLAY_CNT+1,)+state.shape)
        # li[OVERLAY_CNT]=state
        for i in range(0,OVERLAY_CNT):
            action = np.random.randint(0,ENVMAX[1])
            state, reward, is_terminal = env.step(action)
            stList[i] = resetPic(state)
        return stList
    def nextState(self,stList,state):
        for i in range(OVERLAY_CNT,0,-1):
            stList[i] = stList[i-1]
        stList[0] = state
        return stList
    def soft_update(self,net, source):
        for target_param, param in zip(net.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - par.tau) + param.data * par.tau)

    def hard_update(self,net, source):
        for target_param, param in zip(net.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    def saveNowMask(self,x):
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
        x = torch.FloatTensor(x).to(par.device)  # 加一个维度
        x = x.view(x.size(0), -1)
        x = self.actor_net.forward(x)
        # print(x)
        # prob = self.saveNowMask(x).cpu()
        # print(prob)
        # print(x,prob[0].detach().numpy())
        act = np.random.choice(a=ENVMAX[1], p=prob[0].detach().numpy())
        return act, prob[0],
    def choose_actTensor(self, x,maskList):  # 按照当前状态，选取概率最大的动作
        prob = self.actor_net.forward(x)  # 采样一次
        for i in range(len(maskList)):
            for j in range(len(maskList[i])):
                if maskList[i][j]==0:
                    prob[i][j]=-math.inf
        prob = F.softmax(prob, dim=1)

        # prob=torch.max(prob,par.inf)
        return prob, torch.log(prob+par.inf)
    def learn(self):#学习,训练ac算法2个网络
        # t = self.rangeList
        t=np.random.choice(MEMORY_CAPACITY, BATCH_SZ)
        # print(t.shape)
        state = torch.FloatTensor(buffer.memoryQue[0][t]).squeeze(dim=1).to(par.device)  # state
        next_state = torch.FloatTensor(buffer.memoryQue[1][t]).squeeze(dim=1).to(par.device)
        act = torch.LongTensor(buffer.memoryQue[2][t].astype(int)).to(par.device)
        reward = torch.FloatTensor(buffer.memoryQue[3][t]).to(par.device)
        pMask = torch.FloatTensor(buffer.memoryQue[4][t]).to(par.device)
        nextpMask = torch.FloatTensor(buffer.memoryQue[5][t]).to(par.device)
        # ister = torch.FloatTensor(buffer.memoryQue[6][t]).to(par.device)
        # print(next_state.shape,act.shape)

        with torch.no_grad():
            next_prob, next_logprob = self.choose_actTensor(next_state,nextpMask)  # 输入batch_size*shape
            # print(next_prob,next_logprob)
            # print(next_state.shape,next_act.shape)
            # x = torch.cat([next_state.to(par.device), next_act.to(par.device)], 1)
            # print(x.shape)
            next_qx1, next_qx2 = self.critic_net_target.forward(next_state)

            min_nextq = next_prob * (
                        torch.min(next_qx1, next_qx2) - self.halpha * next_logprob)  # Q(next_state,next_act)-halpha*logp
            min_nextq = min_nextq.sum(dim=1).unsqueeze(1)
            # print(reward.shape, min_nextq.shape)
            q_value = reward + par.gamma * min_nextq  # 由贝尔曼期望方程：Q(s,a)=r+gamma*Q(s(t+1)，a(t+1))
            # print(act.shape, next_prob.shape, next_act.shape)
        qx1, qx2 = self.critic_net(state)
        # print(qx1,act)
        qx1 = qx1.gather(1, act)
        qx2 = qx2.gather(1, act)
        # print(qx1)
        q1_loss = F.mse_loss(qx1, q_value)  #
        q2_loss = F.mse_loss(qx2, q_value)  #
        q_loss = q1_loss + q2_loss
        self.optimizer[1].zero_grad()
        q_loss.backward()
        self.optimizer[1].step()

        prob, logprob = self.choose_actTensor(state,pMask)  #
        # print(pMask)
        qx1, qx2 = self.critic_net(state)
        minqx = torch.min(qx1, qx2)
        actor_loss = (self.halpha * logprob - minqx) * prob  # 根据KL散度化简得来
        # print(self.halpha , logprob , minqx)
        actor_loss = actor_loss.sum(dim=1).mean()
        # print(actor_loss)
        self.optimizer[0].zero_grad()
        actor_loss.backward()
        self.optimizer[0].step()

        alpha_loss = -(self.log_alpha * (logprob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.halpha = self.log_alpha.exp()

        # if self.learn_step_i % 5 == 0:
        self.soft_update(self.critic_net_target, self.critic_net)

    def save(self):
        path="mod/"+par.fileName+"/"
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.actor_net, path+"sac_actor_net.pt")  #
        torch.save(self.critic_net, path+"sac_critic_net.pt")  #
        torch.save(self.critic_net_target, path+"sac_critic_net_target.pt")  #

    def read(self):
        path="mod/"+par.fileName+"/"
        if os.path.exists(path+"sac_actor_net.pt") and os.path.exists(path+"sac_critic_net.pt"):
            self.actor_net = torch.load(path+"sac_actor_net.pt")  #
            self.critic_net = torch.load(path+"sac_critic_net.pt")  #
            self.critic_net_target = torch.load(path+"sac_critic_net_target.pt")  #
            self.optimizer = (torch.optim.Adam(self.actor_net.parameters(), lr=par.lr),
                              torch.optim.Adam(self.critic_net.parameters(), lr=par.lr))
rewardInd={0:-0.05,10:0.5,50:0.6,100:0.7,200:0.8,300:0.9}
preDis=100000
def sig(x):
    return 1 / (1 + math.exp(-x))
def setReward(reward,is_terminal):#设置奖励函数函数
    global preDis
    if is_terminal :
        if env.emptyCnt>0:
            return -11
        return 11
    if reward==0:
        dis = env.getDisFromFood()
        ans=0
        if dis>preDis:
            ans=-0.1
        else:
            ans = 0.1
        preDis=dis
        return ans
    return 11

def train(beginShape, cartPole_util=None):
    agent=Agent(beginShape)
    # agent.read()
    sumreward = 0
    pltData=[np.zeros(100),np.zeros(100)]
    pltData_i=0
    epi_time=10
    for episode_i in range(EPOCHS_CNT):#循环若干次
        is_terminal=False
        stList = agent.initState(env)
        frame_i=0
        while (not is_terminal):
            # env.render()
            action,prob_mask = agent.choose_action(stList[0:OVERLAY_CNT])
            # print(action)
            next_state, reward, is_terminal = env.step(action)
            stList = agent.nextState(stList, resetPic(next_state))  # 用stList储存前几步
            sumreward += reward
            frame_i+=1
            next_prob_mask=agent.saveNowMask(torch.ones([1,4]))
            reward = setReward(reward,is_terminal)
            buffer.pushRemember(stList,action,reward,prob_mask,next_prob_mask,1-is_terminal)
            if buffer.memory_i%5==0 and buffer.memory_i>MEMORY_CAPACITY:
                agent.learn()
        # if episode_i%10==0:
        #     # checkWD()

        if episode_i>0 and episode_i%epi_time==0 :
            sumreward /= epi_time
            s = str(episode_i) + "reward:" + str(sumreward)
            wInfo(s)
            pltData[0][pltData_i] = episode_i // epi_time
            pltData[1][pltData_i] = sumreward
            pltData_i += 1
            sumreward = 0
            if pltData_i==100:
                pltData_i=0
                plt.plot(pltData[0], pltData[1], marker='.', ms=10, color='red')
                f = plt.gcf()  # 获取当前图像
                f.savefig("reward.png")  # 保存图片
            if episode_i%2000==0:
                wInfo("save... :"+str(episode_i))
                agent.save()
    return agent

initInfo("begin!!")
if platform.system() =="Linux":
    agent=train(beginShape)
    agent.save()
# agent=train(shape)
# agent.save()
