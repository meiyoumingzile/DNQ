import math
import random
import time
import platform
import torchvision
import cv2
import os
from collections import defaultdict
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.optim as optim
import torch.nn.functional as F#激活函数
from torchvision import transforms#图像处理的工具类
from torchvision import datasets#数据集216YYDS
from torch.utils.data import DataLoader#数据集加载
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#带上全部优化,double,dueling,Priority,noisenet
BATCH_SZ=512
EPOCHS_CNT=50000
TARGET_REPLACE_ITER=200#隔多少论更新参数
MEMORY_CAPACITY=1000#记忆容量,不能很小，否则学不到东西
MAX=10000
inf=0.0001
INF=1000000
ACTION_NUM=[3,4]
STATE_NUM=[3,5]
GAMA=0.965#reward_decay
LR=0.0001#学习率learning_rate,也叫lr
class Environment():
    def __init__(self):
        self.p_deck = np.array([
            [0.856, 0.144, 0, 0, 0],
            [0, 0.67, 0.33, 0, 0],
            [0, 0, 0.771, 0.229, 0],
            [0, 0, 0, 0.865, 0.135],
            [0, 0, 0, 0, 1]
        ])
        self.p_minor = np.array([
            [1, 0, 0, 0, 0],
            [0.85, 0.15, 0, 0, 0],
            [0.65, 0.27, 0.08, 0, 0],
            [0.45, 0.30, 0.17, 0.08, 0],
            [0.30, 0.35, 0.20, 0.08, 0.07]
        ])
        self.p_major = np.array([
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0.9, 0.08, 0.02, 0, 0],
            [0.7, 0.15, 0.10, 0.05, 0],
            [0.6, 0.18, 0.12, 0.07, 0.03]
        ])
        self.p_rebuild = np.array([
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ])
        r=360000
        self.riskReward=[0.01*r,	0.01*r,	0.02*r,	0.03*r,	0.1*r]
        self.actReward=[0,18000,72000,396000]
        self.actIndReward=[0, 4166.4672,3571.2576,428550.912]
        self.state_move=np.array([self.p_minor,self.p_major, self.p_rebuild])
        self.state_move0 = np.array([self.p_deck, self.p_ginder, self.p_sub])
        self.reset()
        self.stepMax=100
    def reset(self):
        self.step_i = 0
        self.state=np.zeros(3)
        return self.state

    def step(self,act):#act有3个数
        self.step_i+=1
        reward=0
        next_state=np.zeros(3)
        indirectReward=0
        for i in range(3):
            d=int(self.state[i])
            if act[i]==0:
                prob = self.state_move0[i][d]
            else:
                prob=self.state_move[act[i]-1][d]
            next_state[i]=np.random.choice(a=[i for i in range(STATE_NUM[1])], p=prob)
            reward+=self.riskReward[int(next_state[i])]+self.actReward[act[i]]
            indirectReward=max(indirectReward,self.actIndReward[act[i]])
        reward+=indirectReward
        return next_state,reward,self.step_i>=self.stepMax
env=Environment()
class QNET(nn.Module):#拟合Q的神经网络
    def __init__(self):
        super().__init__()
        self.advantage = nn.Sequential(#优势函数
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.fc=nn.Sequential(nn.Linear(32, ACTION_NUM[1]),nn.Linear(32, ACTION_NUM[1]),nn.Linear(32, ACTION_NUM[1]))
        self.value = nn.Sequential(
            nn.Linear(3,32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        for con in self.advantage:
            if con=="<class 'torch.nn.modules.linear.Linear'>":
                nn.init.normal_(con.weight, std=0.01)
                nn.init.constant_(con.bias, 0.1)
        for con in self.value:
            if con=="<class 'torch.nn.modules.linear.Linear'>":
                nn.init.normal_(con.weight, std=0.01)
                nn.init.constant_(con.bias, 0.1)
    def forward(self,x):#重写前向传播算法，x是张量
        x=x.cuda()
        x = x.view(x.size(0), -1)
        a=self.advantage(x)
        a0,a1,a2 = self.fc[0](a),self.fc[1](a),self.fc[2](a)
        v = self.value(x)
        return [v + a0 - a0.mean(),v + a1 - a1.mean(),v + a2 - a2.mean()]  # 决斗DQN,这里相加使用语法糖，实际上v 与a 维度不同
class Agent(object):#拟合Q的神经网络

    def __init__(self):
        self.eval_net, self.target_net = QNET().cuda(), QNET().cuda()#初始化两个网络，target and training
        self.eval_net.train()
        #eval_net是新网络，self.target_net是旧网络，target_net相当于学习的成果
        self.learn_step_i = 0  # count the steps of learning process
        self.memory_i = 0  # counter used for experience replay buffer
        self.memoryQue = [np.zeros((MEMORY_CAPACITY,3)),np.zeros((MEMORY_CAPACITY,3)),
                          np.zeros((MEMORY_CAPACITY, 3)),np.zeros((MEMORY_CAPACITY, 1)),np.zeros((MEMORY_CAPACITY, 1))]

        #self.memoryQue为循环队列
        self.lossFun=torch.nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=α)
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(), lr=LR)

    def greedy(self, x):  # 按照当前状态，用ϵ_greedy选取策略
        actVal = self.eval_net.forward(x.cuda())  # 从eval_net里选取结果
        act = torch.max(actVal, 1)[1].data.cpu().numpy()
        return act
    def choose_action(self, x,epsl):  # 按照当前状态，用ϵ_greedy选取策略
        if np.random.uniform(0,1)>epsl:#较大概率为贪心,小概率为随机
            x=torch.Tensor(x).unsqueeze(0).cuda()
            # print(x.shape)
            actVal=self.eval_net.forward(x)#从eval_net里选取结果
            act=[torch.max(actVal[i],1)[1].data.cpu().numpy()[0] for i in range(3)]
            return act
        else:
            act=[np.random.randint(0, ACTION_NUM[1]), np.random.randint(0, ACTION_NUM[1]), np.random.randint(0, ACTION_NUM[1])]
            # print(act)
            return act#随机选取动作

    def pushRemember(self,state,next_state,act,reward,is_ter):#把记忆存储起来，记忆库是个循环队列
        t=self.memory_i%MEMORY_CAPACITY#不能在原地址取模，因为训练时候要用memory_i做逻辑判断
        self.memoryQue[0][t] = state#存状态
        self.memoryQue[1][t] = next_state
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memoryQue[4][t] = np.array(is_ter)
        self.memory_i+=1
    def learn(self):#学习以往经验,frame_i代表当前是游戏哪一帧
        if  self.learn_step_i % TARGET_REPLACE_ITER ==0:#每隔TARGET_REPLACE_ITER步，更新一次旧网络
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_i+=1
        sampleList=np.random.choice(MEMORY_CAPACITY, BATCH_SZ)
        # randomMenory=self.memoryQue[np.random.choice(MEMORY_CAPACITY, BATCH_SZ)]#语法糖，传入一个列表，代表执行for(a in sampleList)self.memoryQue[a]
        #上述代表随机去除BATCH_SZ数量的记忆，记忆按照(state,act,reward,next_state)格式存储
        state=torch.FloatTensor(self.memoryQue[0][sampleList])#state
        next_state = torch.FloatTensor(self.memoryQue[1][sampleList])
        act = torch.LongTensor(self.memoryQue[2][sampleList].astype(int)).cuda()
        reward = torch.FloatTensor(self.memoryQue[3][sampleList]).cuda()
        is_ter = self.memoryQue[4][sampleList]
        q=self.eval_net.forward(state)
        # print(q[0].shape,act.shape,act[:,0:1].shape)
        q_eval = [q[0].gather(1, act[:,0:1]),q[1].gather(1, act[:,1:2]),q[2].gather(1, act[:,2:3]) ] # 采取最新学习成果

        q_next= self.target_net.forward(next_state)  # 按照旧经验回忆.detach()为返回一个深拷贝，它没有梯度
        loss=0
        for i in range(3):
            q_target= reward+GAMA*q_next[i].detach().max(1)[0].view(BATCH_SZ, 1) #不加double,dqn
            # print(q_eval[i].shape,q_target.shape)
            loss=loss+self.lossFun(q_eval[i],q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.target_net, "mod/target_net.pt")#保存模型pt || pth
        torch.save(self.eval_net, "mod/eval_net.pt")  #
    def read(self):
        self.target_net=torch.load( "mod/target_net.pt")#加载模型
        self.eval_net=torch.load("mod/eval_net.pt")  #
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=α)
def sigmoid(x):
    # 直接返回sigmoid函数
    return 1. / (1. + math.exp(-x))
prereward=10000000
def setReward(reward,is_terminal):#设置奖励函数函数
    global  prereward
    reward=math.log(reward)
    if prereward>reward:
        ans=1
    else:
        ans=-1
    prereward=reward
    return ans
def getepsl(frame_i):
    if frame_i>1000000:
        return 0.05
    return max(1-frame_i//1000*0.01,0.1)

def train():
    net=Agent()
    # net.read()
    sumreward = 0
    print("ready go!\n")
    frame_i=0
    for episode_i in range(EPOCHS_CNT):#循环若干次
        state=env.reset()
        is_terminal=False
        batchsum=0
        while(not is_terminal):
            frame_i+=1
            epsl=0.1
            # if net.memory_i<=MEMORY_CAPACITY:
            #     epsl=1
            # else:
            #     epsl = getepsl(frame_i)
            action = net.choose_action(state,epsl)  # 采用ϵ贪婪策略选取一个动作
            next_state,reward,is_terminal = env.step(action)
            sumreward += reward
            reward=setReward(reward,is_terminal)

            batchsum+=reward
            net.pushRemember(state,next_state, action, reward,is_terminal)#记下状态，动作，回报
            if net.memory_i==MEMORY_CAPACITY :
                print("learning")
            if net.memory_i%50==0 and net.memory_i>MEMORY_CAPACITY:  # 当步数大于脑容量的时候才开始学习，每五步学习一次
                # if net.memory_i%100==0:
                #     print(net.memory_i)
                net.learn()
        # print(batchsum)
        if episode_i%50==0 :
            print(str(episode_i)+"reward:"+str(sumreward/50))
            sumreward = 0
            # if episode_i%10000==0:
            #     print("episode_i:"+str(episode_i))
            #     net.save()
    return net
def test(beginShape):
    net = Agent(beginShape)
    net.read()
    for episode_i in range(EPOCHS_CNT):  # 循环若干次
        stList = net.initState(env)  # 用stList储存前几步
        is_terminal = False
        while (not is_terminal):
            env.render()
            action = net.greedy(stList[0:OVERLAY_CNT])  # 采用ϵ贪婪策略选取一个动作
            next_state, reward, is_terminal, info = env.step(action)
            stList = net.nextState(stList, resetPic(next_state))  # 用stList储存前几步
            time.sleep(0.01)
net=train()
# test(shape)

