import math

import numpy as np
import torch
from torch.autograd import Variable
import snake_network
from snake_network import NET_ACTOR,NET_CRITIC
import pyro
import pyro.distributions as dist
import snake_game
from snake_game import CC
import torch.nn.functional as F#激活函数
import torch.multiprocessing as mp

# start to define the workers...
class dppo_workers:
    def __init__(self,id,shared_obs_state, args):
        self.id=id
        self.args = args
        self.env = CC(False)

        # get the numbers of observation and actions...
        num_actions = snake_game.ACTION_SPACE_MAX
        # define the network...
        self.actor_net = NET_ACTOR(490,num_actions).cuda(self.args.cudaID)
        self.critic_net = NET_CRITIC(490,1).cuda(self.args.cudaID)
        self.preDis = 100000
        self.gama=0.97
        self.CLIP_EPSL=0.1
        self.lanta=0.2
        self.shared_obs_state=shared_obs_state
        self.memoryQue=[]
        self.memory_i=0
        self.memory_i = 0  # counter used for experience replay buffer//初始化记忆池
        self.memory_batch=50
        MEMORY_CAPACITY=2000
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        OVERLAY_CNT=1 # 针对此游戏的叠帧操作
        self.OVERLAY_CNT = OVERLAY_CNT
        beginShape= self.resetPic(self.env.reset()).shape
        self.memoryQue = [np.zeros((MEMORY_CAPACITY, OVERLAY_CNT) + beginShape),
                          np.zeros((MEMORY_CAPACITY, OVERLAY_CNT) + beginShape),
                          np.zeros((MEMORY_CAPACITY, 1)), np.zeros((MEMORY_CAPACITY, 1)),
                          np.zeros((MEMORY_CAPACITY, snake_game.ACTION_SPACE_MAX)), np.zeros((MEMORY_CAPACITY, 1))]
        # print(self.env.ini_borad)

    def pushRemember(self,state,next_state,act,reward,probList,ister):#原本是DQN的方法的结构，但这里用它当作缓存数据
        t=self.memory_i%self.MEMORY_CAPACITY
        self.memoryQue[0][t] = state
        self.memoryQue[1][t] = next_state
        self.memoryQue[2][t] = np.array(act)
        self.memoryQue[3][t] = np.array(reward)
        self.memoryQue[4][t] = np.array(probList.detach().numpy())
        self.memoryQue[5][t] = np.array(ister)
        self.memory_i=(self.memory_i+1)
    def resetPic(self,pic):
        # pic=np.array(pic).flatten()
        sz = self.env.SZ + 2
        pic = np.zeros(((sz * sz)))
        headPos = self.env.head.pos
        k = 0
        for i in range(0, sz):
            for j in range(0, sz):
                pic[k] = (self.env.board[i][j] == 0 or self.env.board[i][j] == -1)
                k += 1

        a = np.array([abs(headPos[0] - self.env.foodList[0][0]), abs(headPos[1] - self.env.foodList[0][1]),
                      abs(headPos[0] - self.env.foodList[1][0]), abs(headPos[1] - self.env.foodList[1][1]),
                      self.env.stepCnt, self.env.stepCntUp - self.env.nowstepCnt
                      ])
        pic = np.concatenate((pic, a), axis=0)
        pic=np.expand_dims(pic, axis=0)
        pic=self.shared_obs_state.normalize(pic)
        # print( self.shared_obs_state.normalize(pic))
        return pic

    def initState(self):  #
        state = self.resetPic(self.env.reset())  # 重新开始游戏,(210, 160)
        # print(self.env.board)
        stList = np.zeros((self.OVERLAY_CNT + 1,) + state.shape)
        stList[0]=state
        # li[OVERLAY_CNT]=state
        for i in range(1, self.OVERLAY_CNT):
            action = np.random.randint(0, snake_game.ACTION_SPACE_MAX)
            state, reward, is_terminal = self.env.step(action)
            stList[i] = self.resetPic(state)
        return stList

    def nextState(self, stList, state):
        for i in range(self.OVERLAY_CNT, 0, -1):
            stList[i] = stList[i - 1]
        stList[0] = state
        return stList

    def mask_action(self, x):
        infcnt = 0
        for i in range(len(self.env.fp)):
            act = self.env.fp[i]
            pos = (self.env.head.pos[0] + act[0], self.env.head.pos[1] + act[1])
            if self.env.board[pos[0]][pos[1]] > 0:
                x[0][i] = -math.inf
                infcnt += 1
        if infcnt == len(self.env.fp):
            for i in range(len(self.env.fp)):
                x[0][i] = 1
        return F.softmax(x, dim=1)
    def choose_action(self, x):  # 按照当前状态，选取概率最大的动作
        x = torch.FloatTensor(x).unsqueeze(0).cuda(self.args.cudaID)  # 加一个维度
        # print(x.shape)
        x = self.actor_net.forward(x)
        # print(self.env.head.pos,x)
        prob = self.mask_action(x).cpu()

        act = np.random.choice(a=snake_game.ACTION_SPACE_MAX, p=prob[0].detach().numpy())
        # print(act, prob)
        return act, prob[0]

    def sigmoid(self,x):
        # 直接返回sigmoid函数
        return 1. / (1. + np.exp(-x))
    def setReward(self,reward, is_terminal):  # 设置奖励函数函数
        if is_terminal:
            if self.env.emptyCnt > 0:
                return -1
            return 1
        if reward == 0:
            dis = self.env.getDisFromFood()
            ans = 0
            if dis >= self.preDis:
                ans = -self.sigmoid((50-dis)/40)/30
            else:
                ans = self.sigmoid((40-dis)/40)/30
            self.preDis = dis
            return ans
        return 1
    # start to define the training function...
    def calcDetla(self,mint,cnt=25):#要求state,next_state,reward得是tenSor
        t = np.array([i for i in range(mint,min(mint+cnt,self.MEMORY_CAPACITY,self.memory_i))])
        state = torch.FloatTensor(self.memoryQue[0][t]).cuda(self.args.cudaID)  # state
        next_state = torch.FloatTensor(self.memoryQue[1][t]).cuda(self.args.cudaID)
        reward = torch.FloatTensor(self.memoryQue[3][t]).cuda(self.args.cudaID)
        done = torch.FloatTensor(self.memoryQue[5][t]).cuda(self.args.cudaID)#是否继续，True代表继续
        v = (self.critic_net.forward(state), self.critic_net.forward(next_state))
        detla=reward + self.gama * v[1] * done - v[0]
        sum_detla=0
        e=1
        for a in detla:
            sum_detla+=a*e
            e*=(self.gama*self.lanta)
        return  sum_detla
    def train_network(self, traffic_signal, critic_counter, actor_counter, shared_critic_model, shared_actor_model,
                      shared_obs_state, critic_shared_grad_buffer, actor_shared_grad_buffer, reward_buffer):
        # update the parameters....
        # print("train_network" + str(self.id) + ": ", self.args)
        try:  # 加载保存的参数

            # print(shared_actor_model.state_dict())
            self.actor_net.load_state_dict(shared_actor_model.state_dict())
            self.critic_net.load_state_dict(shared_critic_model.state_dict())
            print("初始化完成",self.id)
        except ZeroDivisionError:  # 处理异常
            print("异常")  # 输出错误原因
        # print(self.env.ini_borad)
        self.__trainCnt=0
        while True:
            # update the parameters...
            # define the memory...

            self.reward_sum = 0
            for _ in range(self.args.collection_length):
                stList = self.initState()

                # state = shared_obs_state.normalize(state)
                game_step = 0
                is_terminal = False
                batch_reward=0
                self.preDis=100000
                self.memory_i=0#PPO算法中，不同游戏要重置经验池
                self.__trainCnt+=1
                while (not is_terminal):
                    # env.render()
                    # print(self.env.head.pos,self.env.board[self.env.head.pos[0]][self.env.head.pos[1]])
                    # print(self.env.ini_borad)
                    action, prob = self.choose_action(stList[0:self.OVERLAY_CNT])
                    # print(self.env.head.pos,action)
                    next_state, reward, is_terminal = self.env.step(action)
                    # print("ister",is_terminal)
                    stList = self.nextState(stList, self.resetPic(next_state))  # 用stList储存前几步
                    self.reward_sum  += reward
                    batch_reward+=reward
                    game_step += 1
                    # print(frame_i, batch_reward)
                    reward = self.setReward(reward, is_terminal)
                    self.pushRemember(stList[1:self.OVERLAY_CNT+1],stList[0:self.OVERLAY_CNT], action, reward, prob, 1 - is_terminal)

                    if batch_reward>0 and self.memory_i % self.MEMORY_CAPACITY == 0:
                        self.update_network(critic_shared_grad_buffer, actor_shared_grad_buffer, shared_critic_model,
                                        shared_actor_model,
                                        critic_counter, actor_counter, traffic_signal)
                if batch_reward > 0:
                    self.update_network(critic_shared_grad_buffer, actor_shared_grad_buffer, shared_critic_model,
                                        shared_actor_model,
                                        critic_counter, actor_counter, traffic_signal)
                print(str(game_step)+" "+str(batch_reward)+" ")
                if self.id==0 and self.__trainCnt%100==0 :
                    self.saveOneMod()
            # start to calculate the gradients for this time sequence...
            reward_buffer.add(self.reward_sum  / self.args.collection_length)

    # calculate the gradients based on the information be collected...
    def update_network(self, critic_shared_grad_buffer, actor_shared_grad_buffer,
                       shared_critic_model, shared_actor_model, critic_counter, actor_counter, traffic_signal):
        # t=(self.memory_i-1+self.MEMORY_CAPACITY)%self.MEMORY_CAPACITY
        t = np.random.choice(min(self.MEMORY_CAPACITY,self.memory_i), min(100,self.memory_i))
        # t[0] = (self.memory_i - 1+self.MEMORY_CAPACITY) % self.MEMORY_CAPACITY

        state = torch.FloatTensor(self.memoryQue[0][t]).cuda(self.args.cudaID)  # state
        next_state = torch.FloatTensor(self.memoryQue[1][t]).cuda(self.args.cudaID)
        act = torch.LongTensor(self.memoryQue[2][t].astype(int)).cuda(self.args.cudaID)
        reward = torch.FloatTensor(self.memoryQue[3][t]).cuda(self.args.cudaID)
        probList = torch.FloatTensor(self.memoryQue[4][t]).cuda(self.args.cudaID)
        done = torch.FloatTensor(self.memoryQue[5][t]).cuda(self.args.cudaID)#是否继续，True代表继续
        # print(probList.shape, act.shape, state.shape, next_state.shape)
        #torch.Size([1, 50, 4]) torch.Size([1, 50, 1]) torch.Size([50, 1, 1, 490]) torch.Size([50, 1, 1, 490])
        # state=F.normalize(state, p=2, dim=2)
        # next_state = F.normalize(next_state, p=2, dim=2)
        # print(state.shape)
        # calculate the discounted reward...
        # returns, advantages, old_action_prob = self.calculate_discounted_reward(state,next_state,act,reward,probList,ister)
        v = (self.critic_net.forward(state), self.critic_net.forward(next_state))

        tderro = reward + self.gama * v[1]*done - v[0]  # tderro也是优势函数,根据具体问题可以在is_ter==True时阻断
        shape=tderro.shape
        # returns = torch.Tensor(shape)
        # advantages = tderro.clone()

        # print(advantages)
        # advantages =torch.sigmoid(advantages)
        # print("归一化",advantages)

        #下面是GAE优化，计算优势函数时，不是直接tderro,而是
        previous_returns = 0
        previous_advantages = 0
        previous_value = 0
        # use gae here...
        advantages = tderro.clone()
        predicted_value=v[0].detach()
        # print(advantages.shape, predicted_value.shape)
        for i in range(len(t)):
            detla = self.calcDetla(t[i])
            # print(detla.shape)
            advantages[i][0]=detla.detach()
        # for idx in reversed(range(len(done))):
        #     if done[idx]:
        #         returns[idx, 0] = returns[idx] + self.args.gamma * previous_returns
        #         # deltas[idx, 0] = reward_batch[idx] + self.args.gamma * previous_value - predicted_value.data[idx, 0]
        #         # advantages[idx, 0] = deltas[idx, 0] + self.args.gamma * self.args.tau * previous_advantages
        #         advantages[idx, 0] = returns[idx, 0] - predicted_value.data[idx, 0]
        #     else:
        #         returns[idx, 0] = returns[idx]
        #         # deltas[idx, 0] = reward_batch[idx] - predicted_value.data[idx, 0]
        #         # advantages[idx, 0] = deltas[idx, 0]
        #         advantages[idx, 0] = returns[idx, 0] - predicted_value.data[idx, 0]
        #
        #     previous_returns = returns[idx, 0]
        #     previous_value = predicted_value.data[idx, 0]
        #     previous_advantages = advantages[idx, 0]
        # calculate the gradients...
        self.calculate_the_gradients(state,next_state,reward, advantages,act,probList,done,
                                                               critic_shared_grad_buffer, actor_shared_grad_buffer,
                                                               shared_critic_model, shared_actor_model, critic_counter,
                                                               actor_counter, traffic_signal)

    # calculate the gradients...
    def calculate_the_gradients(self,state,next_state,reward, advantages,act, probList,done,
                                critic_shared_grad_buffer,actor_shared_grad_buffer, shared_critic_model, shared_actor_model, critic_counter,
                                actor_counter, traffic_signal):

        # put the tensors into the Variable...
        # start to calculate the gradient of critic network firstly....
        for _ in range(self.args.value_step):#训练多次


            self.critic_net.zero_grad()
            # get the init signal...
            signal_init = traffic_signal.get()
            v = (self.critic_net.forward(state), self.critic_net.forward(next_state))
            tderro = reward + self.gama * v[1]*done - v[0]  # tderro也是优势函数,根据具体问题可以在is_ter==True时阻断
            # tderro = reward + self.gama * critic_net(next_state) * done - critic_net(state)
            critic_loss = tderro.pow(2).mean()
            # do the back-propagation...
            # print("cri_backward ")
            critic_loss.backward()
            # add the gradient to the shared_buffer...
            critic_shared_grad_buffer.add_gradient(self.critic_net)
            # after add the gradient, add the counter...
            critic_counter.increment()
            # wait for the cheif's signal...
            while signal_init == traffic_signal.get():
                pass
            self.critic_net.load_state_dict(shared_critic_model.state_dict())

        # start to update the critic_network....
        for _ in range(self.args.policy_step):#训练多次
            # get the init signal....
            self.actor_net.zero_grad()
            signal_init = traffic_signal.get()
            # start to process...

            nowRate = self.actor_net.forward(state)
            # print(probList.shape,nowRate.shape,act.shape,state.shape,next_state.shape)
            # 应该是.
            # torch.Size([batch_size, 4])
            # torch.Size([batch_size, 4])
            # torch.Size([batch_size, 1])
            # torch.Size([batch_size, 1, 1, 490])
            # torch.Size([batch_size, 1, 1, 490])
            # torch.Size([4]) torch.Size([1, 4]) torch.Size([1]) torch.Size([1, 1, 490]) torch.Size([1, 1, 490])
            for i in range(len(probList)):
                for j in range(len(probList[i])):
                    if probList[i][j] == 0:
                        nowRate[i][j] = -math.inf
            nowRate = F.softmax(nowRate, dim=1)
            # print(nowRate, probList)
            # print(nowRate.shape,probList.shape,act)
            nowRate = nowRate.gather(1, act)
            preRate = probList.gather(1, act).clone()
            # print(nowRate.shape,probList.shape)
            adv=advantages.detach()
            ratio = torch.exp(torch.log(nowRate) - torch.log(preRate))  # 计算p1/p2防止精度问题
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.CLIP_EPSL, 1 + self.CLIP_EPSL) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            # do the back propogation
            # print("act_backward ")
            actor_loss.backward()
            actor_shared_grad_buffer.add_gradient(self.actor_net)
            actor_counter.increment()
            while signal_init == traffic_signal.get():#被锁住了，等待更新共享网络
                pass
            self.actor_net.load_state_dict(shared_actor_model.state_dict())

        return critic_loss, actor_loss

    # this is used in the testing...
    def normalize_filter(self, x, mean, std):
        x = (x - mean) / (std + 1e-8)
        x = np.clip(x, -5.0, 5.0)

        return x

    def saveOneMod(self):
        if self.id==0:
            torch.save(self.actor_net, "mod/actor_net.pt")  #
            torch.save(self.critic_net, "mod/critic_net.pt")  #