import numpy as np
import torch
from encoder import handTo01code
from encoder import cardsTo01code
from tractor_network import AgentNet
import pyro
import pyro.distributions as dist


# start to define the workers...
from tractor_game import CC


class Dppoagent:
    def __init__(self,args):
        self.args=args
        in_fea=490#有待输入
        self.net = AgentNet(in_fea).cuda(self.args.cudaID)

    def update_network(self, critic_shared_grad_buffer, actor_shared_grad_buffer,
                       shared_critic_model, shared_actor_model, critic_counter, actor_counter, traffic_signal):
        pass
    def chooseAct(self):


class DppoWorkers:
    def __init__(self, id, shared_obs_state, args):
        self.id = id
        self.args = args
        self.env = CC()
        self.gama = 0.99
        self.CLIP_EPSL = 0.2
        self.lanta = 0.2
        self.shared_obs_state = shared_obs_state
        self.agentList=(Dppoagent(args),Dppoagent(args),Dppoagent(args),Dppoagent(args))


        self.memoryQue = []
        self.memory_i = 0  # counter used for experience replay buffer//初始化记忆池
        self.memory_batch = 50
        MEMORY_CAPACITY = 2000
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        OVERLAY_CNT = 1  # 针对此游戏的叠帧操作
        self.OVERLAY_CNT = OVERLAY_CNT
        beginShape = self.resetPic(self.env.reset()).shape
        self.memoryQue = [np.zeros((MEMORY_CAPACITY, OVERLAY_CNT) + beginShape),
                          np.zeros((MEMORY_CAPACITY, OVERLAY_CNT) + beginShape),
                          np.zeros((MEMORY_CAPACITY, 1)), np.zeros((MEMORY_CAPACITY, 1)),
                          np.zeros((MEMORY_CAPACITY, snake_game.ACTION_SPACE_MAX)), np.zeros((MEMORY_CAPACITY, 1))]


    def initAllState(self):#初始化游戏状态，返回开始时的状态
        self.env.reset()
        return
    def getGameState(self):
        li=[handTo01code(self.env.players[i]) for i in range(4)]

        return li,self.env.getNowUsedCards()
    # start to define the training function...
    def train_agent(self, traffic_signal, critic_counter, actor_counter, shared_model,
                      shared_obs_state, critic_shared_grad_buffer, actor_shared_grad_buffer, reward_buffer):
        try:  # 加载保存的参数

            # print(shared_actor_model.state_dict())
            for agent in self.agentList:
                agent.net.load_state_dict(shared_model.state_dict())
            print("初始化完成", self.id)
        except ZeroDivisionError:  # 处理异常
            print("共享参数异常")  # 输出错误原因
        # print(self.env.ini_borad)
        self.__trainCnt = 0
        while True:
            nowGrade = self.initAllState()
            self.env.reset_game()
            nowGrade=self.env.lordNum
            while (nowGrade<=14):#14是A
                isTer=False
                game_epoch=0
                while(not isTer):
                    self.getState()
                    act=#act是动作组合
                    firstPlayerID, sc, isTer, info = self.env.game_step(act,firstPlayerID)  # 评价谁赢，返回赢者id,本轮分数(双方都会得分)，isTer是游戏有木有结束

                    game_epoch+=1


    # calculate the gradients based on the information be collected...
    def update_network(self, brain_memory, critic_shared_grad_buffer, actor_shared_grad_buffer, \
                       shared_critic_model, shared_actor_model, critic_counter, actor_counter, traffic_signal):
        # process the stored information
        state_batch = torch.Tensor(np.array([element[0] for element in brain_memory]))
        reward_batch = torch.Tensor(np.array([element[1] for element in brain_memory]))
        done_batch = [element[2] for element in brain_memory]
        actions_batch = torch.Tensor(np.array([element[3] for element in brain_memory]))

        # put them into the Variables...
        state_batch_tensor = Variable(state_batch)
        actions_batch_tensor = Variable(actions_batch)
        # calculate the discounted reward...
        returns, advantages, old_action_prob = self.calculate_discounted_reward(state_batch_tensor, \
                                                                                done_batch, reward_batch,
                                                                                actions_batch_tensor)

        # calculate the gradients...
        critic_loss, actor_loss = self.calculate_the_gradients(state_batch_tensor, actions_batch_tensor, \
                                                               returns, advantages, old_action_prob,
                                                               critic_shared_grad_buffer, actor_shared_grad_buffer, \
                                                               shared_critic_model, shared_actor_model, critic_counter,
                                                               actor_counter, traffic_signal)

        return critic_loss.data.cpu().numpy()[0], actor_loss.data.cpu().numpy()[0]

