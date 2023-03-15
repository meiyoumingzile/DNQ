import random

import numpy as np
import torch
import tractor_game
from baselinePolicy import baselineColdeck
from tractor_game import CC,Action
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

    def firstPlayerPolicy(self,selfAllAct):  # 第一个人的出牌策略,需要神经网络决策
        maxLen = 0
        maxLenActList = []
        for a in selfAllAct:  # 找出长度最长的动作
            maxLen = max(maxLen, a.len)
        for i in range(len(selfAllAct)):  # 找出长度最长的动作的列表
            if maxLen == selfAllAct[i].len:
                maxLenActList.append(selfAllAct[i])
        act_i = random.randint(0, len(maxLenActList) - 1)
        return maxLenActList[act_i]

    def otherUseCards(self,env, firstPlayerID, nextID, firstKind, sortCardList2, cards):  # 使用的cards只能是单牌，对子，连对
        ans = []
        if len(cards) == 0:
            return None
        ansUp, ansDown, isHave = env.getAllAct(sortCardList2[nextID], env.players[nextID], cards)
        # print("玩家",nextID)
        # env.dfsPrintActList(sortCardList2[nextID])
        # env.dfsPrintActList(ansUp)
        # env.dfsPrintActList(ansDown)
        nu = len(ansUp)
        nd = len(ansDown)
        # print(ansUp,ansDown)
        if nu > 0:
            ans = ansUp[random.randint(0, nu - 1)]
        else:
            ans = ansDown[random.randint(0, nd - 1)]
        # env.dfsPrintActList(cards)
        env._useCardsContainINF(env.players[nextID], ans, firstKind, randomUpdateINF, sortCardList2[nextID])
        # env.dfsPrintActList(ans)
        return ans
    def playAGame(self,env):
        env.dealCards()
        # env.coldeck(baselineColdeck)  # 换底牌，baselineColdeck是最基本的换底牌策略
        env.coldeck(baselineColdeck)  # 换底牌，使用神经网络k
        isTer = False
        epoch = 0
        firstPlayerID = env.dealer

        while (not isTer):  # 开始出牌
            # env.printAllCards()
            print("轮次：", epoch, "  先出牌玩家：", firstPlayerID)
            act = [None, None, None, None]
            allAct = [[], [], [], [], []]
            sortCardList1 = [[], [], [], [], []]
            sortCardList2 = [[], [], [], [], []]
            for i in range(4):
                sortCardList2[i] = env.players[i].toSortCardsList2(env)  # 会重叠
                sortCardList1[i] = env.players[i].toSortCardsList1(sortCardList2[i], env)  # 去重
            allAct[firstPlayerID] = env.getAllFirstAct(sortCardList1[firstPlayerID], env.players[firstPlayerID])
            env.dfsPrintActList(allAct[firstPlayerID])  # 输出先手动作集合

            act[firstPlayerID] = self.firstPlayerPolicy(allAct[firstPlayerID])  # 获取动作
            isSeq, canSeq = env.judgeSeqUse(act[firstPlayerID], firstPlayerID, sortCardList2)
            if isSeq and canSeq == False:  # 如果不能甩
                print("不能甩！！！")
                if env.players[firstPlayerID].dealerTag < 2:  # 是庄家甩牌失败
                    env.sumSc += 10
                else:
                    env.sumSc = max(0, env.sumSc - 10)
                act[firstPlayerID] = act[firstPlayerID].getMinCard(env)  # 强制出最小的组子合

            firstKind = env.getActKind(act[firstPlayerID])
            env.useCardsContainINF(env.players[firstPlayerID], act[firstPlayerID], firstKind, randomUpdateINF,
                                   sortCardList2[firstPlayerID])
            act[firstPlayerID].println()

            for i in range(1, 4):
                nextID = (firstPlayerID + i) % 4
                act[nextID] = Action()
                # act[nextID].println()
                for a in act[firstPlayerID].one:
                    li = self.otherUseCards(env, firstPlayerID, nextID, firstKind, sortCardList2, [a])
                    act[nextID].add(li)
                for dou in act[firstPlayerID].double:
                    act[nextID].addDou(self.otherUseCards(env, firstPlayerID, nextID, firstKind, sortCardList2, dou))
                # act[nextID].println()
            firstPlayerID, sc, isTer, info = env.game_step(act, firstPlayerID)  # 评价谁赢，返回赢者id,本轮分数(双方都会得分)，isTer是游戏有木有结束
            # reset
            env.printAllInfo(act)
            if isTer:
                env.printUnderCards()
                print(firstPlayerID, str(env.sumSc))
                playerId = env.reset(env.sumSc)  # 重置游戏，-1代表继续,否则代表先达到A的玩家。
                print(playerId, "\n")
                return playerId
            epoch += 1
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
        env = CC()
        while True:
            # randomPlayGame(env)
            # while(True):
            env.reset_game()
            while(True):
                playerId = self.playAGame(env)#！=-1代表到A，本次游戏结束，返回赢得那个人。
                if (playerId != -1):
                    print("先到到A的是:" + str(playerId % 2) + "," + str(playerId % 2 + 2))
                    break



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

