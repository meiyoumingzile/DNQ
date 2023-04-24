import math
import random
from functools import cmp_to_key

import numpy as np
import torch

import doudizhu_game
import baseline.douzero_encoder as douzero_encoder
import baseline.douzeroresnet_encoder as douzeroresnet_encoder
import baseline.perfectdou_encoder as perfectdou_encoder
from doudizhu_network import lstmInfea
from doudizhu_game import Doudizhu,Action,Player,dfsPrintActList
from doudizhu_encoder import handTo01code, getAllActionFea, getBaseFea, actTo01code, actZeros, getAllActionKindFea
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
        self.gamaPow = np.zeros((200))
        self.galanPow = np.zeros((200))#gama*lanta
        self.gamaPow[0]=self.galanPow[0]=1
        for i in range(1,200):
            self.gamaPow[i]=self.gamaPow[i-1]*args.gama
            self.galanPow[i]=self.galanPow[i-1]*args.gama*self.lanta
        self.initMenory()
        self.CLIP_EPSL = 0.1
        self.actPolicyFun={-2:self.chooseActFromDouzero,-3:self.chooseActFromDouzeroResnet,-4:self.chooseActFromPerfectDou,\
                            -1:self.chooseActFromNetMax,0:self.chooseActFromNet,1:self.chooseActFromNet,2:self.chooseAct_random2,\
                           3:self.chooseAct_random3,11:self.chooseActFromNet_withDiv
                           }
    def setNet(self,actor_net,critic_net):#设置网络
        self.net = actor_net
        self.critic_net = critic_net
    def setPlayer(self,p:Player):
        self.player=p
    def initPar(self,roundId,preActQue):
        self.preActQue=preActQue
        self.roundId=roundId
        ans=[]
        n=len(self.worker.historyActionList)
        if n==0:
            self.hisActTensor =torch.zeros((1,1,lstmInfea)).cuda(self.worker.cudaID)
        else:
            x=torch.stack(self.worker.historyActionList[-min(15,n):],dim=0)#unsqueeze(dim=0).cuda(self.worker.cudaID)
            if x.shape[0]%3!=0:
                x=torch.cat((torch.zeros(3-x.shape[0]%3,x.shape[1]),x),dim=0)
            self.hisActTensor=x.reshape(x.shape[0]//3,lstmInfea).unsqueeze(dim=0).cuda(self.worker.cudaID)
            # self.hisActTensor = torch.stack(self.worker.historyActionList[-min(n, 6):],dim=0).unsqueeze(dim=0).cuda(self.worker.cudaID)

    def initMenory(self):#初始化memory,来记录本局游戏的数据
        self.memoryRound = []
        self.memoryMask=[]
        self.memoryAllAct=[]
        self.memoryBaseFea = []
        self.memoryReward = []
        self.memoryAct1 = []
        self.memoryAct2 = []
        self.memoryHisFea=[]
        self.memoryIsDone = []
    def printMemory(self):
        dfsPrintActList(self.memoryAllAct)
        print(self.memoryRound)
        print(self.memoryReward)
        print(self.memoryAct1)
        print(self.memoryAct2)
    def pushMemory(self,roundId,mask,allActList,baseFea,hisFea,reward,act1,act2):#allAct内部是多个list
        self.memoryRound.append(roundId)
        self.memoryMask.append(mask)
        self.memoryAllAct.append(allActList)
        self.memoryBaseFea.append(baseFea)
        self.memoryReward.append(reward)
        self.memoryHisFea.append(hisFea)
        self.memoryAct1.append(act1)
        self.memoryAct2.append(act2)
        self.memoryIsDone.append(True)

    def chooseAct_random2(self,roundId,env,maxAgent,preActQue,allActionList:list):  # 无脑出个数最多的
        n = len(allActionList)
        allActionList.sort(key=cmp_to_key(doudizhu_game._sortAction_cmp1))
        newList = []
        prelen = -1
        for a in allActionList:
            if a.len != prelen:
                newList.append(a)
                prelen = a.len
        id = random.randint(0, len(newList) - 1)
        self.player.useAction(newList[id], doudizhu_game.baselinePolicy_INFfun)
        return newList[id]
    def chooseAct_random3(self,roundId,env,maxAgent,preActQue,allActionList:list):#随机出牌
        id = random.randint(0, len(allActionList) - 1)
        self.player.useAction(allActionList[id], doudizhu_game.baselinePolicy_INFfun)
        return allActionList[id]

    def epslBoltzmann(self,probList,mask, epsl=0.2):  # 修正玻尔兹曼,这里epsl是随机选取的概率
        # print(torch.isnan(probList))
        if torch.any(torch.isnan(probList)):
            print(probList,mask)
            print("have nan")
            exit()
        if mask!=None:
            cnt=mask.sum().item()
            p = epsl / cnt
            ans = probList * (1 - epsl) + p
            for i in range(probList.shape[0]):
                if mask[i] == 0:
                    ans[i] = 0
        else:
            cnt=probList.shape[0]
            # if cnt<1:
            #     print(probList,mask,probList.shape)
            p = epsl / cnt
            ans=probList*(1-epsl)+p
        actid = torch.multinomial(ans, num_samples=1, replacement=True)[0].item()
        return actid,ans
    def updateINF(self, cardi, appendix, allActList):
        baseFea = getBaseFea(self.env, self.player.id, self.preActQue,appendix).cuda(self.worker.cudaID)  # self.id
        hisActFea = self.hisActTensor

        allActFea = getAllActionFea(self.env,allActList).cuda(self.worker.cudaID)
        probList2 = self.net.forward_act(baseFea, hisActFea, allActFea).squeeze(dim=1)
        actid_2,probList2 = self.epslBoltzmann(probList2,None,self.worker.actepsl)
        actid_2_prob = probList2[actid_2].item()

        # dfsPrintActList(allActList)
        self.pushMemory(self.roundId,None,allActFea, baseFea,hisActFea, 0, None, (actid_2, actid_2_prob))
        return actid_2


    def chooseActFromNet_withDiv(self, roundId, env, maxAgent, preActQue,allActionList: list):  # ansUp代表较大的动作，ansDown代表较小的动作
        baseFea = getBaseFea(env,self.player.id,preActQue).cuda(self.worker.cudaID)  # self.id
        hisActFea=self.hisActTensor

        allActFea = getAllActionFea(self.env,allActionList).cuda(self.worker.cudaID)
        probList2 = self.net.forward_act(baseFea, hisActFea, allActFea).squeeze(dim=1)
        actid_2,probList2 = self.epslBoltzmann(probList2,None,self.worker.actepsl)
        actid_2_prob = probList2[actid_2].item()
        act=allActionList[actid_2]
        self.pushMemory(roundId,None,allActFea,baseFea,hisActFea,0,None,(actid_2,actid_2_prob))
        self.player.useAction(act, self.updateINF)
        return act
    def chooseActFromNet(self,roundId,env,maxAgent,preActQue,allActionList:list):#ansUp代表较大的动作，ansDown代表较小的动作
        # return self.chooseActFromNet_withDiv(roundId,env,maxAgent,preActQue,allActionList)
        baseFea = getBaseFea(env,self.player.id,preActQue).cuda(self.worker.cudaID)  # self.id
        allAct_kind = env.classActKind(allActionList)

        # print(self.worker.cudaID,next(self.net.parameters()).device)
        hisActFea=self.hisActTensor
        allActKind,allActFea1=getAllActionKindFea(self.env, allAct_kind)
        probList1 = self.net.forward_act(baseFea, hisActFea, allActFea1.cuda(self.worker.cudaID)).squeeze(dim=1)
        actid_1, probList1 = self.epslBoltzmann(probList1, None, self.worker.actepsl)
        actid_1_prob = probList1[actid_1].item()
        self.pushMemory(roundId, None, allActFea1, baseFea, hisActFea, 0, None, (actid_1, actid_1_prob))
        actid_1 = allActKind[actid_1]
        allActFea2 = getAllActionFea(self.env, allAct_kind[actid_1]).cuda(self.worker.cudaID)
        if len(allAct_kind[actid_1])<=1:
            actid_2 = 0
            actid_2_prob =1
        else:
            probList2 = self.net.forward_act(baseFea, hisActFea, allActFea2).squeeze(dim=1)
            actid_2,probList2 = self.epslBoltzmann(probList2,None,self.worker.actepsl)
            actid_2_prob = probList2[actid_2].item()
        act=allAct_kind[actid_1][actid_2]
        self.pushMemory(roundId,None,allActFea2,baseFea,hisActFea,0,None,(actid_2,actid_2_prob))
        self.player.useAction(act, self.updateINF)
        return act
    def chooseActFromNetMax(self,roundId,env,maxAgent,preActQue,allActionList:list):
        baseFea = getBaseFea(env,self.player.id,preActQue).cuda(self.worker.cudaID)  # self.id
        allAct_kind = env.classActKind(allActionList)

        # print(self.worker.cudaID,next(self.net.parameters()).device)
        hisActFea=self.hisActTensor
        probList1= self.net.forward_fp(baseFea, hisActFea).squeeze(dim=0)  # base的特征向量
        for i in range(len(allAct_kind)):
            if len(allAct_kind[i])==0:
                probList1[i]=-math.inf
        probList1 = torch.softmax(probList1, dim=0)

        actid_1 = torch.argmax(probList1).item()

        if len(allAct_kind[actid_1])<=1:
            actid_2 = 0
            actid_2_prob =1
        else:
            allActFea = getAllActionFea(self.env,allAct_kind[actid_1]).cuda(self.worker.cudaID)
            probList2 = self.net.forward_act(baseFea, hisActFea, allActFea).squeeze(dim=1)
            actid_2 = torch.argmax(probList2).item()
        act=allAct_kind[actid_1][actid_2]
        self.player.useAction(act, self.updateINF)
        return act
    def chooseActFromDouzero(self,roundId,env,maxAgent,preActQue,allActionList:list):
        return self.chooseActFromOtherNet(roundId, env, maxAgent, preActQue, allActionList, self.douzero_models,douzero_encoder)

    def chooseActFromDouzeroResnet(self,roundId,env,maxAgent,preActQue,allActionList:list):
        return self.chooseActFromOtherNet(roundId,env,maxAgent,preActQue,allActionList,self.douzeroResnet_models,douzeroresnet_encoder)

    def chooseActFromOtherNet(self,roundId,env,maxAgent,preActQue,allActionList:list,model,encoderPyshell):
        playerTag = (self.player.id - env.dealer + 3) % 3
        obs = encoderPyshell.toObs(playerTag, encoderPyshell.Infoset(env, self.player.id, allActionList))
        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        # print("大小", z_batch.shape, x_batch.shape)
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(self.worker.cudaID), x_batch.cuda(self.worker.cudaID)
        y_pred = model[playerTag].forward(z_batch, x_batch, return_value=True)['values'].detach().cpu()
        best_act_id = torch.argmax(y_pred, dim=0)[0].item()
        act=allActionList[best_act_id]
        self.player.useAction(act, self.updateINF)
        return act
    def chooseActFromPerfectDou(self,roundId,env,maxAgent,preActQue,allActionList:list):
        playerTag = (self.player.id - env.dealer + 3) % 3
        model=self.perfectdou_models[playerTag]
        obs=perfectdou_encoder.toObs(playerTag,perfectdou_encoder.Infoset(env, self.player.id, allActionList))
        act_str=perfectdou_encoder.getAct(model,obs)
        act_id,act=perfectdou_encoder.strToMyAction(act_str,allActionList)
        # print(act_str)
        # dfsPrintActList(allActionList)
        # print(act)
        return act

    def initLearn(self):#
        # print(self.memoryIsDone)
        up=len(self.memoryBaseFea)
        memory_i=0
        self.advantages = torch.zeros((up, 1)).cuda(self.worker.cudaID)
        while memory_i<up:
            statei = self.memoryBaseFea[memory_i]
            hisActi = self.memoryHisFea[memory_i]
            self.detlaList=[]
            prev = self.critic_net.forward(statei, hisActi)
            i=memory_i+1

            while(self.memoryIsDone[i]):#calc detla
                statei = self.memoryBaseFea[i]
                hisActi = self.memoryHisFea[i]
                reward = self.memoryReward[i-1]
                v = self.critic_net.forward(statei, hisActi)
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
            # print(memory_i, n, up)
        mean = self.advantages.mean()
        std = self.advantages.std()
        self.advantages = (self.advantages - mean) / (std)
        # print()
        # print(self.advantages)
        # print(self.memoryReward)
        # exit()
    def selflearn(self,worker,updateNetwork):#updateNetwork为更新函数
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
                mask= self.memoryMask[i]
                done=self.memoryIsDone[i]
                allActList=self.memoryAllAct[i]
                advantage = self.advantages[i].clone().detach()
                v = self.critic_net.forward(statei, hisActi)

                if done:
                    next_statei = self.memoryBaseFea[i + 1]
                    next_hisActi = self.memoryHisFea[i + 1]
                    next_v = self.critic_net.forward(next_statei, next_hisActi)
                    tderro = reward + self.gamaPow[1] * next_v - v  # 因为自动作没有奖励，所以Q(s,a)都是Gt
                else:
                    tderro= reward - v
                critic_loss = tderro.pow(2).mean()
                sumLoss_critic.append(critic_loss)
                # critic_loss.backward()
                # self.optimizer[0].step()

                if actPolicy1!=None:
                    probList1 = self.net.forward_fp(statei,hisActi).squeeze(dim=0)
                    actid=torch.Tensor([actPolicy1[0]]).cuda(self.worker.cudaID).long()
                    probList1=torch.softmax(probList1,dim=0)
                    _,probList1_new=self.epslBoltzmann(probList1, mask,self.worker.actepsl)
                    # print(probList1)
                    # print(nowRate)
                    nowRate=probList1_new.gather(0, actid)
                    # print(nowRate)
                    # print(actPolicy1[0],nowRate)
                    preRate =torch.Tensor([actPolicy1[1]]).cuda(self.worker.cudaID)#旧策略
                    # print(nowRate.shape,preRate.shape)
                    ratio = torch.exp(torch.log(nowRate) - torch.log(preRate))  # 计算p1/p2防止精度问题
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.CLIP_EPSL, 1 + self.CLIP_EPSL) * advantage
                    actor_loss_bin = (-torch.max(torch.min(surr1, surr2),( 1 - self.CLIP_EPSL)*advantage)-self.args.entropy_coef*torch.log(nowRate)).mean()
                    sumLoss_actor1.append(actor_loss_bin)

                allActFea=self.memoryAllAct[i].cuda(self.worker.cudaID)
                #下面是更新act网络
                probList2 = self.net.forward_act(statei,hisActi,allActFea).squeeze(dim=1)
                # print(probList2.shape)
                _,probList2_new=self.epslBoltzmann(probList2, None,self.worker.actepsl)
                act = torch.Tensor([actPolicy2[0]]).cuda(self.worker.cudaID).long()
                # print(probList)
                nowRate = probList2_new.gather(0, act)
                # print(probList2,act,nowRate)
                preRate = torch.Tensor([actPolicy2[1]]).cuda(self.worker.cudaID)  # 旧策略
                # print(nowRate.shape,preRate.shape)
                ratio = torch.exp(torch.log(nowRate) - torch.log(preRate))  # 计算p1/p2防止精度问题
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.CLIP_EPSL, 1 + self.CLIP_EPSL) * advantage
                actor_loss = (-torch.max(torch.min(surr1, surr2),( 1 - self.CLIP_EPSL)*advantage)-self.args.entropy_coef*torch.log(nowRate)).mean()
                sumLoss_actor2.append(actor_loss)
                # print(loss_actor,actor_loss)
                # actor_loss.backward()
                # self.optimizer[2].step()
            updateNetwork(self,worker,sumLoss_critic,sumLoss_actor1,sumLoss_actor2)