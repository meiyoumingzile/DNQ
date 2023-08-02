import math
from collections import deque

import torch

# from baseline.douzero_encoder import Infoset, getModel, toDouzeroObs, _cards2array_1
# from baseline.env.env import Env
from DNQ.mygame.douDiZhu.doudizhu_encoder import getBaseFea, getAllActionFea
from doudizhu_game import Doudizhu, dfsPrintActList, Action, baselinePolicy, baselinePolicy_first, INF, \
    printDeckListForId
from doudizhu_cheat import mkDeck, cheat1, setHandCards
import random
isPrint=False
# douzero_models=getModel("douzero",0)
# print(douzero_models)
def bidFun(pid,already,actList):#叫牌网络
    act_id = random.randint(0, len(actList) - 1)

    return actList[act_id]
def INFfun(cardi,appendix,allActList):
    n=len(allActList)
    # dfsPrintActList(allActList)
    act_id=random.randint(0,n-1)
    return act_id
def policy(env:Doudizhu,nowPlayerid,maxPlayerid,allAct:list,INFfun):
    n=len(allAct)
    # id=random.randint(0,n-1)
    playerTag=(nowPlayerid-env.dealer+3)%3
    obs=toDouzeroObs(playerTag,Infoset(env,nowPlayerid,allAct))
    z_batch = torch.from_numpy(obs['z_batch']).float()
    x_batch = torch.from_numpy(obs['x_batch']).float()
    # print("大小", z_batch.shape, x_batch.shape)
    if torch.cuda.is_available():
        z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
    y_pred = douzero_models[playerTag].forward(z_batch, x_batch, return_value=True)['values'].detach().cpu()
    best_act_id = torch.argmax(y_pred, dim=0)[0].item()
    # print(best_act_id)
    id=best_act_id
    # print("使用："+allAct[id].toString())
    env.players[nowPlayerid].useAction(allAct[id],INFfun)
    if isPrint:
        if maxPlayerid==-1:
            print("先出玩家", nowPlayerid, end=" ")
        else:
            print("玩家", nowPlayerid, end=" ")
        dfsPrintActList(allAct)
        print("动作id:", id, end=" ")
        dfsPrintActList(allAct[id])
    return id,allAct[id]
env=Doudizhu()
for _ in range(0):
    env.reset()
    # deck,dealer=mkDeck(cheat1)
    # env.dealCards(deck, dealer)
    env.dealCards()
    isTer=False
    maxPlayerId=env.dealer
    winPlayer=-1
    while(winPlayer==-1):
        # env.printPlayerHand()
        que=deque(maxlen=2)
        allFirstAct = env.players[maxPlayerId].getAllFirstAction(all=True)
        env.players[maxPlayerId].printCards(False)
        print(env.players[maxPlayerId].getMinStep())
        maxact_id,maxact=baselinePolicy_first(env,maxPlayerId,-1,allFirstAct,INFfun)#env:Doudizhu,nowPlayerid,maxPlayerid,allAct:list,INFfun
        pid=maxPlayerId
        # print(maxact.playerId, maxPlayerId)
        # fea=getBaseFea(env,maxPlayerId,maxact)
        # fea = getAllActionFea(allFirstAct)
        # print(fea.shape)
        li=env.classActKind(allFirstAct)
        # dfsPrintActList(li)
        # Infoset()
        winPlayer=-1
        hisli=[maxact]
        while(True):
            # for i in range(3):
            #     ans = env.players[i].getAllFirstAction()
            #     dfsPrintActList(ans)
            pid = (pid + 1) % 3
            allAct=env.players[pid].getAllAction(maxact,all=True)
            act_id, act = baselinePolicy(env,pid,maxPlayerId,allAct,INFfun)
            hisli.append(act)
            maxPlayerId,maxact, winPlayer=env.step(maxact,act,maxPlayerId,pid)
            que.append(act)
            # env.printPlayerHand()
            if winPlayer!=-1 or len(que)==2 and que[0].isPass() and que[-1].isPass():
                break
        # dfsPrintActList(hisli)
    # print(winPlayer)
# act=Action([4,4,4,5,5,5],[INF,INF,INF,INF],playerId=0)
# env.reset()
# deck,dealer=mkDeck(cheat1)
# env.dealCards(bidFun=bidFun)
# print(env.scList,env.dealer)
# # printDeckListForId(deck)
# env.printPlayerHand()
# for i in range(3):
#     allAct = env.players[i].getAllFirstAction()
#     dfsPrintActList(allAct)

pid=0
# setHandCards(env.players[pid],["6","6","6","7","7","7","8","8","8","9","9","9","10","10","J","J","J","A","A"])
setHandCards(env.players[pid],["9","9","9","9","10","10","10","10"])
# env.players[pid].getHandList()
# print(env.players[pid].getDouCnt(False))
act=Action([6,6,6,7,7,7],[2,3])
ans = env.players[pid].getAllAction(act)
# ans=env.players[pid].getAllFirstAction()

# li=[]
# for a in ans:
#     if a.kind=="triseq_1":
#         li.append(a)
dfsPrintActList(ans)
ans=env.players[pid].getAllDouAction()
dfsPrintActList(ans)
print(env.players[pid].getMinStep())

# dfsPrintActList(allAct)
# arr=_cards2array_1([1,2,3,4,5,1,2,3,4,13])
# print(arr)