from collections import deque

from DNQ.mygame.douDiZhu.doudizhu_encoder import getBaseFea, getAllActionFea
from doudizhu_game import Doudizhu, dfsPrintActList, Action, baselinePolicy, baselinePolicy_first, INF
from doudizhu_cheat import mkDeck, cheat1, setHandCards
import random
isPrint=False
def INFfun(cardi,appendix,allActList):
    n=len(allActList)
    # dfsPrintActList(allActList)
    act_id=random.randint(0,n-1)
    return act_id
def policy(env:Doudizhu,nowPlayerid,maxPlayerid,allAct:list,INFfun):
    n=len(allAct)
    id=random.randint(0,n-1)
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
        allFirstAct = env.players[maxPlayerId].getAllFirstAction()
        maxact_id,maxact=baselinePolicy_first(env,maxPlayerId,-1,allFirstAct,INFfun)#env:Doudizhu,nowPlayerid,maxPlayerid,allAct:list,INFfun
        pid=maxPlayerId
        # print(maxact.playerId, maxPlayerId)
        # fea=getBaseFea(env,maxPlayerId,maxact)
        # fea = getAllActionFea(allFirstAct)
        # print(fea.shape)
        li=env.classActKind(allFirstAct)
        dfsPrintActList(li)
        winPlayer=-1
        hisli=[maxact]
        while(True):
            # for i in range(3):
            #     ans = env.players[i].getAllFirstAction()
            #     dfsPrintActList(ans)
            pid = (pid + 1) % 3
            allAct=env.players[pid].getAllAction(maxact)
            act_id, act = baselinePolicy(env,pid,maxPlayerId,allAct,INFfun)
            hisli.append(act)
            maxPlayerId,maxact, winPlayer=env.step(maxact,act,maxPlayerId,pid)
            que.append(act)
            if winPlayer!=-1 or len(que)==2 and que[0].isPass() and que[-1].isPass():
                break
        # dfsPrintActList(hisli)
    print(winPlayer)
act=Action([4,4,4,5,5,5],[INF,INF,INF,INF],playerId=0)
print(act.kind)
# env.reset()
# deck,dealer=mkDeck(cheat1)
# env.dealCards(deck, dealer)
# pid=0
# setHandCards(env.players[pid],["A","A","10","10","10","10","J","J","J","J","Q","Q","Q","K","K","K","K"])
# act=Action([6,6,6,7,7,7,8,8,8],[2,2,3,3,4,4])
# allAct = env.players[pid].getAllAction(act)
# # print(act.kind)
# dfsPrintActList(allAct)
