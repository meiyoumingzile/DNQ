import math
import random
import numpy as np
from functools import cmp_to_key

import tractor_game
# from DNQ.mygame.TractorArtifice.cheater import mkDeck, cheator1
from tractor_game import Player,fenInd,getNum,Card,getDecor,CC
env=CC()
def _baselinecmp(a,b):#比较两组牌大小。返回1是a大，返回0是b大
    return a[1]>b[1] and 1 or -1
def _baselineColdeck(env:CC,p: Player):#换牌算法,cards有8张
    cards_i=0
    cards=np.zeros(tractor_game.UNDERCARD_CNT, dtype=int)
    # for i in range(5):
    #     env.printActList(p.cards_decorList[i], p.cards_decorLen[i])
    li1 = []
    li2=[]
    for i in range(4):
        if i!=p.lordDecor and p.cards_decorLen[i]>0:
            fen = 0
            for j in range(p.cards_decorLen[i]):#选择最小的牌扣掉
                num=getNum(p.cards_decorList[i][j])
                fen+=fenInd[num]
            if fen==0:
                li1.append((i,p.cards_decorLen[i]))
            else:
                li2.append((i, p.cards_decorLen[i]))
    if len(li1)>0:#先把没有分的花色扣掉
        li1=sorted(li1,key=cmp_to_key(_baselinecmp))
        for a in li1:
            i = a[0]
            j=a[1] - 1
            while j >-1:  # 选择最小的牌扣掉
                if j>0 and p.cards_decorList[i][j-1]==p.cards_decorList[i][j]:
                    j-=2
                    continue
                cards[cards_i] = p.cards_decorList[i][j]
                cards_i += 1
                env._useCard_decor(p,i,j)
                j-=1
                if cards_i == 8:
                    return cards
    if len(li2)>0:  # 有分的花色均匀扣
        li2 = sorted(li2, key=cmp_to_key(_baselinecmp))
        li2_len=[a[1] for a in li2]
        isCheck=True
        while isCheck:  #
            isCheck = False
            for k in range(len(li2)):
                i=li2[k][0]
                j=li2_len[k]-1
                if j<0:
                    continue
                isCheck=True
                if j>0 and p.cards_decorList[i][j-1]==p.cards_decorList[i][j]:#检测到对子
                    li2_len[k]-=2
                    continue
                num=getNum(p.cards_decorList[i][j])
                if fenInd[num]==0 and num!=1:#检测到非分单牌且不是A
                    cards[cards_i]=p.cards_decorList[i][j]
                    cards_i+=1
                    env._useCard_decor(p,i,j)
                    if cards_i==8:
                        return cards
                li2_len[k] -= 1

    #如果还有没得扣的，扣对子和分数
    for a in li1:# 剩余的对子扣掉
        i = a[0]
        for j in range(a[1] - 1, -1, -1):  # 剩余的对子扣掉
            if p.cards_decorList[i][j]!=0:
                cards[cards_i] = p.cards_decorList[i][j]
                cards_i += 1
                env._useCard_decor(p,i,j)
                if cards_i == 8:
                    env.sortPlayerHand(p)
                    return cards
    for a in li2:#  有分的花色的剩余的对子扣掉
        i = a[0]
        for j in range(a[1] - 1, -1, -1):  #  有分的花色的剩余的对子扣掉
            num = getNum(p.cards_decorList[i][j])
            if num==0 and p.cards_decorList[i][j]!=0:
                cards[cards_i] = p.cards_decorList[i][j]
                cards_i += 1
                env._useCard_decor(p,i,j)
                if cards_i == 8:
                    env.sortPlayerHand(p)
                    return cards
    for a in li2:#  有分的花色的剩余的分扣掉
        i = a[0]
        for j in range(a[1] - 1, -1, -1):  #  有分的花色的剩余的分扣掉
            num = getNum(p.cards_decorList[i][j])
            if num>0 and p.cards_decorList[i][j]!=0:
                cards[cards_i] = p.cards_decorList[i][j]
                cards_i += 1
                env._useCard_decor(p,i,j)
                if cards_i == 8:
                    env.sortPlayerHand(p)
                    return cards
    i=p.lordDecor
    for j in range(p.cards_decorLen[i] - 1, -1, -1):  #  有分的花色的剩余的分扣掉
        num = getNum(p.cards_decorList[i][j])
        if num == 0 and p.cards_decorList[i][j] != 0:
            cards[cards_i] = p.cards_decorList[i][j]
            cards_i += 1
            env._useCard_decor(p,i,j)
            if cards_i == 8:
                env.sortPlayerHand(p)
                return cards
    return cards
def baselineColdeck(env:CC,p: Player):#换牌算法,cards有8张
    # print("庄家是" + str(env.dealer))
    cards=_baselineColdeck(env,p)
    # print(sum_cnt)
    return cards
def baselineAct_followSmall1(player_cards,knowCardsMax,isFen=False):#无脑出小牌，

    for a in player_cards[2]:#卡看有没有拖拉机
        pass
    return 1
def baselineAct_checkCardsKind(knowCards0):#knowCards0是排序后的
    oneList,doubleList,traList=env.sortCardList1(knowCards0)
    if len(knowCards0)==1:
        return 1
    # elif :
    #
    # return

def baselineAct(p,knowCards,cardsCnt,knowCards_seq,maxi,kind):#player_cards代表自己玩家的手牌，knowCards是已知前置位的牌,knowCards_seq代表第几个出
    #cardsCnt代表knowCards每组牌的数量
    #maxi是最大的玩家的位置，kind代表出的花色，级牌算主牌的花色
    #player_cards通过排序被编码成了4个颜色，每个颜色又分为3组分别代表；oneList,doubleList,traList
    #
    cardsMax=knowCards[maxi]
    player_cards=p.toSortCardsList1(env)
    if knowCards_seq==0:#先出
        #算跑了多少分，如果外置位分数大于50，则无脑出大的。外置位分数计算方式：200-自己的分数-底牌的分数(如果可见)-已经出去的别人的分数
        #如果外置位分数不大于50，随机尽量出小的，可以出分
        #
        pass
    elif knowCards_seq==1:#第二个出
        #无脑大过0号，否则随机跟小牌且尽量不跟分
        if len(cardsMax[2])>0:#敌方又拖拉机
            if len(player_cards[kind][2])==0:#我方没拖拉机
                if len(player_cards[kind][1])==0:#我方没对子
                    if len(player_cards[kind][0])>=cardsCnt:#这一类花色有牌可出
                        cards=baselineAct_followSmall1()#跟小牌
                        return

            # else:#个数大于它且比他多
        for a in cardsMax[2]:  # 看看有没有拖拉机
            pass
        pass
    elif knowCards_seq == 2:#第三个
        #如果1号大，就无脑大过1号
        #如果0号大，且0号是王或级牌或大于1张的甩牌拖拉机对子，就跟分，没有分跟小牌
        #如果0号大，且0号较小，无脑大过他

        pass
    elif knowCards_seq == 3:#最后出牌，策略是：
        #如果我方大:就无脑跟分,能用分杀就用分杀,没有分就随机跟小牌;
        #如果敌方大:且没有分:能用分杀就用分杀，否则就随机跟小牌且尽量不跟分；
        #如果敌方大且有分:就尽量大过前面的，大不过就随机跟小牌且尽量不跟分
        pass

def randomPlayGame(env):#4个人双方随机游戏
    beginList=[47, 51, 26, 36, 5, 38, 36, 50, 9, 15, 6, 50, 4, 3, 53, 31, 16, 39, 20, 21, 45, 53, 3, 30, 46, 23, 7, 47, 29, 22, 28, 10, 52, 48, 12, 14, 27, 1, 5, 25, 44, 51, 42, 31, 54, 33, 13, 15, 34, 1, 48, 22, 38, 33, 37, 17, 43, 42, 25, 9, 18, 34, 10, 6, 28, 35, 32, 39, 2, 24, 40, 30, 12, 17, 26, 52, 14, 13, 4, 35, 11, 29, 27, 7, 8, 49, 11, 19, 2, 8, 40, 32, 45, 43, 20, 49, 19, 41, 21, 23, 24, 16, 46, 18, 41, 37, 44, 54]
    beginList1=[44, 9, 35, 35, 24, 3, 22, 36, 2, 38, 18, 33, 20, 8, 54, 39, 30, 16, 48, 43, 37, 43, 32, 45, 20, 4, 50, 24, 19, 49,
     22, 16, 37, 10, 17, 31, 13, 41, 12, 28, 7, 29, 18, 10, 46, 14, 53, 11, 7, 33, 26, 21, 27, 8, 31, 42, 1, 34, 52, 3,
     15, 49, 39, 26, 15, 4, 53, 21, 23, 28, 47, 51, 25, 23, 29, 38, 52, 34, 19, 30, 25, 5, 6, 41, 13, 9, 50, 27, 17, 46,
     48, 36, 11, 14, 45, 44, 1, 47, 40, 40, 5, 42, 6, 2, 32, 54, 12, 51]
    env.dealCards()  # 发牌测试,*代表拆包元组

    env.coldeck(baselineColdeck)#换底牌，baselineColdeck是最基本的换底牌策略
    firstPlayerID=env.dealer
    sumSc=0
    for epoch in range(tractor_game.HANDCARD_CNT):#开始出牌
        # env.printAllCards()
        # print("轮次：",epoch,"  先出牌玩家：",firstPlayerID)
        act = [[], [], [], [], []]
        allAct=[[],[],[],[],[]]
        sortCardList1=[[],[],[],[],[]]
        sortCardList2 = [[], [], [], [], []]
        for i in range(4):
            sortCardList2[i] = env.players[i].toSortCardsList2(env)
            sortCardList1[i]=env.players[i].toSortCardsList1(sortCardList2[i],env)

            # print(cards[i])
            # env.dfsPrintActList(sortCardList1[i])

            # allAct[i]=env.getMaxCards(sortCardList1[i],env.players[i])
            # env.dfsPrintActList(allAct[i])
        allAct[firstPlayerID]=env.getAllFirstAct(sortCardList2[firstPlayerID],env.players[firstPlayerID])
        act_id = random.randint(0, len(allAct[firstPlayerID]) - 1)
        # act_id = 13
        act[firstPlayerID] = allAct[firstPlayerID][act_id]
        # print("玩家", firstPlayerID)
        # print(act_id)
        # env.dfsPrintActList(act[firstPlayerID])
        firstKind=env.getActKind(act[firstPlayerID])
        env.useCardsContainINF(env.players[firstPlayerID], act[firstPlayerID], firstKind, randomUpdateINF)
        # env.dfsPrintActList(act[firstPlayerID])
        for i in range(1,4):
            nextID=(firstPlayerID+i)%4
            ansUp,ansDown,isHave = env.getAllAct(sortCardList2[nextID], env.players[nextID],act[firstPlayerID])
            # print("玩家",nextID)
            # env.dfsPrintActList(sortCardList2[nextID])
            # env.dfsPrintActList(ansUp)
            # env.dfsPrintActList(ansDown)
            nu=len(ansUp)
            nd=len(ansDown)
            if nd>0:
                act[nextID]=ansDown[random.randint(0,nd-1)]
            else:
                act[nextID] = ansUp[random.randint(0, nu - 1)]
            env.useCardsContainINF(env.players[nextID],act[nextID],firstKind,randomUpdateINF)


                # env.dfsPrintActList(tarAct)
            # env.printAllCards()

        firstPlayerID,sc,isTer,info=env.game_step(act,firstPlayerID)#评价谁赢，返回赢者id,本轮分数(双方都会得分)，isTer是游戏有木有结束
        if isTer :
            print(firstPlayerID,env.sumSc)
            # print(firstPlayerID,info)
            break


    # env.printAllCards()
    # p = env.players[0]
    # li = env.getMaxCards(p.toSortCardsList(), p)
    # env.printActList(env.underCards)
    # for i in range(5):
    #     env.printActList(li[i], len(li[i]))
# randomPlayGame(env)

# k=0
# while(True):
#     randomPlayGame(env)
#     k += 1
#     if env.reset(env.sumSc)!=-1:
#         print(env.dealer%2)
#         break
#以下是测试
# print( (0 - 1) % 13)

# env.dealCards()#发牌测试
# env.coldeck(baselineColdeck)
# p=env.players[0]
# li=env.getMaxCards(p.toSortCardsList(),p)
# env.printActList(env.underCards)
# for i in range(5):
#     env.printActList(li[i],len(li[i]))
# print(env.orderInd)
# for i in range(8):
#     Card(env.deck[i+100]).print()
# actList,doubleList,traList=env.sortCardList1([1,12,12,13,11,9,9,6,8,8,4,4,3,3,2,2,53,53,54,54,14,14,15,15,16,16,17])
# env.printActList(actList)
# env.printActList(doubleList)
# print(traList)
# print(to01code(env.players[0]))

