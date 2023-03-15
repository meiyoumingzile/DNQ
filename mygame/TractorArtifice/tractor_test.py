import random
import numpy as np
import tractor_game
from cheater import mkDeck, cheator1
from tractor_game import Action,randomUpdateINF
from baselinePolicy import baselineColdeck

def firstPlayerPolicy(selfAllAct):#第一个人的出牌策略
    maxLen = 0
    maxLenActList=[]
    for a in selfAllAct:#找出长度最长的动作
        maxLen=max(maxLen,a.len)
    for i in range(len(selfAllAct)):#找出长度最长的动作的列表
        if maxLen==selfAllAct[i].len:
            maxLenActList.append(selfAllAct[i])
    act_i=random.randint(0, len(maxLenActList) - 1)
    return maxLenActList[act_i]
def printCmp(a):
    if isinstance(a, int) or isinstance(a, np.int32) or isinstance(a, np.int64):
       return True
    else:
        return len(a)>1
def otherUseCards(env,firstPlayerID,nextID,firstKind,sortCardList2,cards):#使用的cards只能是单牌，对子，连对
    ans=[]
    if len(cards)==0:
        return None
    ansUp, ansDown, isHave = env.getAllAct(sortCardList2[nextID], env.players[nextID],cards)
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
    env._useCardsContainINF(env.players[nextID], ans, firstKind, randomUpdateINF,sortCardList2[nextID])
    # env.dfsPrintActList(ans)
    return ans
def randomPlayGame(env):#4个人双方随机游戏
    beginList=[39, 39, 23, 12, 26, 5, 53, 1, 38, 30, 46, 54, 48, 40, 36, 6, 28, 46, 26, 18, 7, 16, 2, 27, 5, 22, 20, 47, 41, 41, 34, 8, 3, 31, 30, 13, 16, 23, 15, 48, 13, 51, 4, 37, 44, 33, 25, 52, 34, 9, 37, 21, 3, 17, 50, 29, 24, 51, 49, 38, 35, 43, 24, 6, 18, 32, 22, 29, 7, 20, 11, 19, 15, 36, 14, 42, 27, 45, 14, 12, 50, 45, 52, 31, 11, 42, 40, 47, 33, 54, 32, 8, 28, 21, 10, 49, 9, 25, 53, 44, 1, 4, 17, 19, 10, 2, 35, 43]
    # deck1, setDecor, setNum, setDealer=mkDeck(cheator1)
    # env.dealCards(deck1, setDecor, setNum, setDealer)  # 发牌测试,*代表拆包元组
    env.dealCards()

    env.coldeck(baselineColdeck)#换底牌，baselineColdeck是最基本的换底牌策略
    # env.printAllCards()
    firstPlayerID=env.dealer
    sumSc=0
    isTer=False
    epoch=0
    while(not isTer):#开始出牌
        # env.printAllCards()
        # print("轮次：",epoch,"  先出牌玩家：",firstPlayerID)
        act = [None,None,None,None]
        allAct=[[],[],[],[],[]]
        sortCardList1=[[],[],[],[],[]]
        sortCardList2 = [[], [], [], [], []]
        for i in range(4):
            sortCardList2[i] = env.players[i].toSortCardsList2(env)#会重叠
            sortCardList1[i]=env.players[i].toSortCardsList1(sortCardList2[i],env)#去重

            # print(cards[i])
            # env.dfsPrintActList(sortCardList1[i])

            # allAct[i]=env.getMaxCards(sortCardList1[i],env.players[i])
            # env.dfsPrintActList(allAct[i])
        allAct[firstPlayerID]=env.getAllFirstAct(sortCardList1[firstPlayerID],env.players[firstPlayerID])
        # env.dfsPrintActList(allAct[firstPlayerID])#输出先手动作集合
        act[firstPlayerID] = firstPlayerPolicy(allAct[firstPlayerID])#获取动作
        isSeq, canSeq = env.judgeSeqUse(act[firstPlayerID], firstPlayerID, sortCardList2)
        if isSeq and canSeq == False:  # 如果不能甩
            print("不能甩！！！")
            if env.players[firstPlayerID].dealerTag < 2:  # 是庄家甩牌失败
                env.sumSc += 10
            else:
                env.sumSc = max(0, env.sumSc - 10)
            act[firstPlayerID] = act[firstPlayerID].getMinCard(env)  # 强制出最小的组子合
        # elif isSeq:
        #     print("能甩！！！")
        firstKind=env.getActKind(act[firstPlayerID])
        env.useCardsContainINF(env.players[firstPlayerID], act[firstPlayerID], firstKind, randomUpdateINF,sortCardList2[firstPlayerID])
        # print("玩家", firstPlayerID)
        # print(env.players[firstPlayerID].cards_decorList)
        # env.dfsPrintActList(sortCardList2[firstPlayerID])
        # env.dfsPrintActList(allAct[firstPlayerID],printCmp)
        # env.dfsPrintActList(act[firstPlayerID] )
        # print(firstKind)
        # act[firstPlayerID].println()
        for i in range(1,4):
            nextID=(firstPlayerID+i)%4
            act[nextID]= Action()
            # act[nextID].println()
            for a in act[firstPlayerID].one:
                li=otherUseCards(env, firstPlayerID, nextID, firstKind, sortCardList2, [a])
                act[nextID].add(li)
            for dou in act[firstPlayerID].double:
                act[nextID].addDou(otherUseCards(env, firstPlayerID, nextID, firstKind,sortCardList2, dou))
            # act[nextID].println()
        firstPlayerID,sc,isTer,info=env.game_step(act,firstPlayerID)#评价谁赢，返回赢者id,本轮分数(双方都会得分)，isTer是游戏有木有结束
        # reset
        # env.printAllInfo(act)
        if isTer :
            # env.printUnderCards()
            # print(firstPlayerID,str(env.sumSc))
            playerId=env.reset(env.sumSc)#重置游戏，-1代表继续,否则代表先达到A的玩家。
            # print(playerId,"\n")
            return playerId
        epoch+=1
    return 5



env= tractor_game.CC()

# randomPlayGame(env)
game_epoch=0
while(True):
    env.reset_game()
    for game_i in range(100):
        playerId=randomPlayGame(env)
        if(playerId!=-1):
            # print("先到到A的是:"+str(playerId%2)+","+str(playerId%2+2))
            break
        game_epoch+=1
    print(game_epoch)
# ans=[]
# for i in range(1,109):
#     ans.append(tractor_game.cardToString(i))
# print(ans)
