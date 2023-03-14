import math
import random
from functools import cmp_to_key

import numpy as np
import pygame
import pygame.locals
import torch
from pygame.locals import *
import sys
import time

CARDS_CNT=108
CARDS_CNT2=54
UNDERCARD_CNT=8
INF=1000
HANDCARD_CNT = (CARDS_CNT - UNDERCARD_CNT) // 4
fenInd=[0,0,0,0,0,5,0,0,0,0,10,0,0,10,0,0,0,0,0,0]
decorName=["♠","♥","♣","♦","王"]
NameTodecorId={"♠":0,"♥":1,"♣":2,"♦":3,"王":4}
numName=["","A","2","3","4","5","6","7","8","9","10","J","Q","K"]

def getKind(id,dec,num):  # id从1开始，判断类别，会把级牌和王算进主牌里
    if id<1:
        return 0
    id = (id - 1) % CARDS_CNT2 + 1
    d = (id - 1) // 13  # 黑桃是0，红糖是1，梅花是2，方片是3，王是4
    n= (id - 1) % 13 + 1  # 点数
    if d==4 or n==num:
        return dec
    return d
def getDecor(id):  # id从1开始，判断花色
    return int((id - 1) // 13)
def getNum(id):  # id从1开始，判断点数
    return int((id - 1) % 13+1)
def getDecorAndNum(id):
    if id<1:
        return 0,0
    id = (id - 1) % CARDS_CNT2 + 1
    decor = (id - 1) // 13  # 黑桃是0，红糖是1，梅花是2，方片是3，王是4
    num = (id - 1) % 13 + 1  # 点数
    return int(decor),int(num)
def getKindAndNum(id,dec,num):
    if id<1:
        return 0,0
    id = (id - 1) % CARDS_CNT2 + 1
    d = (id - 1) // 13  # 黑桃是0，红糖是1，梅花是2，方片是3，王是4
    n = (id - 1) % 13 + 1  # 点数
    if d==4 or n==num:## 黑桃是0，红糖是1，梅花是2，方片是3，王是4
        return dec,num
    return int(d),int(n)
def toIndex(decor,num):
    return num+decor*13
def cardToString(id,up=False):#
    decor, num = getDecorAndNum(id)
    if id == INF:
        return "<任意>"
    elif num == 0:
        return "<null>"
    elif decor == 4:
        return "<"+ (num == 1 and "小" or "大") + "王>"
    else:
        return "<" + decorName[decor] + numName[num] + ">"
stringToCardId={}
def initStringToCard():
    stringToCardId["INF"]=INF
    stringToCardId["大王"] = 54
    stringToCardId["小王"] = 53
    for i in range(1,CARDS_CNT2+1):
        stringToCardId[cardToString(i)]=i
        stringToCardId[cardToString(i).lower()] = i
        stringToCardId[cardToString(i)[1:-1]] = i
        stringToCardId[str(i)] = i
initStringToCard()
# def getKind(id,lord):#得到类别#主牌算主的花色,lord为级牌
#     kind = (id - 1) // 13  # 黑桃是0，红糖是1，梅花是2，方片是3，王是4
#     num = (id - 1) % 13 + 1  # 点数，[1,13]王是14
#     if num == lord:
#         kind
#     return
class Action():
    def __init__(self, one=[],double=[]):  # double里包含拖拉机，如[[3,3],[4,4,5,5]]
        self.one=one.copy()
        self.double = double.copy()
        self.len=len(one)
        for dou in double:
            self.len+=len(dou)
    def add(self,one=[],double=[]):
        for a in one:
            self.one.append(a)
        self.len += len(one)
        for dou in double.copy():
            self.double.append(dou)
            self.len += len(dou)
    def addOne(self,a):
        self.one.append(a)
        self.len += 1
    def addDou(self,dou):
        self.double.append(dou.copy())
        self.len+=len(dou)
    def isCombination(self):#判断是否为甩牌
        return len(self.one)+len(self.double)>1
    def getDouleCnt(self):#返回对子数量
        return (self.len-len(self.one))//2
    def getDouleLen(self):#返回对子数组长度
        return len(self.double)
    def isSeq(self):#是否为甩牌
        return len(self.double)+len(self.one)>1
    def getFen(self):
        sc=0
        for dou in self.double:
            for a in dou:
                num = getNum(a)  # 点数，[1,13]王是14
                sc += fenInd[num]  # 分数
        for a in self.one:
            num = getNum(a)  # 点数，[1,13]王是14
            sc += fenInd[num]  # 分数
        return sc
    def print(self,i=0):
        print("act"+str(i)+":",end="")
        i=0
        for dou in self.double:
            for a in dou:
                Card(a).print(i)
                i+=1
        for a in self.one:
            Card(a).print(i)
            i+=1
    def println(self,i=0):
        self.print(i)
        print("")
    def toString(self):
        ans=""
        for dou in self.double:
            for a in dou:
                ans+=cardToString(a)
        for a in self.one:
            ans+=cardToString(a)
        return ans

    def tolist(self):
        li=self.one.copy()
        for dou in self.double:
            for a in dou:
                li.append(a)
        return li
    def toTensor(self):
        x=torch.zeros((2,2))
        return x
    # def sort(self):
    #     self.one=sorted(self.one, key=cmp_to_key(self._sortCardList_cmp1))

class Card():
    def __init__(self,id,holder=-1):#id从1开始，前
        self.id=id
        if id<109:
            self.decor,self.num=getDecorAndNum(id)
            self.sc=fenInd[self.num] #分数

        self.holder=holder#持有者
    def print(self,i=0):
        if self.id==INF:
            print("<i:{} ".format(i) + "任意 >", end=" ")
        elif self.num==0:
            print("<i:{} ".format(i)+ "null >", end=" ")
        elif self.decor==4:
            print("<i:{} ".format(i)+(self.num==1 and "小"or"大") + "王>", end=" ")
        else:
            print("<i:{} ".format(i)+decorName[self.decor]+numName[self.num]+">",end=" ")
    def println(self,i=0):
        self.print(i)
        print("")

class Player():
    def __init__(self,id):
        self.id=id
        self.dealerTag=0 #0是庄家，1是庄家跟班，23是闲家
        self.initCards()
    def initCards(self):
        self.cards = np.zeros(HANDCARD_CNT + 8,dtype='int')
        self.cards_i = 0
        self.cards_decorList = [np.zeros(HANDCARD_CNT + 8,dtype='int'), np.zeros(HANDCARD_CNT + 8,dtype='int'), np.zeros(HANDCARD_CNT + 8,dtype='int'),
                                np.zeros(HANDCARD_CNT + 8,dtype='int'), np.zeros(HANDCARD_CNT + 8,dtype='int')]  # 不同花色放在一起
        self.cards_cnt=np.zeros((CARDS_CNT2+1),dtype='int')
        self.cards_decorLen = [0, 0, 0, 0, 0]
        #cards_maxList代表已知的牌中，比它大的牌的数量
        self.cardsCnt = 0
        self.cards_lord = np.zeros(HANDCARD_CNT + 8,dtype='int')
    def initCards_usedCards_cnt(self,env):#初始化usedCards_cnt
        z0 = np.zeros(13 + 5, dtype='int')
        self.usedCards_cnt=[z0.copy(), z0.copy(), z0.copy(),
                                z0.copy(), z0.copy()]
        self.usedCards_len=[0, 0, 0, 0, 0]
        self.lordNum_cnt=np.zeros(4,dtype='int')

        if self.lordDecor==4:
            self.usedCards_len[4]=2#无主情况下主牌只有4张
            for i in range(4):
                self.usedCards_len[i]=13
        else:
            for i in range(4):
                self.usedCards_len[i] = 12
            self.usedCards_len[self.lordDecor] += 4  # 大小王级牌和主级牌
        self.numCardKind=env.getOrderID(54)-3
        for i in range(5):
            for j in range(self.cards_decorLen[i]):
                a=self.cards_decorList[i][j]
                if getNum(a)==self.lordNum and a<53:#是级牌
                    self.lordNum_cnt[getDecor(a)]+=1
                k=env.getOrderID(a)
                # print(a,k)
                self.usedCards_cnt[i][k]+=1
        if self.dealerTag==0:
            for a in env.underCards:
                k=env.getOrderID(a)
                kind,num=getKindAndNum(a,self.lordDecor,self.lordNum)
                self.usedCards_cnt[kind][k] += 1
                if kind!=self.lordDecor and num==self.lordNum:
                    self.lordNum_cnt[kind]+=1
    def getHandCardCnt(self):
        return self.cardsCnt

    def setNum(self,num):
        self.lordNum = num
    def setLord(self,decor,num):
        self.lordNum = num
        self.lordDecor = decor
        self.cards_lord = self.cards_decorList[decor]
    def getLordCnt(self):
        if self.lordDecor!=4:
            return self.cards_decorLen[4]+self.cards_decorLen[self.lordDecor]
        return self.cards_decorLen[4]
    def getSelfMaxCard(self,decor,cmp):#cmp是比较函数，返回i，是牌的次序号，代表大于i的牌都满足cmp，i是从大到小第一个不满足cmp的牌顺序的编号。
        # 该函数返回自己牌以及自己见到的牌当中，某种花色从大到小第一张不满足cmp的顺序
        for i in range(self.usedCards_len[decor]-1,-1,-1):
            if i==self.numCardKind:
                for j in range(4):
                    if j!=self.lordDecor and not cmp(self.lordNum_cnt[j]):
                        return i
            elif not cmp(self.usedCards_cnt[decor][i]):
                return i
        return -1
    def toSortCardsList1(self,sortCardList2,env):#去重
        li=[]
        for i in range(5):
            # oneList, doubleList, traList = env.sortCardList1(self.cards_decorList[i])
            li.append((env.sortCardList1(sortCardList2[i][0],sortCardList2[i][1],sortCardList2[i][2])))
        return li
    def toSortCardsList2(self,env):#相互包含
        li=[]
        for i in range(5):
            # oneList, doubleList, traList = env.sortCardList1(self.cards_decorList[i])
            li.append((env.sortCardList2(self.cards_decorList[i][0:self.cards_decorLen[i]])))
        return li
    def printCards(self):
        print("玩家{}".format(self.id),end=" ")
        k=0
        for i in  range(5):
            for j in range(self.cards_decorLen[i]):
                Card(self.cards_decorList[i][j]).print(k)
                k+=1
        print("")
class CC():
    def __init__(self):

        self.reset_game()
    def __reset(self):
        self.players = [Player(0), Player(1), Player(2), Player(3)]  # 0代表没有
        self.playerTable = [np.zeros((HANDCARD_CNT),dtype='int'), np.zeros((HANDCARD_CNT),dtype='int'), np.zeros((HANDCARD_CNT),dtype='int'),
                            np.zeros((HANDCARD_CNT),dtype='int')]  # 已经出去的卡牌
        self.playerTable_i = [0,0,0,0]  # 已经出过的牌的编号
        self.lordDecor = -1  # 主牌花色
        self.sumSc = 0  # 闲家得分
        self.underCards = np.zeros(UNDERCARD_CNT,dtype='int')  # 底牌
        self.nowDealCardPlayer = 0  # 当前到发牌的玩家
        self.round_i = 0  # 轮数
        self.deck = np.arange((CARDS_CNT),dtype='int') + 1  # 卡组
        self.useCards_i=0#放回deck
        for i in range(0,CARDS_CNT2):
            self.deck[i+CARDS_CNT2]=self.deck[i]

    def reset_game(self):#重置游戏进度
        self.__reset()
        self.round_lordNum = [2, 2]
        self.dealer = -1
        self.lordNum = 2  # 级牌
        for i in range(4):
            self.players[i].setNum(2)
            self.players[i].round_lordNum=2
        # self.setOrder()
        # li1 = [1,5,2,2,3,3,4,4,13,13,12]
        # li2 = [14,14,15,15,16,16,17,17,22,23,22]
        # self.judgeUseSeq(li1)

        # actList1,c2,li=self.sortCardList(li1)#
        # actList1, c2, li = self.sortCardList(li1)  #
        # actList2, c2, li = self.sortCardList(li2)  #
        # self.printCardsList(actList1)
        # self.printCardsList(actList2)
        # print(self.useCmpCards(li1,li2))
        # print(self.deck)
    def reset(self,preSc):#更新级牌,但不重置游戏进度,preSc代表上局闲家的分数,返回-1代表继续游戏，返回庄家编号代表谁先达到A
        self.__reset()
        g = self.getGrade(preSc)#结算等级
        if g>=0:#换庄
            self.dealer =(self.dealer+1)%4#换庄
            a = self.round_lordNum[self.dealer%2]
            if 2<=a <= 5:  # 5，10，k不能跳
                self.round_lordNum[self.dealer%2] = min(a + g, 5)
            elif 5<a <=10:
                self.round_lordNum[self.dealer%2] = min(a + g, 10)
            elif 10<a <=13:
                self.round_lordNum[self.dealer%2] = min(a + g, 13)
            else:
                self.round_lordNum[self.dealer%2]=1
                if g>0:
                    return self.dealer
        else:
            g = -g
            self.dealer = (self.dealer + 2) % 4  # 换同伙坐庄
            a = self.round_lordNum[self.dealer%2]
            if 2 <= a < 5:  # 5，10，k不能跳
                self.round_lordNum[self.dealer % 2] = min(a + g, 5)
            elif 5 <= a < 10:
                self.round_lordNum[self.dealer % 2] = min(a + g, 10)
            elif 10 <= a < 13:
                self.round_lordNum[self.dealer % 2] = min(a + g, 13)
            else:
                self.round_lordNum[self.dealer % 2] = 1
                if a + g>14:
                    return self.dealer
        self.lordNum=self.round_lordNum[self.dealer%2]
        return -1
    def setDealerID(self,dealer):
        self.players[dealer].dealerTag=0
        self.players[(dealer+1)%4].dealerTag = 2
        self.players[(dealer+2)%4].dealerTag = 1
        self.players[(dealer + 3) % 4].dealerTag =3
    def addCard(self,p:Player,cardID):#isDealCard代表是不是发牌阶段，is=True代表是
        cardID = (cardID - 1) % CARDS_CNT2 + 1
        decor, num = getDecorAndNum(cardID)
        if num == self.lordNum:
            decor = 4
        p.cards[p.cards_i] = cardID
        p.cards_i += 1
        p.cards_decorList[decor][p.cards_decorLen[decor]] = cardID  # 不同花色放在一起
        p.cards_decorLen[decor] += 1
        p.cardsCnt += 1
        p.cards_cnt[cardID] += 1
    def _useCard_decor(self,p:Player,decor,j):
        cardID=p.cards_decorList[decor][j]
        for i in range(p.cards_i):
            if p.cards[i] == cardID:
                p.cards[i] = 0
                break
        p.cards_decorList[decor][j]=0

    def __delCard(self, p: Player, i, kind):  #删除一个牌,但不会更新牌组
        cardID=p.cards_decorList[kind][i]
        p.cards_decorList[kind][i] = 0  # 变成0
        # p.cards_decorLen[kind] -= 1
        p.cardsCnt -= 1
        p.cards_cnt[cardID] -= 1
        self.deck[self.useCards_i]=cardID
        self.playerTable[p.id][self.playerTable_i[p.id]] = cardID  # 放入弃牌堆
        self.playerTable_i[p.id]+=1
        self.useCards_i+=1
        order = self.getOrderID(cardID)
        decor,num=getDecorAndNum(cardID)
        for i in range(4):
            if i != p.id:
                self.players[i].usedCards_cnt[kind][order] += 1
                if num == self.lordNum and decor<4:
                    self.players[i].lordNum_cnt[decor]+=1
    def getNowUsedCards(self):
        cardList=self.deck[0:self.useCards_i]
        return cardList
    def __useCard(self, p: Player, cardID,kind):  # isDealCard代表是不是发牌阶段，isPlay=True代表是出牌时
        cardID = (cardID - 1) % CARDS_CNT2 + 1
        for i in range(p.cards_decorLen[kind]):
            if p.cards_decorList[kind][i]==cardID:
                self.__delCard(p,i,kind)
                return

    def useCards(self,p:Player,cards):#使用一组牌
        decInd=[0,0,0,0,0]
        for a in cards:
            kind=getKind(a,self.lordDecor,self.lordNum)
            self.__useCard(p,a,kind)
            decInd[kind]+=1
        for k in range(5):
            if decInd[k]==0:
                continue
            j=0
            for i in range(p.cards_decorLen[k]):
                if p.cards_decorList[k][i]>0:
                    p.cards_decorList[k][j]=p.cards_decorList[k][i]

                    j+=1
            p.cards_decorLen[k]=j
    def _updateSortCardList(self,sortCardList:list,act:list):
        if sortCardList == None:
            return
        for a in act:  # 刷新sortCardList
            kind, num = getKindAndNum(a, self.lordDecor, self.lordNum)
            li = sortCardList[kind]  # 遍历这个花色
            for i in range(len(li[0])):  # 从单牌里删除
                if li[0][i] == a:
                    li[0].pop(i)
                    break
            for i in range(len(li[1])):  # 从对子里删除
                if li[1][i] == a:
                    li[1].pop(i)
                    break
            # print(sortCardList)
            for i in range(len(li[2])):
                # print(li[2])
                m = len(li[2][i])
                for j in range(m):  # 遍历每个拖拉机
                    if li[2][i][j] == a:
                        tra = li[2].pop(i)
                        beginTra = tra[:j]
                        endTra = tra[j + 1:]
                        if len(beginTra) > 1:  # 剩下的长度大于1，放回li[2]
                            li[2].append(beginTra)
                        if len(endTra) > 1:  # 剩下的长度大于1，放回li[2]
                            li[2].append(endTra)
                        break
                else:#这段用来代替goto,表示找到if时，跳出2层循环。python用goto要单独安装包
                    continue
                break
            # print(sortCardList)
            li[2].sort(key=cmp_to_key(self._sortCardList_cmp2))
    def _useCardsContainINF(self,p:Player, act:list, kind,fun,sortCardList=None):#sortCardList代表分好类的牌，这个牌也会跟着变
        actlen=len(act)
        if actlen==0:
            return
        m=0
        while(m<actlen):
            if act[m] == INF:
                break
            m+=1
        self.useCards(p,act[0:m])
        if m==actlen:
            self._updateSortCardList(sortCardList,act)
            return
        decInd = [0, 0, 0, 0, 0]
        for i in range(m,actlen):#寻找所有INF
            k,j=fun(p,act[0:i], kind)#返回卡牌编号)
            act[i]=p.cards_decorList[k][j]
            self.__delCard(p, j,k)
            decInd[k] += 1
        for k in range(5):
            if decInd[k] == 0:
                continue
            j = 0
            for i in range(p.cards_decorLen[k]):
                if p.cards_decorList[k][i] > 0:
                    p.cards_decorList[k][j] = p.cards_decorList[k][i]
                    j += 1
            p.cards_decorLen[k] = j
        self._updateSortCardList(sortCardList, act)

    def useCardsContainINF(self,p:Player, act:Action, kind,fun,sortCardList=None):  # 替换动作里的INF,传入sortCardList2代表该玩家的手牌分组，会互相重叠，act是已经选好的动作，kind是先手玩家所出的颜色。fun是跟牌的决策策略
        self._useCardsContainINF(p,act.one,kind,fun,sortCardList)
        for dou in act.double:
            self._useCardsContainINF(p,dou, kind, fun,sortCardList)
    def checkReDealCards(self):
        if self.players[0].getLordCnt()+ self.players[2].getLordCnt()<10:
            return True
        if self.players[1].getLordCnt()+ self.players[3].getLordCnt()<10:
            return True
        for i in range(4):
            if self.players[i].getLordCnt()>=HANDCARD_CNT-2: #一个人几乎全是主牌，可以重开
                return True
            fen=0
            p=self.players[i]
            for j in range(HANDCARD_CNT):
                fen+=fenInd[getNum(int(p.cards[j]))]
            if fen<15:
                return True
        return False
    def printCardsList(self,li,up=-1):
        if up<0:
            up=len(li)
        for i in range(up):
            self.getCardVal(li[i]).print(i)
        print("")
    def printAllCards(self):
        for i in range(4):
            self.players[i].printCards()
            print("")
        print("底牌",end=": ")
        self.printCardsList(self.deck[-8:])
        print("")

    def __dfsPrintActList(self,newLi, li0,printFun=None):#printFun是打印这张牌的条件
        if isinstance(li0,np.ndarray):
            n=li0.shape[0]
        else:
            n=len(li0)

        for i in range(n):
            if isinstance(li0[i],int) or isinstance(li0[i],np.int32) or isinstance(li0[i],np.int64):
                if printFun==None or printFun(li0[i]):
                    newLi.append(cardToString(li0[i]))
            elif isinstance(li0[i],Action):
                if printFun==None or printFun(li0[i]):
                    newLi.append(li0[i].toString())
            else:
                t=[]
                self.__dfsPrintActList(t,li0[i])
                if printFun==None or printFun(t):
                    newLi.append(t)
    def dfsPrintActList(self, li,printFun=None):
        newLi=[]
        if isinstance(li,Action):
            newLi.append(li.toString())
        else:
            self.__dfsPrintActList(newLi,li,printFun)
        print(newLi)
    def shuffleDeck(self,T=5):
        up=len(self.deck)-1
        while(T>0):
            for i in range(up, 0, -1):
                rnd = random.randint(0, i)  # 每次随机出0-i-1之间的下标
                rnd1 = random.randint(0, up)  # 每次随机出0-i-1之间的下标
                rnd2 = random.randint(0, up)  # 每次随机出0-i-1之间的下标
                # print(rnd)
                self.deck[i], self.deck[rnd] = self.deck[rnd], self.deck[i]
                self.deck[rnd1], self.deck[rnd2] = self.deck[rnd2], self.deck[rnd1]
            T-=1
            # for i in range(len(self.deck)-1,0,-1):
            #     rnd=random.randint(0,i-1)# 每次随机出0-i-1之间的下标
            #     self.deck[i],self.deck[rnd] = self.deck[rnd],self.deck[i]



    def dealCards(self,beginDeckList=None,setDecor=-1,setNum=-1,setDealer=-1):#发牌

        if beginDeckList is None:
            self.shuffleDeck()  # 洗牌
        else:
            for i in range(len(self.deck)):
                self.deck[i]=beginDeckList[i]
        print(self.deck.tolist())
        self.deck_i = 0
        self.nowDealCardPlayer = 0
        for i in range(4):
            self.players[i].initCards()
        haveSetLord=False#有没有人叫主
        while(self.deck_i<CARDS_CNT-UNDERCARD_CNT):
            haveSetLord=(self.dealCard() or haveSetLord)
        #如果没人叫主，则从随机决定
        if setDecor!=-1 or setDealer!=-1 or setNum>0:
            print(setDecor, setDealer,setNum)
            
            print("指定主牌:",decorName[setDecor]+str(setNum),"   庄家：",setDealer)
            self.lordNum=setNum
            self.setLord(setDecor, setDealer)
            self.dealer = setDealer
        else:
            if not haveSetLord:
                print("随机决定主牌")
                self.setLord(random.randint(0, 3), random.randint(0, 3))  #dealer必须在==-1时，才会被改变
            if self.checkReDealCards():#满足一定条件可以重开
                print("重新发牌")
                self.dealCards()
                return
        print("庄家是玩家{:d}, 主牌:".format(self.dealer),end="")
        # print(self.lordNum,self.lordDecor)
        Card(self.getLord()).println()
        # self.printAllCards()
        for i in range(4):#整理手牌
            p=self.players[i]
            self.mergeLords(p)#合并再整理手牌
            # for j in range(5):  # 整理手牌
            #     print(str(j)+":"+str(p.cards_decorLen[i]))
            # self.players[i].printCards()
            # print("")
            # print("玩家"+str(i))
            # for i in range(5):
            #     env.printCardsList(p.cards_decorList[i], p.cards_decorLen[i])



    def dealCard(self):#发牌
        self.addCard(self.players[self.nowDealCardPlayer],self.deck[self.deck_i])
        self.deck_i+=1
        lord_tmp = self.snatchLord_v0(self.nowDealCardPlayer)  # 抢主
        have=False
        #self.printAllCards()
        if lord_tmp > -1 :
            self.setLord(lord_tmp,self.nowDealCardPlayer)
            have=True
        # self.players[self.nowDealCardPlayer].printCards()
        self.nowDealCardPlayer=(self.nowDealCardPlayer+1)%4
        return have
    def coldeck(self,fun):#换牌，
        p = self.players[self.dealer]
        self.underCards=self.deck[-8:].copy()
        for a in self.deck[-8:]:
            self.addCard(p, a)
        self.sortPlayerHand(p)
        self.underCards=fun(self,p)
        self.sortPlayerHand(p)
        # self.dfsPrintActList(self.underCards)
        for i in range(5):  # 修正长度
            while p.cards_decorLen[i] > 0 and p.cards_decorList[i][p.cards_decorLen[i] - 1] == 0:
                p.cards_decorLen[i] -= 1
        self.deck[-8:]=self.underCards
        self.mergeLords(p)
        # self.printAllCards()
        for i in range(HANDCARD_CNT,len(p.cards)):
            p.cards[i]=0
        self.setDealerID(self.dealer)

        for i in range(4):
            self.players[i].initCards_usedCards_cnt(self)
        # for i in range(5):
        #     env.printCardsList(p.cards_decorList[i], p.cards_decorLen[i])
        # print("")
    def setOrder(self):#设置单牌的大小
        self.orderInd = np.zeros((CARDS_CNT+1),dtype='int')
        self.decorInd = np.zeros((CARDS_CNT + 1),dtype='int')#花色分类，大小王和级牌都算主牌里
        h=CARDS_CNT//2
        self.orderInd[0]=-1
        self.orderInd[h]=99#大王
        self.orderInd[h-1]= 98  # 小王
        for j in range(4):
            for i in range(1,14):
                self.orderInd[j*13+i]=i
                self.decorInd[j * 13 + i] =j
            self.orderInd[j * 13+1] =14#A更大
        for i in range(1, 14):#主牌更大
            self.orderInd[self.lordDecor * 13 + i]=i+40
            self.decorInd[self.lordDecor * 13 + i] = 4
        self.orderInd[self.lordDecor * 13 + 1]= 14+40  # A更大
        for i in range(4):#级牌
            self.orderInd[self.lordNum+i*13]=60
            self.decorInd[self.lordNum+i*13] = 4
        self.orderInd[self.lordNum + self.lordDecor * 13] =  61

        #紧凑化
        ind=[0]*100
        for i in range(1,h+1):
            d=int(self.orderInd[i])
            ind[d]+=1
        k=1
        for i in range(1,100):
            if ind[i]>0:
                ind[i] = k
                k+=1
        for i in range(1,h+1):
            self.orderInd[h+i] = self.orderInd[i]=ind[int(self.orderInd[i])]
            self.decorInd[h+i] = self.decorInd[i]
        self.unlordMax=self.lordDecor==4 and 13 or 12#无主就是13

    def getCardVal(self,id):#id从1开始
        return Card(id)
    def getDecor(self,id):#id从1开始
        return (id - 1) // 13
    def cmpCard(self,a,b):#比较两组牌大小。返回1是a大，返回0是b大
        if a==0 or b==0 :
            return b==0
        # print(self.orderInd)
        num1=self.orderInd[a]
        num2=self.orderInd[b]
        d1, d2 = self.getDecor(a), self.getDecor(b)
        if num1<=self.unlordMax and num2<=self.unlordMax:#都不是主
            if d1==d2:#花色相同
                return num1>=num2
            else:
                return d1>=d2
        elif num1>self.unlordMax and num2>self.unlordMax:  # 都是主牌
            if num1==num2:#地位相同
                return d1>=d2
            else:
                return num1>num2
        return num1>self.unlordMax
    def _sortCardList_cmp(self,a,b):#比较两组牌大小。返回1是a大，返回-1是b大
        if self._ind[a]>1 and self._ind[b]>1 or self._ind[a]==1 and self._ind[b]==1:
            num1 = self.orderInd[a]
            num2 = self.orderInd[b]
            if num1<=self.unlordMax and num2<=self.unlordMax:#都不是主牌
                if self.getDecor(a) == self.getDecor(b):  # 花色相同
                    return num1 >= num2 and -1 or 1
                return self.getDecor(a) > self.getDecor(b) and -1 or 1
            return num1>=num2 and -1 or 1
        elif self._ind[a]>1:
            return -1
        return 1
    def _sortCardList_cmp1(self,a,b):#比较两组牌大小。返回1是a大，返回0是b大
        return (self.cmpCard(a,b) and -1 or 1)
    def _sortCardList_cmp2(self,a,b):#比较两组牌大小。把拖拉机排序
        lena = len(a)
        lenb = len(b)
        num1 = self.orderInd[a[0]]
        num2 = self.orderInd[b[0]]
        if lena==lenb:
            if num1<=self.unlordMax and num2<=self.unlordMax:#都不是主牌
                if self.getDecor(a[0]) == self.getDecor(b[0]):  # 花色相同
                    return num1 >= num2 and -1 or 1
                return self.getDecor(a[0]) > self.getDecor(b[0]) and -1 or 1
            return num1>=num2 and -1 or 1
        return lena>lenb and -1 or 1
    def sortCardList2(self,actList):#对卡牌排序，找出几个拖拉机，几个对子,几个单牌，会重叠,并且要求是相同颜色
        self._ind=np.zeros((CARDS_CNT2+1))
        n=len(actList)
        cnt2 = 0
        for i in range(n):
            self._ind[actList[i]]+=1
            if self._ind[actList[i]]==2:
                cnt2+=1
        actList = sorted(actList, key=cmp_to_key(self._sortCardList_cmp))
        traList=[]
        doubleList = []
        prePos=-1
        for i in range(cnt2-1):#找拖拉机
            next_i=(i+1)*2
            s = self.orderInd[actList[i * 2]] - self.orderInd[actList[next_i]]
            if (s == 0 or s== 1):
                if prePos==-1:
                    prePos=i*2
            elif prePos!=-1:
                traList.append([actList[t] for t in range(prePos,i*2+2,2)])
                prePos=-1
            doubleList.append(actList[i*2])
        if cnt2!=0:
            if prePos != -1 :
                traList.append([actList[t] for t in range(prePos,cnt2*2,2)])
            doubleList.append(actList[(cnt2-1) * 2])
        traList = sorted(traList, key=cmp_to_key(self._sortCardList_cmp2))
        actList = sorted(actList, key=cmp_to_key(self._sortCardList_cmp1))
        return actList,doubleList,traList
    def sortCardList1(self,actList,doubleList,traList):#对卡牌排序，找出几个拖拉机，几个对子,会把不同类别区分开(去重)//,要提前调用sortCardList2
        cnt2=len(doubleList)
        ind=np.zeros(55,dtype=int)
        for tra in traList:
            for a in tra:
                ind[a]+=1
        for a in doubleList:
            ind[a] += 1
        doubleList1 = []
        actList1=[]
        for a in actList:
            if ind[a]==0:
                actList1.append(a)
        for a in doubleList:
            if ind[a]==1:
                doubleList1.append(a)
        return actList1,doubleList1,traList
    def _isLord(self,actList):
        for a in actList:
            if self.orderInd[a]<=self.unlordMax:
                return False
        return True
    def isLord(self,act:Action):
        if self._isLord(act.one) == False:
            return False
        for dou in act.double:
            if self._isLord(dou)==False:
                return False
        return True
    def getActKind(self,act:Action):#获得一个动作类别，如果什么都有或者为空返回INF
        kind=INF
        for a in act.one:
            k=getKind(a,self.lordDecor,self.lordNum)
            if kind!=INF and kind!=k:
                return INF
            kind=k
        for dou in act.double:
            for a in dou:
                k = getKind(a, self.lordDecor, self.lordNum)
                if kind != INF and kind != k:
                    return INF
                kind = k
        return kind
    def _useCmpCards(self,act1:Action,act2:Action):#必须二者是同一种类别!!!
        for i in range(len(act1.double)):#先判断2是否为对子或拖拉机
            for j in range(0,len(act1.double[i]),2):
                if act2.double[i][j]!=act2.double[i][j+1]:
                    return 1
        if len(act1.double)>0:#有对子或者拖拉机，就看对应牌的大小
            return self.orderInd[act1.double[0][0]]>=self.orderInd[act2.double[0][0]]
        for i in range(len(act1.one)):  # 最后判断单牌
            if self.orderInd[act1.one[i]] != self.orderInd[act2.one[i]]:  # 单牌且不等
                return self.cmpCard(act1.one[i], act2.one[i])
        return 1  # 完全想等，先出的大


    def useCmpCards(self,act1:Action,act2:Action):#比较两组牌大小，保证两组牌一样多，且actList1为先出，actList2为后出。返回1是actList1大，否则actList2大
        n=act1.len
        k1 = self.getActKind(act1)#判断什么类型
        k2 = self.getActKind(act2)
        if k1==self.lordDecor and k2==self.lordDecor:#如果都是主牌,先判断拖拉机，在判断单对子
            return self._useCmpCards(act1,act2)
        elif k1!=self.lordDecor and k2!=self.lordDecor:#如果都是非主牌,判断花色，花色相同比大小，否则先出的大。
            if k2!=k1:#act2是杂牌
               return 1
            return self._useCmpCards(act1,act2)
        elif k1==self.lordDecor:#先出的是主牌，后出的不是，先出的大
            return 1
        else:#先出的不是主牌，后出的是主牌,要看主牌能否完全管上先出的。这里先出的一定是同一花色。主牌要杀拖拉机，只需要相应数量对子。
            for i in range(len(act1.double)):  # 先判断2是否为对子或拖拉机
                for j in range(0, len(act1.double[i]), 2):
                    if act2.double[i][j] != act2.double[i][j + 1]:
                        return 1
            return 0

    # def judgeRoundWin(self):
    def getOrderID(self,a):#从0开始，非主牌是[0,11],主牌[0,15]
        k = self.orderInd[int(a)]
        if k > self.unlordMax:
            k -= self.unlordMax
        return int(k-1)

    def getMaxCards_cmp1(self,a):
        return a==2
    def getMaxCards_cmp2(self,a):
        return a>0
    def getMaxCards(self, sortCardsList1,p:Player):  # 返回已知的玩家手里最大的牌,来判断先手玩家是否可以甩牌,sortCardsList是一名玩家分好类的手牌，且不会相互包含
        li=[]

        for kind in range(5):
            if kind==self.lordDecor:#主牌不能甩
                continue
            k1 = p.getSelfMaxCard(kind, self.getMaxCards_cmp1)#从大到小找到第一个没有出过2张牌的牌的顺序号
            k2 = p.getSelfMaxCard(kind, self.getMaxCards_cmp2)#从大到小找到第一个出过0张牌的牌的顺序号
            act=Action()
            for a in sortCardsList1[kind][0]:
                k=self.getOrderID(a)
                if k>=k1:#看比它大的有多少已经出过2张，如果都出过2张，说明它是最大的牌
                    act.addOne(a)
            for a in sortCardsList1[kind][1]:#看是不是最大的对子
                k = self.getOrderID(a)
                if k>=k2:#看有多少出了0张，如果都没有出了0张
                    act.addDou([a,a])
            maybeTra=[]
            seqCnt=0

            for i in range(p.usedCards_len[kind] - 1, -1,-1):#寻找所有可能的拖拉机
                if i == p.numCardKind:
                    c=0
                    for j in range(4):
                        if j != p.lordDecor:
                            c+=p.lordNum_cnt[j]==0
                    seqCnt+=c
                    if c<3 and seqCnt>0:
                        if seqCnt>1:
                            maybeTra.append((i,seqCnt))
                        # maxSeqCnt = max(maxSeqCnt, seqCnt)
                        seqCnt = 0
                elif p.usedCards_cnt[kind][i]==0:
                    seqCnt+=1
                elif seqCnt>0:
                    if seqCnt > 1:
                        maybeTra.append((i+seqCnt,seqCnt))
                    seqCnt = 0
            if seqCnt>1:
                maybeTra.append((-1+seqCnt,seqCnt))
            # if p.id == 0:
            #     print(maybeTra)
            for traList in sortCardsList1[kind][2]:#看自己的拖拉机
                n=len(traList)
                k = self.getOrderID(traList[0])
                for tra in maybeTra:
                    # if p.id == 0:
                    #     print(tra, n)
                    if tra[1]>=n and tra[0]>k:#可能的拖拉机的长度大于等于自己的拖拉机，且可能的拖拉机比我方的次序大。
                        break
                else:#正常结束循环，说明我方的拖拉机最大
                    act.addDou([traList[i//2] for i in range(len(traList)*2)])
            if act.isSeq():
                li.append(act)
        return li#返回一个5元列表，代表一定可以甩的牌
    def judgeSeqUse(self,act:Action,meid,sortCardList4):#判断一组牌是否可以甩,pid为一个玩家的手牌,sortCardList4是4个人的分好类的手牌，且会相互包含
        # 是否为甩牌,是否可以出
        if act.isSeq():#是甩牌
            mina=INF
            card=0
            for a in act.one:#找出最小的牌
                if mina>self.orderInd[a]:
                    mina=self.orderInd[a]
                    card=a
            decor, num = getDecorAndNum(card)
            for pid in range(4):
                if pid != meid:
                    li=sortCardList4[pid]
                    if len(li[decor][0])>0 and self.orderInd[li[decor][0][0]]>self.orderInd[card]:#寻找这个花色所有比card顺序大的牌：
                        return True,False

            for pid in range(4):
                if pid != meid:#遍历每个人手牌的对子。
                    li = sortCardList4[pid]
                    for dou in act.double:
                        decor, num = getDecorAndNum(dou[0])
                        lendou=len(dou)#它是li里面拖拉机牌数的2倍，比如3344，则li里面表示为34
                        if lendou==2:
                            a=li[decor][1]
                            if len(a) > 0 and self.orderInd[a[0]] > self.orderInd[dou[0]]:  # 寻找这个花色所有比card顺序大的牌：
                                return True, False
                        else:
                            tractors = li[decor][2]
                            for tra in tractors:
                                if len(tra)*2>=lendou and self.orderInd[tra[0]] > self.orderInd[dou[0]]:
                                    return True, False
            return True, True
        return False,True
    def getAllFirstAct(self,sortCardsList1,p:Player):#返回作为先出玩家的所有可能的动作(不包含甩牌),sortCardsList1是一名玩家分好类的手牌，且不会相互包含
        ans=[]
        #actList[cnt2 * 2:], doubleList, traList
        maxCards=self.getMaxCards(sortCardsList1,p)
        for i in range(5):
            for a in sortCardsList1[i][0]:
                ans.append(Action([a]))
            for a in sortCardsList1[i][1]:
                ans.append(Action([a]))
                ans.append(Action([],[[a,a]]))
            for tractor in sortCardsList1[i][2]:#拖拉机
                ltr=len(tractor)
                for a in tractor:
                    ans.append(Action([a]))
                    ans.append(Action([], [[a, a]]))
                ans.append(Action([],[[tractor[t//2] for t in range(ltr*2)]]))
                if ltr>2:
                    ans.append(Action([],[[tractor[t//2] for t in range(2,ltr*2)]]))
        ans += maxCards
        return ans
    def getAllAct_dfs(self,doubleList,doubleList_i,ansDown,nowList,nowList_i,n):
        if nowList_i==n:
            ansDown.append(nowList.copy())
            return
        doubleList_len=len(doubleList)
        for i in range(doubleList_i,doubleList_len):
            if doubleList_len-i<n-nowList_i:
                return
            else:
                nowList.append(doubleList[i])
                nowList.append(doubleList[i])
                self.getAllAct_dfs(doubleList,i+1,ansDown,nowList,nowList_i+1,n)
                nowList.pop()
                nowList.pop()


    def getAllAct(self, sortCardsList, p: Player, cardsList_max):  # 得到所有非第一家的动作，sortCardsList是p玩家分好类的手牌，且会相互包含
        # 返回所有大过之前最大的玩家的牌和小于之前玩家的牌
        # cardsList_max只能是单张，对子，连对
        #小于玩家的牌有太多组合会被忽略
        n = len(cardsList_max)
        kind=getKind(cardsList_max[0],self.lordDecor,self.lordNum)
        n2=n//2
        ansUp = []
        ansDown = []
        isHave=False
        c0,c1,c2=len(sortCardsList[kind][0]),len(sortCardsList[kind][1]),len(sortCardsList[kind][2])
        if n == 1:  # 单张牌
            isHave = c0 > 0
            ansDown=self.getFollowAct_one(p,cardsList_max[0],kind)
            if kind == self.lordDecor or c0 > 0:  # 是主牌或副牌，但有这类副牌
                for i in range(c0):
                    a = sortCardsList[kind][0][i]
                    if self.orderInd[a] > self.orderInd[cardsList_max[0]]:
                        ansUp.append([a])
            else:  # 用主牌杀
                for i in range(p.cards_decorLen[self.lordDecor]):
                    a = p.cards_decorList[self.lordDecor][i]
                    ansUp.append([a])
        elif n == 2:
            isHave = c1>0
            if kind == self.lordDecor or c1>0:  # 这类牌有对子
                for a in sortCardsList[kind][1]:
                    if self.orderInd[a] > self.orderInd[cardsList_max[0]]:
                        ansUp.append([a,a])
            elif p.cards_decorLen[kind] == 0:  # 用主牌杀
                for a in sortCardsList[self.lordDecor][1]:
                    ansUp.append([a,a])
            if c1>0:
                for a in sortCardsList[kind][1]:
                    if self.orderInd[a] <= self.orderInd[cardsList_max[0]]:
                        ansDown.append([a, a])
            else:
                ansDown.append([INF ,INF])  #比如先手出了66，而自己有7，则此时依然返回INFINF
        else:#拖拉机
            #较大的拖拉机和较小的拖拉机
            if kind == self.lordDecor or c2>0:  # 这类牌有拖拉机
                for tractor in sortCardsList[kind][2]:
                    for j in range(len(tractor)-n2+1):#比如最大的拖拉机是3344，而自己手里又55667788，则此时有3种出牌方式管上它
                        isHave = True
                        if self.orderInd[tractor[j]]>self.orderInd[cardsList_max[0]]:
                            ansUp.append([tractor[j+k//2] for k in range(n)])
                        else:
                            ansDown.append([tractor[j + k // 2] for k in range(n)])
            elif p.cards_decorLen[kind] == 0:  # 用主牌杀
                for tractor in sortCardsList[self.lordDecor][2]:
                    for j in range(len(tractor)-n2+1):
                        ansUp.append([tractor[j+k // 2] for k in range(n)])

            if not isHave:#如果没有同样长度的拖拉机,则从对子里出
                tempLi = []
                if len(sortCardsList[kind][1]) > n2:
                    self.getAllAct_dfs(sortCardsList[kind][1], 0, ansDown, tempLi, 0, n2)  # 从对子里选取所有组合
                else:
                    for a in sortCardsList[kind][1]:
                        tempLi.append(a)
                        tempLi.append(a)
                    tempLi_len = len(tempLi)
                    for i in range(tempLi_len, n):
                        tempLi.append(INF)
                        # 剩余牌优先从这个颜色里选取
                        #没有这个颜色再从其他任意牌选
                        #比如先手是'<♥A>', '<♥A>', '<♥K>', '<♥K>', '<♥Q>', '<♥Q>'，我有JJQQ9,此时会返回JJQQINFINF
                    ansDown.append(tempLi)

        return ansUp,ansDown,isHave

    def getFollowAct_one(self, p: Player, card_max, kind):  # 返回小于某张牌的全部牌
        ans=[]
        if p.cards_decorLen[kind]>0:#大于0
            for i in range(p.cards_decorLen[kind]):
                a=p.cards_decorList[kind][i]
                if self.orderInd[a]<=self.orderInd[card_max]:
                    ans.append([a])
        else:#
            for i in range(5):
                if i!=self.lordDecor:
                    for j in range(p.cards_decorLen[i]):
                        ans.append([p.cards_decorList[i][j]])
        return ans

    def snatchLord_v0(self,id):#发牌时候抢主的常规策略
        if self.lordDecor>=0:
            return -1

        for i in range(4):
            p=self.players[id]
            c=p.cards_decorLen[i]+p.cards_decorLen[4]

            if  c>=0 :#没人亮过，超过均值就亮牌,且牌数大于等于4,主牌一共36个，平局每人9个
                for j in range(p.cards_decorLen[4]):
                    decor,num=getDecorAndNum(p.cards_decorList[4][j])
                    if num ==self.lordNum and decor==i:  # 有级牌
                        return i
        return -1
    def game_step(self,actList4,first_playerId):#输入4个人的出牌组合,保证每个人出的牌数量一样多，返回赢得那个人和所得分数
        playerId=first_playerId
        for i in range(1,4):
            k=(first_playerId+i)%4
            if self.useCmpCards(actList4[playerId],actList4[k])==0:#actList4[k]大
                playerId=k
        #找找到最大的那个人
        sc=0
        n=actList4[first_playerId].len#所有人的牌数量都是n
        for i in range(4):
            sc+=actList4[i].getFen()
        if self.players[playerId].dealerTag>1:
            self.sumSc+=sc
        if self.playerTable_i[0]>=HANDCARD_CNT:#游戏结束,结算底牌分数
            sc_under = rate=0
            if self.players[playerId].dealerTag>1:#不是庄家赢
                for a in self.underCards:#结算底牌分数
                    num = getNum(a) # 点数，[1,13]王是14
                    sc_under += fenInd[num]  # 分数
                rate=2
                if n>1:#判断倍数
                    doubleCnt=actList4[playerId].getDouleCnt()*2
                    if doubleCnt>0:#每有一个对子，倍数+2
                        rate=2+doubleCnt
                    else:#普通甩牌没有对子，倍数+1
                        rate=2+1#普通甩牌为3倍
                self.sumSc += sc_under*rate
            return playerId, sc,True,{"fen":sc_under*rate}
        return playerId,sc,False,{}#返回赢得玩家id，本轮得分，是否结束游戏，结算信息
    def getGrade(self,sc):
        if sc<80:
            return (sc==0 and 0 or (sc//40+1))-3#判断庄家升几级
        else:
            return sc // 40-2#如果不足120仅仅换庄，不升级
    def setLord(self,kind,playerID):
        self.lordDecor=kind
        if self.dealer==-1:
            self.dealer=playerID
        for i in range(4):
            self.players[i].setLord(kind,self.lordNum)
        self.setOrder()
    def getLord(self):
        return self.lordNum+self.lordDecor*13
    def sortPlayerHand(self,player):
        player.cards = sorted(player.cards, key=cmp_to_key(self._sortCardList_cmp1))
        for i in range(5):
            player.cards_decorList[i]= sorted(player.cards_decorList[i], key=cmp_to_key(self._sortCardList_cmp1))

    def mergeLords(self,player):#把级牌放入主牌所在花色
        if self.lordDecor!=4:
            len1=player.cards_decorLen[self.lordDecor]
            len2=player.cards_decorLen[4]
            player.cards_decorList[self.lordDecor][len1:len1+len2]=player.cards_decorList[4][0:len2]
            player.cards_decorLen[self.lordDecor]+=len2
            player.cards_decorLen[4]=0
            for i in range(len2):
                player.cards_decorList[4][i]=0
        self.sortPlayerHand(player)
    def searchFen(self,cardList):
        ans=0
        for a in cardList:
            num=getNum(a)
            ans+=fenInd[num]
        return ans

    def printAllInfo(self, act):  # act代表4个人每个人出的牌,类型是Action
        for i in range(4):
            self.players[i].printCards()
        for i in range(4):
            act[i].println(i)
        print("当前闲家的分" + str(self.sumSc))
    def printUnderCards(self):  # act代表4个人每个人出的牌
        self.printCardsList(self.deck[-8:])

def randomUpdateINF(p:Player,act:list, kind):#随机选取动作，kind是第一个出牌的玩家的花色。返回种类和在cards_decorList中位置的编号
    actList=[]
    for j in range(p.cards_decorLen[kind]):#先看本花色有木有
        if p.cards_decorList[kind][j]!=0:
            actList.append((kind,j))
    if len(actList)==0:#本花色没有，去其它花色找
        for i in range(5):
            if i==kind or p.cards_decorLen[i]==0:
                continue
            for j in range(p.cards_decorLen[i]):
                if p.cards_decorList[i][j] != 0:
                    actList.append((i,j))
    ans=actList[random.randint(0,len(actList)-1)]
    return ans[0],ans[1]
def getAllAct(self, sortCardsList, p: Player, cardsList_max, kind):  # sortCardsList是p玩家分好类的手牌
    # 返回所有大过之前最大的玩家的牌和小于之前玩家的牌
    # cardsList_max只能是单张，对子，连对
    # 小于玩家的牌有太多组合会被忽略
    n = len(cardsList_max)
    ansUp = []
    ansDown = []
    if n == 1:  # 单张牌
        ansDown = self.getFollowAct_one(p, cardsList_max[0], kind)
        if kind == self.lordDecor or p.cards_decorLen[kind] > 0:  # 是主牌或副牌，但有这类副牌
            for i in range(p.cards_decorLen[kind]):
                a = p.cards_decorList[kind][i]
                if self.orderInd[a] > self.orderInd[cardsList_max[0]]:
                    ansUp.append([a])
# env=CC()
# env.dealCards(None,0,5,0)#发牌测试
# env.dfsPrintActList([9,27])
# act1=Action([1,9,10],[[2,2]])
# act2=Action([1,9,10],[[54,54]])
# print(env.useCmpCards(act1,act2))
# <i:0 ♠9>
# <i:0 ♦A>
# <i:0 ♦10>
# <i:0 ♦9>
# actList,doubleList,traList=env.sortCardList2([11,11,12,12,13,13,14,1,2,3,4,5,5,6,15,6,8,8,9,9])
# print(actList,doubleList,traList)
# ansDown=[]
# nowList=[]
# begin_time = time.time()
# env.getAllAct_dfs(doubleList,0,ansDown,nowList,0,4)
# passed_time = time.time() - begin_time
# print(ansDown)
# print(passed_time)
# li=[39, 39, 23, 12, 26, 5, 53, 1, 38, 30, 46, 54, 48, 40, 36, 6, 28, 46, 26, 18, 7, 16, 27, 5, 22, 20, 47, 41, 41, 34, 8, 3, 31, 30, 13, 16, 23, 15, 48, 13, 51, 4, 37, 44, 33, 25, 52, 34, 9, 37, 21, 3, 17, 50, 29, 24, 51, 49, 38, 35, 43, 24, 6, 18, 32, 22, 29, 7, 20, 11, 19, 15, 36, 14, 42, 27, 45, 14, 12, 50, 45, 52, 31, 11, 42, 40, 47, 33, 54, 32, 8, 21, 10, 49, 9, 25, 53, 44, 1, 4, 17, 19, 10, 2, 35, 43,28,2]
# env.dealCards()#发牌测试
# env=CC()
# env.dealCards()
# # actList,doubleList,traListt=env.sortCardList2(li)
# # print(env.dfsPrintActList(actList))
# li=np.zeros(33,dtype=np.int32)
# li[0]=2
# li = sorted(li, key=cmp_to_key(env._sortCardList_cmp1))
# env.dfsPrintActList(li)
# print(env.orderInd)
# actList, doubleList, traListt =env.sortCardList2([1,3,53,53,5,5,31,31,3,4,4,6,6,8,8,11,12,1])
# env.dfsPrintActList(actList)
# env.dfsPrintActList(doubleList)
# env.dfsPrintActList(traListt)




