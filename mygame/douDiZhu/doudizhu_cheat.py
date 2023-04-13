import math
import random
from functools import cmp_to_key
import numpy as np


#作弊器，给玩家3个玩家设置好手牌，来让智能体探索到更好的动作
from doudizhu_game import CARDS_CNT, NameTodecorId, UNDERCARD_CNT, stringToCardId, Player

numset={"A":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"J":11,"Q":12,"K":13}

def numToCardId_decorList(a,decorList):
    cardid=0
    if a.find("♠") > -1 or a.find("♥") > -1 or a.find("♦") > -1 or a.find("♣") > -1 or a.find("大王") > -1 or a.find("小王") > -1:
        cardid = stringToCardId[a]
    elif a in numset:
        li = decorList[numset[a] - 1]
        m = len(li)
        if m > 0:
            k = random.randint(0, m - 1)
            cardid = li[k] * 13 + numset[a]
            li.pop(k)
    return cardid
def mkDeck(fun):
    deck=[0]*CARDS_CNT
    cardDir,setDealer=fun()
    plen=[0,0,0]
    decorList=[[0,1,2,3] for i in range(13)]
    for pid in range(3):
        for a in cardDir[pid]:
            cardid=numToCardId_decorList(a,decorList)
            if cardid>0:
                deck[plen[pid]*3+pid]=cardid
                plen[pid]+=1
            if plen[pid]*3+pid>=CARDS_CNT-UNDERCARD_CNT:
                break
    # print(deck)
    ind=np.zeros((CARDS_CNT+1))
    for a in deck:
        ind[a]+=1
    li=[]
    for i in range(1,CARDS_CNT+1):
        if ind[i]==0:
            li.append(i)
    for i in range(CARDS_CNT):
        if deck[i]==0:
            deck[i]=li.pop()
    deck=np.array(deck)
    # print(deck)
    return deck,setDealer
def setHandCards(p:Player,li):#["Q","K","10","J","9","小王","大王","A", "A", "A", "A"]
    decorList=[[0,1,2,3] for i in range(13)]
    for i in range(len(li)):
        li[i]=numToCardId_decorList(li[i],decorList)
    p.cards=li.copy()
def cheat1():#自定义作弊器
    setDealer=0
    dir=[[],[],[],[]]
    dir[0]=["Q","K","10","J","9","小王","大王","A", "A", "A", "A"]
    dir[1] = ["7","7","7", "8", "8", "8"]

    return dir,setDealer
# mkDeck1(cheat1)