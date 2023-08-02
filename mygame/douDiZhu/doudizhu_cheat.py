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
def setAllPlayerHandCards(players,li):#自定义玩家手牌，如：env.players,[["A", "A"],["大王", "3"],["4", "4"]]
    decorList=[[0,1,2,3] for i in range(13)]
    deck=[i+1 for i in range(54)]
    cnt=[0,0,0]
    for j in range(3):
        for i in range(len(li[j])):
            li[j][i]=numToCardId_decorList(li[j][i],decorList)
            deck.remove(li[j][i])
        players[j].cards=li[j].copy()
        cnt[j] = 51 // 3 + (players[j].dealerTag == 0) * 3-len(li[j])
    deck_i=0
    for j in range(3):
        uncards=deck[deck_i:deck_i+cnt[j]]
        deck_i+=cnt[j]
        players[j].uncards=uncards.copy()
        players[j].beginCards=uncards.copy()+players[j].cards.copy()
def cheat1():#自定义作弊器
    setDealer=0
    dir=[[],[],[],[]]
    dir[0]=["Q","K","10","J","9","7","A", "A", "A", "A","大王","2"]#0是地主
    dir[2] = ["4","4","4","5","5","5","6","6","6" ,"7","8", "8", "8","8","Q","Q","小王"]
    dir[1] = ["3", "3", "3", "3", "4", "10","J","7","7","K","10","K","K","J","2","2","2"]

    return dir,setDealer
