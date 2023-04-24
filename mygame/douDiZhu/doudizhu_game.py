import math
import random
from functools import cmp_to_key

import numpy as np


CARDS_CNT=54
UNDERCARD_CNT=3
INF=1000
HANDCARD_CNT = (CARDS_CNT - UNDERCARD_CNT) // 3
decorName=["♠","♥","♣","♦","王"]
NameTodecorId={"♠":0,"♥":1,"♣":2,"♦":3,"王":4}
numName=["","A","2","3","4","5","6","7","8","9","10","J","Q","K"]
cardOrder=np.zeros((CARDS_CNT+1),dtype='int')#顺序
orderToNum=np.zeros((20),dtype='int')#顺序
actKindList=["pass", "bomb", "seq", "douseq", "triseq", "triseq_1", "triseq_2", "solo", "dou", "tri", "tri_1", "tri_2", "qua_2"]#"qua_4"
actKindToId={}

def getDecor(id):  # id从1开始，判断花色
    return int((id - 1) // 13)
def cardIdListlistToNum(cards):
    for i in range(len(cards)):
        cards[i]=getNum(cards[i])
def getNum(id):  # id从1开始，判断点数,14是王
    if id<53:
        return int((id - 1) % 13+1)
    return 13+id%52
def getDecorAndNum(id):
    if id<1:
        return 0,0
    id = (id - 1) % CARDS_CNT + 1
    decor = (id - 1) // 13  # 黑桃是0，红糖是1，梅花是2，方片是3，王是4
    num = (id - 1) % 13 + 1  # 点数
    return int(decor),int(num)
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
def init():
    for i in range(len(actKindList)):
        actKindToId[actKindList[i]]=i

    stringToCardId["INF"]=INF
    stringToCardId["大王"] = 54
    stringToCardId["小王"] = 53
    for i in range(1,CARDS_CNT+1):
        stringToCardId[cardToString(i)]=i
        stringToCardId[cardToString(i).lower()] = i
        stringToCardId[cardToString(i)[1:-1]] = i
        stringToCardId[str(i)] = i
    order=[12,13,1,2,3,4,5,6,7,8,9,10,11]
    for i in range(1,CARDS_CNT-1):
        cardOrder[i]=order[(i-1)%13]
        orderToNum[cardOrder[i]]=getNum(i)
    cardOrder[53]=14
    orderToNum[cardOrder[53]] = 14
    cardOrder[54] = 15
    orderToNum[cardOrder[54]] = 15
init()
MAXNUM=16
MAXORDER=16
# print(cardOrder)
def _sortCardList_cmp1(a,b):#按照顺序排序,大王小王排到最后
    return cardOrder[a]-cardOrder[b]

class Card():
    def __init__(self,id,holder=-1):#id从1开始，前
        self.id=id
        if id<109:
            self.decor,self.num=getDecorAndNum(id)

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
class Action():
    def __init__(self, one=[],appendix=[],playerId=-1):  # double里包含拖拉机，如[[3,3],[4,4,5,5]]
        self.cards=one.copy()
        self.appendix=appendix.copy()
        self.len=len(one)+len(appendix)
        self.sortCards()
        self.updateKind()
        self.playerId=playerId
    def updateKind(self):
        if self.len==0 or self.cards[0]==0:
            self.kind="pass"
        elif self.isBomb():
            self.kind = "bomb"
        elif self.isSeq():
            self.kind = "seq"
        elif self.isDouseq():
            self.kind = "douseq"
        elif self.isTriseq():
            l=len(self.appendix)
            if l==0:
                self.kind = "triseq"
            elif l==len(self.cards)//3:
                self.kind = "triseq_1"
            elif l==len(self.cards)//3*2:
                self.kind = "triseq_2"
        elif self.len==1:
            self.kind="solo"
        elif self.len==2:
            self.kind="dou"
        elif self.len==3:
            self.kind="tri"
        elif self.len==4:#3带1
            self.kind="tri_1"
        elif self.len==5:#3带2或者顺子
            if len(self.cards)==3:
                self.kind="tri_2"
        elif self.len==6 and len(self.cards)==4:
            self.kind = "qua_2"
        elif self.len==8 and len(self.cards)==4:
            self.kind = "qua_4"
        else:
            self.kind="erro"
        if self.kind=="erro":
            print("erro: "+self.toString())
            exit()
    def toString(self):
        if self.isPass():
            return self.kind
        ans=""
        for a in self.cards:
            ans+=cardToString(a)
        for a in self.appendix:
            ans+=cardToString(a)
        return ans
    def tolist(self):
        if self.isPass():
            return []
        ans=self.cards.copy()
        if len(self.appendix)>0 and 0<self.appendix[0]<55:
            ans+=self.appendix.copy()
        return ans
    def print(self,i=0):
        print("act"+str(i)+":",end="")
        i=0
        for a in self.cards:
            Card(a).print(i)
            i += 1
        for a in self.appendix:
            Card(a).print(i)
            i+=1
    def add(self,li:list):
        for a in range(li):
            self.cards.append(a)
        self.len += len(li)
    def addApp(self,li:list):
        for a in range(li):
            self.appendix.append(a)
        self.len += len(li)
    def isBomb(self):
        if self.len==4 and len(self.cards)==4:
            num=getNum(self.cards[0])
            for i in range(1,4):
                if num!=getNum(self.cards[i]):
                    break
            else:
                return 1
        if self.len==2 and getNum(self.cards[0])>13 and getNum(self.cards[1])>13:#是对王
            return 1
        return 0

    def isSeq(self):#是不是顺子
        if self.len>=5:
            d1=cardOrder[self.cards[0]]
            for i in range(0,len(self.cards)):
                if cardOrder[self.cards[i]]-d1!=i:
                    return 0
            return 1
        return 0
    def isDouseq(self):#是不是连对
        if self.len>=6 and self.len%2==0:
            d1=cardOrder[self.cards[0]]
            for i in range(0,len(self.cards),2):
                if cardOrder[self.cards[i]]-d1!=i//2 and cardOrder[self.cards[i]]==cardOrder[self.cards[i+1]]:
                    return 0
            return 1
        return 0
    def isTriseq(self):#是不是三连对
        n=len(self.cards)
        if n>=6 and n%3==0:
            d1=cardOrder[self.cards[0]]
            for i in range(0,n,3):
                if cardOrder[self.cards[i]]-d1!=i//3 and cardOrder[self.cards[i]]==cardOrder[self.cards[i+1]] and  cardOrder[self.cards[i]]==cardOrder[self.cards[i+2]]:
                    return 0
            return 1
        return 0
    def isPass(self):
        return self.len==0 or self.cards[0]==0
    def sortCards(self):
        self.cards.sort(key=cmp_to_key(_sortCardList_cmp1))
    def clone(self):
        return Action(self.cards,self.appendix,self.playerId)
class Player():
    def __init__(self,env,id):
        self.env=env
        self.id=id
        self.dealerTag=0 #0是庄家，1和2都是闲家
        self.initCards()
        self.last_move=Action()
    def initCards(self):
        self.sc=0
        self.cards =[]#手牌
        self.uncards=[]#已经用了的牌
        self.beginCards=[]
        # np.zeros(HANDCARD_CNT + 8, dtype='int')

    def addCard(self,a):
        self.cards.append(a)
    def getCardCnt(self):
        return len(self.cards)
    def sortCards(self):
        self.cards.sort(key=cmp_to_key(_sortCardList_cmp1))
    def getHandList(self):
        self.adList = [[] for i in range(MAXNUM)]
        for a in self.cards:
            self.adList[getNum(a)].append(a)
        return self.adList
    def getAllDouAction(self,already=[]):#得到对子
        ans=[]
        adList=self.adList
        for i in range(1, 14):#一张，二张
            if len(adList[i]) > 1 and not (i in already):
                ans.append(Action([adList[i][0],adList[i][1]],playerId=self.id))
        return ans
    def getAllOneAction(self,tar=0):#得到单牌
        ans=[]
        adList=self.adList
        for i in range(1, MAXNUM):#一张，二张
            if len(adList[i]) > 0 and getNum(adList[i][0])!=tar:
                ans.append(Action([adList[i][0]],playerId=self.id))
        return ans
    def pushDouKing(self,ans:list):
        if len(self.adList[14]) > 0 and len(self.adList[15]) > 0:
            ans.append(Action([self.adList[14][0], self.adList[15][0]], playerId=self.id))
            return True
        return False
    def getAllAction_plane_dfs_2(self,ans,tra,appendix,i):
        for j in range(i,MAXNUM):
            if len(self.adList[j])<2:
                continue
            appendix.append(self.adList[j].pop())
            appendix.append(self.adList[j].pop())
            if len(appendix)==len(tra)//3*2:
                ans.append(Action(tra,appendix,playerId=self.id))
            self.getAllAction_plane_dfs_2(ans,tra,appendix,j)
            self.adList[j].append(appendix.pop())
            self.adList[j].append(appendix.pop())
    def getAllAction_plane_dfs_1(self,ans,tra,appendix,i):
        for j in range(i,MAXNUM):
            if len(self.adList[j])==0:
                continue
            appendix.append(self.adList[j].pop())
            if len(appendix)==len(tra)//3:
                ans.append(Action(tra,appendix,playerId=self.id))
            self.getAllAction_plane_dfs_1(ans,tra,appendix,j)
            self.adList[j].append(appendix.pop())

    def getAllFirstAction(self,all=False):#注意，斗地主没有同花顺.这里三带一和飞机，四带2被省略去了。
        douCnt=0
        adList=self.getHandList()
        ans = []  # 0代表不管
        for i in range(1, MAXNUM):#一张，二张,三张，四张
            leni=len(adList[i])
            if leni > 0:
                ans.append(Action([adList[i][0]],playerId=self.id))
            if leni > 1:
                douCnt+=leni//2
                ans.append(Action([adList[i][0],adList[i][1]],playerId=self.id))
            if leni > 2:
                ans.append(Action([adList[i][0],adList[i][1],adList[i][2]],playerId=self.id))
            if leni > 3:
                ans.append(Action(adList[i],playerId=self.id))
        self.pushDouKing(ans)#
        if all:#不允许省略过大动作
            for i in range(1, 14):  # 三带一和4带2
                leni = len(adList[i])
                if leni >=3:
                    for j in range(1, MAXNUM):
                        if len(adList[j])>0 and j!=i:
                            ans.append(Action([adList[i][0], adList[i][1], adList[i][2]], [adList[j][0]], playerId=self.id))
                        if len(adList[j])>1 and j!=i:
                            ans.append(Action([adList[i][0], adList[i][1], adList[i][2]], [adList[j][0],adList[j][1]], playerId=self.id))
                if leni ==4:
                    for j in range(1, MAXNUM):
                        if len(adList[j])==0:
                            continue
                        a=adList[j].pop()
                        for k in range(j, MAXNUM):
                            if  len(adList[k])>0 and k!=i and j!=i:
                                ans.append(Action(adList[i], [a,adList[k][0]], playerId=self.id))
                        adList[j].append(a)
        else:
            for i in range(1, 14):#三带一和4带2
                leni = len(adList[i])
                if leni == 3:
                    if len(self.cards)>3 :
                        ans.append(Action([adList[i][0],adList[i][1],adList[i][2]],[INF],playerId=self.id))
                    if douCnt>=2:
                        ans.append(Action([adList[i][0],adList[i][1],adList[i][2]],[INF,INF],playerId=self.id))
                elif leni==4:
                    if len(self.cards)>4 :
                        ans.append(Action([adList[i][0],adList[i][1],adList[i][2]],[INF],playerId=self.id))
                    if douCnt>=3:
                        ans.append(Action([adList[i][0],adList[i][1],adList[i][2]],[INF,INF],playerId=self.id))
                    if len(self.cards)>5:
                        ans.append(Action(adList[i], [INF, INF],playerId=self.id))
                    # if len(self.cards)>7 and douCnt>2:#4带4
                    #     ans.append(Action(adList[i], [INF, INF,INF, INF], playerId=self.id))

        cnt=0
        for i in range(1,13):#顺子
            if len(adList[orderToNum[i]])>0:
                cnt+=1
                for k in range(5,cnt+1):
                    ans.append(Action([adList[orderToNum[j]][0] for j in range(i-k+1,i+1)],playerId=self.id))
            else:
                cnt=0
        cnt = 0
        for i in range(1,13):#连对
            if len(adList[orderToNum[i]])>1:
                cnt+=1
                for k in range(3,cnt+1):
                    tra=[]
                    for j in range(i - k + 1, i + 1):
                        tra.append(adList[orderToNum[j]][0])
                        tra.append(adList[orderToNum[j]][1])
                    ans.append(Action(tra,playerId=self.id))
            else:
                cnt=0
        cnt=0
        if all:  # 不允许省略过大动作
            for i in range(1,13):#飞机,不带牌,以及带未知牌
                if len(adList[orderToNum[i]])>2:
                    cnt+=1
                    for k in range(2,cnt+1):
                        tra=[]
                        for j in range(i - k + 1, i + 1):
                            tra.append(adList[orderToNum[j]][-3])
                            tra.append(adList[orderToNum[j]][-2])
                            tra.append(adList[orderToNum[j]][-1])
                            self.adList[orderToNum[j]].pop()
                            self.adList[orderToNum[j]].pop()
                            self.adList[orderToNum[j]].pop()
                        ans.append(Action(tra,playerId=self.id))
                        self.getAllAction_plane_dfs_1(ans,tra,[],1)
                        self.getAllAction_plane_dfs_2(ans,tra,[],1)
                        for a in tra:
                            num=getNum(a)
                            self.adList[num].append(a)
                else:
                    cnt=0
        else:
            for i in range(1,13):#飞机,不带牌,以及带未知牌
                if len(adList[orderToNum[i]])>2:
                    cnt+=1
                    for k in range(2,cnt+1):
                        tra=[]
                        nowDouCnt=0
                        for j in range(i - k + 1, i + 1):
                            tra.append(adList[orderToNum[j]][0])
                            tra.append(adList[orderToNum[j]][1])
                            tra.append(adList[orderToNum[j]][2])
                            nowDouCnt+=len(adList[orderToNum[j]])//2
                        ans.append(Action(tra,playerId=self.id))
                        if len(self.cards)-k*3>=k:
                            ans.append(Action(tra,[INF for i in range(k)],playerId=self.id))
                        if douCnt-nowDouCnt>=k:
                            ans.append(Action(tra,[INF for i in range(k*2)],playerId=self.id))
                else:
                    cnt=0
        return ans
    def getDouCnt(self,isSame=True):
        douCnt = 0
        if isSame:
            for i in range(1, 14):  # 找对子
                douCnt += len(self.adList[i])//2
        else:
            for i in range(1, 14):  # 找对子
                douCnt += len(self.adList[i])>0
        return douCnt
    def getOneCnt(self,tar=0):
        douCnt = 0
        for i in range(1, MAXNUM):  # 找单牌
            if len(self.adList[i]) > 0 and getNum(self.adList[i][0])!=tar:  # 找到一个单排
                douCnt += 1
        return douCnt
    def getAllAction(self,firstAct:Action,all=False):  # 该函数获取比之前最大的那个动作大的自己的所有动作，注意，斗地主没有同花顺.这里三带一和飞机，四带2被省略去了。
        #牌的类型有："pass","bomb","seq","douseq","triseq", "triseq_1","triseq_2","solo","dou","tri","tri_1","tri_2","qua_2","qua_2"

        adList = self.getHandList()
        finum=getNum(firstAct.cards[0])
        ans = [Action([],playerId=self.id)]  # 0代表不管

        if firstAct.kind=="bomb":#如果是炸弹
            for i in range(1, 15):  # 炸弹
                if len(adList[i]) > 3 and cardOrder[adList[i][0]]>cardOrder[firstAct.cards[0]]:
                    ans.append(Action(adList[i],playerId=self.id))
            self.pushDouKing(ans)#
        else:#区分类型
            if firstAct.kind=="solo":
                for i in range(1, MAXNUM):  # 单牌
                    if len(adList[i]) > 0 and cardOrder[adList[i][0]] > cardOrder[firstAct.cards[0]]:
                        ans.append(Action([adList[i][0]],playerId=self.id))
            elif firstAct.kind=="dou":
                for i in range(1, 14):  # 对子
                    if len(adList[i]) > 1 and cardOrder[adList[i][0]] > cardOrder[firstAct.cards[0]]:
                        ans.append(Action([adList[i][0],adList[i][1]],playerId=self.id))
            elif firstAct.kind=="tri":
                for i in range(1, 14):  # 三张
                    if len(adList[i]) > 2 and cardOrder[adList[i][0]] > cardOrder[firstAct.cards[0]]:
                        ans.append(Action([adList[i][0],adList[i][1],adList[i][2]],playerId=self.id))
            elif firstAct.kind=="tri_1":
                if all:
                    for i in range(1, 14):  #
                        leni=len(adList[i])
                        if leni>=3 and len(self.cards)-leni>0 and cardOrder[adList[i][0]] > cardOrder[firstAct.cards[0]]:
                            for j in range(1, MAXNUM):
                                if len(adList[j]) > 0 and j != i:
                                    ans.append(Action([adList[i][0], adList[i][1], adList[i][2]], [adList[j][0]],
                                                      playerId=self.id))
                else:
                    for i in range(1, 14):  #
                        if len(adList[i])==3 and len(self.cards)>3 and cardOrder[adList[i][0]] > cardOrder[firstAct.cards[0]]:
                            ans.append(Action([adList[i][0], adList[i][1], adList[i][2]], [INF],playerId=self.id))
                        if len(adList[i])==4 and len(self.cards)>4 and cardOrder[adList[i][0]] > cardOrder[firstAct.cards[0]]:
                            ans.append(Action([adList[i][0], adList[i][1], adList[i][2]], [INF],playerId=self.id))
            elif firstAct.kind=="tri_2":
                if all:
                    dcnt = self.getDouCnt()
                    for i in range(1, 15):  #
                        if len(adList[i]) > 2 and len(adList[i]) // 2 < dcnt and cardOrder[adList[i][0]] > cardOrder[firstAct.cards[0]]:
                            for j in range(1, MAXNUM):
                                if len(adList[j]) > 1 and j != i:
                                    ans.append(Action([adList[i][0], adList[i][1], adList[i][2]], [adList[j][0], adList[j][1]],playerId=self.id))
                else:
                    dcnt=self.getDouCnt()
                    for i in range(1, 15):  #
                        if len(adList[i]) > 2 and len(adList[i])//2<dcnt and cardOrder[adList[i][0]] > cardOrder[firstAct.cards[0]]:
                            ans.append(Action([adList[i][0], adList[i][1], adList[i][2]], [INF,INF],playerId=self.id))

            elif firstAct.kind=="qua_2":
                if all:
                    if len(self.cards) >= 6:
                        for i in range(1, 15):  #
                            if len(adList[i]) == 4 and cardOrder[adList[i][0]] > cardOrder[firstAct.cards[0]]:
                                for j in range(1, MAXNUM):
                                    if len(adList[j]) == 0:
                                        continue
                                    a = adList[j].pop()
                                    for k in range(j, MAXNUM):
                                        if len(adList[k]) > 0 and k!=i and j!=i:
                                            ans.append(Action(adList[i], [a, adList[k][0]], playerId=self.id))
                                    adList[j].append(a)
                else:
                    if len(self.cards) >=6 :
                        for i in range(1, 15):  #
                            if len(adList[i])==4 and cardOrder[adList[i][0]] > cardOrder[firstAct.cards[0]]:
                                ans.append(Action(adList[i], [INF,INF],playerId=self.id))
            elif firstAct.kind == "qua_4":#按照不考虑4带2来
                if len(self.cards) >= 8 and self.getDouCnt(isSame=False)>=3:
                    for i in range(1, 15):  #
                        if len(adList[i]) == 4 and cardOrder[adList[i][0]] > cardOrder[firstAct.cards[0]]:
                            ans.append(Action(adList[i], [INF, INF,INF, INF], playerId=self.id))
            elif firstAct.kind=="seq":
                cnt=0
                order=cardOrder[firstAct.cards[0]]+1
                for i in range(order, 13):
                    if len(adList[orderToNum[i]]) > 0:
                        cnt += 1
                        if cnt>=firstAct.len:
                            ans.append(Action([adList[orderToNum[j]][0] for j in range(i-firstAct.len+1,i+1)],playerId=self.id))
                    else:
                        cnt = 0
            elif firstAct.kind == "douseq":
                cnt = 0
                order = cardOrder[firstAct.cards[0]] + 1
                for i in range(order, 13):
                    if len(adList[orderToNum[i]]) > 1:
                        cnt += 1
                        if cnt >= firstAct.len//2:
                            tra = []
                            for j in range(i - firstAct.len//2 + 1, i + 1):
                                tra.append(adList[orderToNum[j]][0])
                                tra.append(adList[orderToNum[j]][1])
                            ans.append(Action(tra,playerId=self.id))
                    else:
                        cnt = 0
            elif firstAct.kind[:6] == "triseq":
                dcnt=self.getDouCnt()
                cnt = 0
                ficnt=len(firstAct.cards)//3
                order = cardOrder[firstAct.cards[0]] + 1
                for i in range(order, 13):
                    if len(adList[orderToNum[i]]) > 2:
                        cnt += 1
                        if cnt >= ficnt:
                            tra = []
                            douSum = 0
                            for j in range(i - ficnt + 1, i + 1):
                                tra.append(adList[orderToNum[j]][-3])
                                tra.append(adList[orderToNum[j]][-2])
                                tra.append(adList[orderToNum[j]][-1])
                                douSum += len(adList[orderToNum[j]]) // 2
                                self.adList[orderToNum[j]].pop()
                                self.adList[orderToNum[j]].pop()
                                self.adList[orderToNum[j]].pop()
                            if firstAct.kind=="triseq":
                                ans.append(Action(tra,playerId=self.id))

                            elif firstAct.kind=="triseq_1" and len(self.cards)>=4*ficnt:
                                if all:
                                    self.getAllAction_plane_dfs_1(ans, tra, [], 1)
                                else:
                                    ans.append(Action(tra, [INF for i in range(ficnt)], playerId=self.id))
                            elif firstAct.kind=="triseq_2" and dcnt-douSum>=ficnt:
                                if all:
                                    self.getAllAction_plane_dfs_2(ans, tra, [], 1)
                                else:
                                    ans.append(Action(tra, [INF for i in range(ficnt * 2)], playerId=self.id))
                            for a in tra:
                                num = getNum(a)
                                self.adList[num].append(a)
                    else:
                        cnt = 0

            for i in range(1, 15):  #炸弹
                if len(adList[i]) > 3:
                    ans.append(Action(adList[i],playerId=self.id))
            self.pushDouKing(ans)#
        return ans
    def useAction_list(self,actList:list):
        # print(self.cards,actList)
        for a in actList:
            self.cards.remove(a)
            self.uncards.append(a)
        self.getHandList()
        if len(self.cards)==0:
            self.env.done=False
    def useAction(self,act:Action,INFfun):#使用一组牌,INFfun是怎么处理INF
        if act.isPass():
            self.last_move = act.clone()
            self.env.historyAct.append(act)
            return
        self.env.updateScList(act)
        self.env.bomb_num+=act.isBomb()
        self.useAction_list(act.cards)
        if len(act.appendix)>0 and act.appendix[0]==INF:
            if act.kind=="tri_2" or act.kind=="triseq_2":#需要带对子
                for i in range(0,len(act.appendix),2):
                    allActList=self.getAllDouAction()
                    actid=INFfun(i,act.cards+act.appendix[0:i],allActList)
                    selAct=allActList[actid]
                    act.appendix[i]=selAct.cards[0]
                    act.appendix[i+1] = selAct.cards[1]
                    self.useAction_list(selAct.cards)
                    # print("selAct:" + selAct.toString())
                    # self.printCards()
            else:
                for i in range(len(act.appendix)):
                    allActList = self.getAllOneAction(getNum(act.cards[0]))
                    actid = INFfun(i,act.appendix[0:i], allActList)
                    selAct = allActList[actid]
                    act.appendix[i] = selAct.cards[0]
                    self.useAction_list(selAct.cards)
        else:
            self.useAction_list(act.appendix)
        self.last_move = act.clone()
        self.env.historyAct.append(act)

    def printCards(self):
        print("玩家{}:手牌".format(self.id),end=" ")
        k=0
        for a in self.cards:
            Card(a).print(k)
            k+=1
        print("")
        print("玩家{}已用".format(self.id), end=" ")
        k = 0
        for a in self.uncards:
            Card(a).print(k)
            k += 1
        print("")
class Doudizhu():
    def __init__(self):
        self.name="doudizhu"
        self.firstBidPlayerId=0
        self.reset()
    def reset(self):
        self.players = [Player(self,0), Player(self,1), Player(self,2)]  # 0代表没有
        self.historyAct = []  # 3个人已经出去的卡牌
        self.underCards = np.zeros(UNDERCARD_CNT,dtype='int')  # 底牌
        self.nowDealCardPlayer = 0  # 当前到发牌的玩家
        self.round_i = 0  # 轮数
        self.deck = np.arange((CARDS_CNT),dtype='int') + 1  # 卡组
        self.useCards_i=0#放回deck
        self.dealer=-1
        self.lordCard =-1
        self.bomb_num=0
        self.done=True#游戏是否要继续，True代表继续
    def CrossShuffle(self):#交叉洗牌
        for i in range(CARDS_CNT // 2-1,-1,-1):  # 交叉洗牌
            rnd1, rnd2 = i, i * 2
            self.deck[rnd1], self.deck[rnd2] = self.deck[rnd2], self.deck[rnd1]

    def shuffleDeck(self,T=10):
        up=len(self.deck)-1
        # self.CrossShuffle()
        # print(self.deck)
        while(T>0):
            self.CrossShuffle()

            for i in range(up, 0, -1):
                rnd = random.randint(0, i)  # 每次随机出0-i-1之间的下标
                # rnd1 = random.randint(0, up)  # 每次随机出0-i-1之间的下标
                # rnd2 = random.randint(0, up)  # 每次随机出0-i-1之间的下标
                # print(rnd)
                self.deck[i], self.deck[rnd] = self.deck[rnd], self.deck[i]
                # self.deck[rnd1], self.deck[rnd2] = self.deck[rnd2], self.deck[rnd1]
            T-=1
    def bidLord(self,bidFun):#叫主
        actList=[0,1,2,3]
        already=np.array([-1,-1,-1])
        for i in range(3):  # 抢主
            pid = (self.firstBidPlayerId + i) % 3
            act=bidFun(i,pid,already,actList)
            already[pid]=act
            for j in range(1,act+1):
                if j in actList:
                    actList.remove(j)
        return int(np.argmax(already)),already
    def dealCards(self,beginDeckList=None,setDealer=-1,bidFun=None):#发牌，轮流先叫地主
        self.scList = [1, 1, 1]  # 玩家倍数
        if beginDeckList is None:
            self.shuffleDeck()  # 洗牌
        else:
            self.deck=beginDeckList.copy()
        # print(self.deck.tolist())
        self.deck_i = 0
        self.nowDealCardPlayer=0
        while (self.deck_i < CARDS_CNT - UNDERCARD_CNT):
            self.players[self.nowDealCardPlayer].addCard(self.deck[self.deck_i])
            self.deck_i += 1
            self.nowDealCardPlayer = (self.nowDealCardPlayer + 1) % 3

        if setDealer!=-1:#人为设置，调试用
            self.lordCard = random.randint(0, CARDS_CNT - UNDERCARD_CNT - 1)
            self.dealer = setDealer

            self.lordCard = self.deck[self.lordCard]
        else:
            # self.lordCard=random.randint(0, CARDS_CNT - UNDERCARD_CNT-1)
            if bidFun==None:
                self.dealer = self.firstBidPlayerId
            else:
                self.dealer,bidList=self.bidLord(bidFun)
                if sum(bidList)==0:#都不叫这局重开
                    # self.dealer=random.randint(0,2)
                    return False
                for i in range(3):
                    self.scList[i] *= max(1,bidList[i])
                # print("叫分情况：",bidList)
            self.firstBidPlayerId=(self.firstBidPlayerId+1)%3
            self.lordCard=self.deck[self.lordCard]

        self.players[self.dealer].addCard(self.deck[self.deck_i])
        self.players[self.dealer].addCard(self.deck[self.deck_i+1])
        self.players[self.dealer].addCard(self.deck[self.deck_i+2])
        for i in range(3):
            self.players[i].sortCards()
            self.players[(i+self.dealer)%3].dealerTag=i
            self.players[i].beginCards=self.players[i].cards.copy()
        self.underCards=self.deck[-3:].copy()
        self.scList[self.dealer]*= 2
        self.beginScList= self.scList.copy()
        return True
        # print("地主是玩家："+str(self.dealer),"地主牌是"+cardToString(self.lordCard))


    def step(self,act1:Action,act2:Action,p1,p2):#判断,比较大小,p1,p2代表智能体编号(不是player编号)。返回赢得那个人
        bomb1=act1.isBomb()
        bomb2 = act2.isBomb()
        pass1=act1.isPass()
        pass2 = act2.isPass()
        anspid=(p1,p2)
        ansact=(act1,act2)
        if bomb1 and bomb2:
            fp=cardOrder[act1.cards[0]]<cardOrder[act2.cards[0]]
        elif bomb2 or pass1:
            fp=1
        elif bomb1 or pass2:
            fp=0
        else:#其余情况只看第一张牌大小就可以了
            fp = cardOrder[act1.cards[0]] < cardOrder[act2.cards[0]]
        winPlayer=-1#winPlayer!=-1代表没有人赢
        if len(self.players[act1.playerId].cards)==0:
            winPlayer=p1
        if len(self.players[act2.playerId].cards)==0:
            winPlayer=p2
        return anspid[fp],ansact[fp],winPlayer
    def getLastHistoryAct(self):#找到第一个不是pass的历史动作
        for i in reversed(range(len(self.historyAct))):
            if not self.historyAct[i].isPass():
                return self.historyAct[i]
        return None

    def updateScList(self,act):
        if act.isBomb():  # 如果没选取动作就会报错
            for i in range(3):
                self.scList[i] = min(self.scList[i]*2,self.beginScList[i]*16)
    def classActKind(self,allActList):
        # "pass", "bomb", "seq", "douseq", "triseq", "triseq_1", "triseq_2", "solo", "dou", "tri", "tri_1", "tri_2", "qua_2"
        actkindList=[[] for i in range(13)]
        for i in range(len(allActList)):
            act=allActList[i]
            actkindList[actKindToId[act.kind]].append(act)
        return actkindList

    def printPlayerHand(self):
        for i in range(3):
            self.players[i].printCards()

def __dfsPrintActList(newLi, li0,printFun=None):#printFun是打印这张牌的条件
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
            __dfsPrintActList(t,li0[i])
            if printFun==None or printFun(t):
                newLi.append(t)
def dfsPrintActList(li,printFun=None):
    newLi=[]
    if isinstance(li,Action):
        newLi.append(li.toString())
    else:
        __dfsPrintActList(newLi,li,printFun)
    print(newLi)
def printDeckListForId(deck):
    print("[",end="")
    for a in deck:
        print(a, end=',')
    print("]")
def _sortAction_cmp1(a:Action,b:Action):#按照顺序排序,大王小王排到最后
    if a.len==b.len:
        return cardOrder[a.cards[0]]-cardOrder[b.cards[0]]
    return a.len-b.len
def baselinePolicy_first(env:Doudizhu,nowPlayerid,maxPlayerid,allAct:list,INFfun):#人类经验设计的baseline：如果敌方大就无脑管住，如果我方大就不管
    n = len(allAct)
    allAct.sort(key=cmp_to_key(_sortAction_cmp1))
    newList=[]
    prelen=-1
    for a in allAct:
        if a.len!=prelen:
            newList.append(a)
            prelen=a.len
    id = random.randint(0, len(newList)- 1)
    env.players[nowPlayerid].useAction(newList[id], INFfun)
    return id, newList[id]
def baselinePolicy(env:Doudizhu,nowPlayerid,maxPlayerid,allAct:list,INFfun):#人类经验设计的baseline：如果敌方大就无脑管住，如果我方大就不管
    # n = len(allAct)
    # allAct.sort(key=cmp_to_key(_sortAction_cmp1))
    id=0
    # print(len(allAct))
    if (env.players[maxPlayerid].dealerTag * env.players[nowPlayerid].dealerTag) ==0 and len(allAct)>1:
        id = random.randint(1, len(allAct) - 1)
    env.players[nowPlayerid].useAction(allAct[id], INFfun)
    return id, allAct[id]
def baselinePolicy_INFfun(cardi,appendix,allActList):#选择最小的牌
    n=len(allActList)
    # dfsPrintActList(allActList)
    mina=INF
    act_id=0
    for i in range(n):
        act=allActList[i]
        if mina>cardOrder[act.cards[0]]:
            mina = cardOrder[act.cards[0]]
            act_id=i
    return act_id
def randomPolicy_bidFun(i,pid,already,actList):#叫牌网络
    act_id = random.randint(0, len(actList) - 1)
    return actList[act_id]