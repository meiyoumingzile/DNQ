import torch
from doudizhu_game import Player,getNum,Card,getDecor,Action,Doudizhu,actKindToId

from doudizhu_game import orderToNum

codeWide=54
def getBelongFea(p:Player):#1代表地主
    ans = torch.zeros((3))
    ans[p.dealerTag]=1
    return ans
def getBombFea(cards):
    ans = torch.zeros((15))
    for a in cards:  # 找炸弹
        ans[getNum(a)-1]+=1
    for i in range(13):
        ans[i]=(ans[i]==4)
    ans[13] = (ans[13] == 1 and ans[14] == 1)
    return ans[0:-1]
def unhandTo01code(p:Player):#handTo01code,得到玩家已经出的牌的特征向量包括炸弹数量
    ans = cardsTo01code(p.uncards)
    belong = getBelongFea(p)
    return torch.cat((ans, belong), dim=0)
def handTo01code(p:Player):#handTo01code,得到玩家剩余手牌的图向量
    ans = cardsTo01code(p.cards)
    belong = getBelongFea(p)
    return torch.cat((ans,belong), dim=0)
def handToCnt(p:Player):#handTo01code,得到玩家剩余手牌的向量
    sumcards=len(p.cards)+len(p.uncards)
    return torch.Tensor([len(p.cards)/sumcards,len(p.uncards)/sumcards])#返回手里已有的牌的数量，剩余的牌的数量
def handTo01code_not(p:Player):#handTo01code,得到不可见的玩家的手牌向量
    return torch.zeros((15))
def cardsTo01code(cards):#
    ans = torch.zeros((15))
    for card in cards:
        a = getNum(card)
        ans[a - 1] += 1
    return ans
def __cardsToGraph_seq(ans,l,r):
    for j in range(l, r):
        for k in range(l, r):
            ans[orderToNum[j] - 1][orderToNum[k] - 1] += 1
def cardsToGraph(cards):#
    ans = torch.zeros((15,15))
    cnt = [0]*15
    for card in cards:
        a = getNum(card)
        cnt[a - 1] += 1
    for i in range(0, 15):
        ans[i][i] = cnt[i]

    for i in range(0, 15):
        if cnt[i]==3:
            for j in range(15):
                if cnt[j]>0:
                    ans[i][j]+=1
        if cnt[i]==4:
            for j in range(15):
                if cnt[j]>0:
                    ans[i][j]+=1
    seq1 =i= 0
    for i in range(1, 13):  # 顺子
        num=orderToNum[i] - 1
        if cnt[num] > 0:
            seq1 += 1
        else:
            if seq1>=5:
                __cardsToGraph_seq(ans,i - seq1 + 1, i + 1)
            seq1 = 0
    if seq1 >= 5:
        __cardsToGraph_seq(ans,i - seq1 + 1, i + 1)

    seq1 =i= 0
    for i in range(1, 13):  # 顺子
        num=orderToNum[i] - 1
        if cnt[num] > 1:
            seq1 += 1
        else:
            if seq1>=3:
                __cardsToGraph_seq(ans,i - seq1 + 1, i + 1)
            seq1 = 0
    if seq1 >=3:
        __cardsToGraph_seq(ans,i - seq1 + 1, i + 1)

    # print(edgeList)
    # print(edgeList1, edgeList2)
    if cnt[13]>0 and cnt[14]>0:
        ans[13][14] += 1
        ans[14][13] += 1
    return ans.unsqueeze(dim=0)


def actTo01code(act: Action,player:Player):#playerId代表属于哪个人
    cnt = [0]*15
    for card in act.cards:
        if card > 0 and card <= codeWide:
            a = getNum(card)
            cnt[a - 1] += 1
    for card in act.appendix:
        if card > 0 and card <= codeWide:
            a = getNum(card)
            cnt[a - 1] += 1
    ans=torch.Tensor(cnt)
    belong = getBelongFea(player)
    return torch.cat((ans,belong), dim=0)
def actZeros():#playerId代表属于哪个人
    ans = torch.zeros((15+3))
    return ans

def getBaseFea(env:Doudizhu,selfId,preActQue,otherCards=None,kind=0):#BaseFea代表底牌的特征向量，主牌特征向量，已经出过的牌的特征向量，每个人手牌特征向量。selfId代表玩家编号，只能可见自己的手牌，如果selfId<0,则可见所有人手牌
    #useSeq为出牌顺序
    # 包括手牌、其他玩家出的牌、上家的牌等特征矩阵以及其他玩家手牌数量和炸弹数量的编码
    #kind==0代表可以看见全部人的手牌,kind==1代表不可看见其他玩家的手牌,kind==2代表记录下其他玩家已经出过的牌，但不可知道没出过的
    graphFeaList=[]
    if kind==0:
        # x = torch.cat((handTo01code(env.players[0]), handTo01code(env.players[1]), handTo01code(env.players[2])), dim=0)
        for i in range(3):
            graphFeaList.append(handTo01code(env.players[(selfId+i)%3]))
        for i in range(3):
            graphFeaList.append(unhandTo01code(env.players[(selfId+i)%3]))
    elif kind==1:
        for i in range(3):
            graphFeaList.append(handTo01code_not(env.players[(selfId+i)%3]))
        for i in range(3):
            graphFeaList.append(unhandTo01code(env.players[(selfId+i)%3]))

    x_cnt=torch.cat((handToCnt(env.players[selfId]), handToCnt(env.players[(selfId+1)%3]), handToCnt(env.players[(selfId+2)%3])), dim=0)
    n=len(preActQue)
    if n==0:
        que1,que2 = actZeros(),actZeros()
    elif n==1:
        que1, que2 = actZeros(), actTo01code(preActQue[0],env.players[preActQue[0].playerId])
    else:
        que1, que2 = actTo01code(preActQue[-1],env.players[preActQue[-1].playerId]), actTo01code(preActQue[0],env.players[preActQue[0].playerId])
    scList=torch.log2(torch.Tensor(env.scList))
    graphFea=cardsToGraph(env.players[selfId].cards)
    ans=torch.cat(graphFeaList+[que1,que2,x_cnt,scList], dim=0)
    return ans.unsqueeze(dim=0),graphFea

def getAllActionFea(env:Doudizhu,allActList:list):#得到全部动作的向量
    n=len(allActList)
    ans = torch.zeros((n, 15+3))
    # print(firstPlayerId,actList)
    for i in range(n):
        # zeros=torch.zeros((14))
        # zeros[actKindToId[allActList[i].kind]]=1
        ans[i] = actTo01code(allActList[i],env.players[allActList[i].playerId])

    return ans
def getAllActionKindFea(env:Doudizhu,allAct_kind:list):#得到全部动作的向量
    n=len(allAct_kind)
    ans = []
    ans_id=[]
    for i in range(len(allAct_kind)):
        if len(allAct_kind[i]) != 0:
            zeros = torch.zeros((14))
            zeros[actKindToId[allAct_kind[i][0].kind]] = 1
            x=torch.cat((torch.zeros((codeWide+3)),zeros),dim=0)
            ans.append(x)
            ans_id.append(i)
    return ans_id,torch.stack(ans)

# li=[actTo01code(Action([1,2])) for i in range(1)]
# x=torch.stack(li,dim=0)
# print(x.shape)
# tensor = x.repeat(100, 1, 1)
# print(tensor)





