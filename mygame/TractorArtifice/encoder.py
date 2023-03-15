import torch
import tractor_game
from tractor_game import Player,fenInd,getNum,Card,getDecor,Action,CC
# actCodeFea=54

def handTo01code(p:Player):#handTo01code
    ans = torch.zeros((54))
    for i in range(5):
        for j in range(p.cards_decorLen[i]):
            a=int(p.cards_decorList[i][j])
            if a>0:
                ans[a-1]+=1
    return ans
def cardsTo01code(cadsList:list):
    ans = torch.zeros((54))
    for card in cadsList:
        if card>0:
            ans[int(card - 1)]+=1
    return ans
def actTo01code(act: Action):
    ans = torch.zeros((54))
    for a in act.one:
        if a > 0:
            ans[int(a)-1] += 1
    for dou in act.double:
        for a in dou:
            if a > 0:
                ans[int(a)-1] += 1
    return ans
def lordNumFea(env: CC):
    ans = torch.zeros((17))
    ans[env.lordNum-1]=1
    ans[env.lordDecor+13] = 1
    return ans
def tableCardFea(env: CC):#已经出过的牌的特征向量
    ans = torch.zeros((54))
    for i in range(4):
        li=env.playerTable[i]
        for j in range(env.playerTable_i[i]):
            ans[int(li[j])-1]+=1
    return ans
def underCardsFea(env: CC):#底牌的特征向量
    ans = torch.zeros((54))
    for a in env.underCards:
        ans[int(a) - 1]+= 1
    return ans

def getBaseFea(env:CC,selfId,seeUnder=True):#BaseFea代表底牌的特征向量，主牌特征向量，已经出过的牌的特征向量，每个人手牌特征向量。selfId代表玩家编号，只能可见自己的手牌，如果selfId<0,则可见所有人手牌
    if selfId<0:
        x = torch.cat((handTo01code(env.players[0]), handTo01code(env.players[1]), handTo01code(env.players[2]), handTo01code(env.players[3])), dim=0)
    else:
        x=torch.zeros((54*4))
        x[selfId*54:(selfId+1)*54]=handTo01code(env.players[selfId])
    ans=torch.cat((x, tableCardFea(env), seeUnder and underCardsFea(env) or torch.zeros((54)), lordNumFea(env)), dim=0)
    return ans