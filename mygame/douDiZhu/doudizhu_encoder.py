import torch
import doudizhu_game
from doudizhu_game import Player,getNum,Card,getDecor,Action,Doudizhu

codeWide=54
def getBombFea(p:Player):
    adList = p.getHandList()
    ans = torch.zeros((15))
    for i in range(1, 14):  # 找炸弹
        if len(adList[i])==4:
            ans[i-1]=1
    if len(adList[14])==2:
        ans[14] = 1
    return ans
def handTo01code(p:Player):#handTo01code,维度54+15+3=72
    ans = torch.zeros((codeWide))
    x=getBombFea(p)
    belong=torch.zeros((3))
    belong[p.id]=1
    for card in p.cards:
        a=int(card)
        if a>0:
            ans[a-1]+=1
    return torch.cat((ans,x,belong), dim=0)
# def cardsTo01code(cadsList:list,playerId):
#     ans = torch.zeros((codeWide+4))
#     for card in cadsList:
#         a = int(card)
#         if a>0:
#             ans[a-1]+=1
#             ans[54] += fen[a]
#     ans[playerId] = 1
#     return ans
def actTo01code(act: Action):#playerId代表属于哪个人
    playerId=act.playerId
    ans = torch.zeros((codeWide+3))
    for a in act.cards:
        if a > 0 and a<=codeWide:
            ans[int(a)-1] += 1
    for a in act.appendix:
        if a > 0 and a<=codeWide:
            ans[int(a)-1] += 1
    ans[codeWide+playerId]=1
    return ans
def actZeros():#playerId代表属于哪个人
    ans = torch.zeros((codeWide+3))
    return ans
def underCardsFea(env: Doudizhu):#底牌的特征向量
    ans = torch.zeros((codeWide+3))
    for a in env.underCards:
        ans[int(a) - 1]+= 1
    ans[codeWide + env.dealer] = 1
    return ans

def getBaseFea(env:Doudizhu,selfId,maxCardAct,seeAllCards=True):#BaseFea代表底牌的特征向量，主牌特征向量，已经出过的牌的特征向量，每个人手牌特征向量。selfId代表玩家编号，只能可见自己的手牌，如果selfId<0,则可见所有人手牌
    #useSeq为出牌顺序
    # 包括手牌、其他玩家出的牌、上家的牌等特征矩阵以及其他玩家手牌数量和炸弹数量的0 / 1
    # 编码
    if seeAllCards:
        x = torch.cat((handTo01code(env.players[0]), handTo01code(env.players[1]), handTo01code(env.players[2])), dim=0)
    else:
        x=torch.zeros((codeWide*3))
        x[selfId*codeWide:(selfId+1)*codeWide]=handTo01code(env.players[selfId])
    if maxCardAct==None:
        maxCardActFea = actZeros()
    else:
        maxCardActFea = actTo01code(maxCardAct)  # 动作和所属的人
    underCards=underCardsFea(env)
    ans=torch.cat((x,maxCardActFea, underCards), dim=0)
    return ans.unsqueeze(dim=0)
def getAllActionFea(allActList:list):#得到全部动作的向量
    n=len(allActList)
    ans = torch.zeros((n, codeWide+3))
    # print(firstPlayerId,actList)
    for i in range(n):
        ans[i] = actTo01code(allActList[i])
    return ans

# li=[actTo01code(Action([1,2])) for i in range(1)]
# x=torch.stack(li,dim=0)
# print(x.shape)
# tensor = x.repeat(100, 1, 1)
# print(tensor)





