import torch
from DNQ.mygame.TractorArtifice.game_env.tractor_game import Player,fenInd, Action,CC
codeWide=55
fen=[fenInd[i%13]/10 for i in range(0,55)]
# print(fen)
def handTo01code(p:Player):#handTo01code
    ans = torch.zeros((codeWide))
    for i in range(5):
        for j in range(p.cards_decorLen[i]):
            a=int(p.cards_decorList[i][j])
            if a>0:
                ans[a-1]+=1
                ans[54]+=fen[a]
    return ans
def cardsTo01code(cadsList:list,playerId):
    ans = torch.zeros((codeWide+4))
    for card in cadsList:
        a = int(card)
        if a>0:
            ans[a-1]+=1
            ans[54] += fen[a]
    ans[codeWide+playerId] = 1
    return ans
def actTo01code(act: Action,playerId):#playerId代表属于哪个人
    ans = torch.zeros((codeWide+4))
    for a in act.one:
        if a > 0:
            ans[int(a)-1] += 1
            ans[54] += fen[int(a)]
    for dou in act.double:
        for a in dou:
            if a > 0:
                ans[int(a)-1] += 1
                ans[54] += fen[int(a)]
    ans[codeWide+playerId]=1
    return ans
def lordNumFea(env: CC,useSeq,selfId,isDealer):#编码主牌花色和数字，以及自己出牌顺序.以及自己是否为庄家
    ans = torch.zeros((27))
    ans[env.lordNum-1]=1
    ans[env.lordDecor+13] = 1
    ans[17+useSeq] = 1
    ans[21 + selfId] = 1
    ans[25] = isDealer
    ans[26] = env.sumSc/5
    return ans
def tableCardFea(env: CC):#已经出过的牌的特征向量
    ans = torch.zeros((codeWide))
    for i in range(4):
        li=env.playerTable[i]
        for j in range(env.playerTable_i[i]):
            a=int(li[j])
            ans[a-1]+=1
            ans[54] += fen[a]
    return ans
def underCardsFea(env: CC):#底牌的特征向量
    ans = torch.zeros((codeWide))
    for a in env.underCards:
        ans[int(a) - 1]+= 1
        ans[54] += fen[int(a)]
    return ans

def getBaseFea(env:CC,useSeq,selfId,isDealer,seeAllCards=True,seeUnder=True):#BaseFea代表底牌的特征向量，主牌特征向量，已经出过的牌的特征向量，每个人手牌特征向量。selfId代表玩家编号，只能可见自己的手牌，如果selfId<0,则可见所有人手牌
    #useSeq为出牌顺序
    if seeAllCards:
        x = torch.cat((handTo01code(env.players[0]), handTo01code(env.players[1]), handTo01code(env.players[2]), handTo01code(env.players[3])), dim=0)
    else:
        x=torch.zeros((codeWide*4))
        x[selfId*codeWide:(selfId+1)*codeWide]=handTo01code(env.players[selfId])
    underCards= torch.zeros((codeWide))
    if seeUnder:
        underCards=underCardsFea(env)
    ans=torch.cat((x, underCards, lordNumFea(env,useSeq,selfId,isDealer)), dim=0)
    ans = torch.cat((x, underCards, lordNumFea(env, useSeq, selfId, isDealer)), dim=0)
    return ans.unsqueeze(dim=0)
def getActionFeature(actList:list,firstPlayerId,layer=4):#动作列表
    ans = torch.zeros((layer, codeWide + 4))
    # print(firstPlayerId,actList)
    for i in range(layer):
        id = (firstPlayerId + i) % 4
        if isinstance(actList[id], Action):  # 类型是Action
            ans[i] = actTo01code(actList[id], id)
    return ans.unsqueeze(dim=0)
def addActionFeature(ans,actList:list,firstPlayerId,tarId):#往动作向量里插入动作列表
    # print(firstPlayerId,actList)

    i=(tarId-firstPlayerId+4)%4
    ad= cardsTo01code(actList, tarId)
    ans[0][i] +=ad





