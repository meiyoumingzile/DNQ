import torch
import tractor_game
from tractor_game import Player,env,fenInd,getNum,Card,getDecor

def handTo01code(p:Player):
    ans = torch.zeros((54))
    for i in range(5):
        for j in range(p.cards_decorLen[i]):
            a=int(p.cards_decorList[i][j])
            if a>0:
                ans[a-1]+=1
    return ans
def cardsTo01code(cadsList):#要传入是否是分和是否是主牌
    ans = torch.zeros((54))
    for card in cadsList:
        if card>0:
            ans[int(card - 1)]+=1
    return ans



