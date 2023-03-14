import copy
import torch
import torch.nn.functional as F
import numpy as np

import torch
from torch import nn

class AgentNet(nn.Module):#
    def __init__(self,in_fea):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=14*6, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=14*6, hidden_size=128, batch_first=True)#本LSTM接受出牌的动作序列，可能只有1，也可能很多。
        self.mlp_base = nn.Sequential(  # 求动作
            nn.Linear(in_fea, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 512),
        )
        self.mlp_act1 = nn.Sequential(  # 求是否管上前面的人，二分类问题
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )
        self.mlp_act2 = nn.Sequential(  # 求动作平分
            nn.Linear(512+128, 512),#512是base网络的输出，128是单张牌之间lstm的输出
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.mlp_q = nn.Sequential(
            nn.Linear(512+14*6, 512),#512是base网络的输出，128是单张牌之间lstm的输出
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward_base(self, seeingCards, handCards, underCards, epochCards):
        # seeingCards,handCards,underCards,epochCards：每个玩家已经出的牌，自己手牌,庄家扣的底牌,本轮其他玩家出的牌,分数
        # epochCards = epochCards.view(epochCards.size(0), -1).unsqueeze(dim=0)
        _, (hc, hn) = self.lstm1(epochCards)
        hc = hc.squeeze(dim=0)
        handCards = handCards.view(handCards.size(0), -1)
        seeingCards = seeingCards.view(seeingCards.size(0), -1)
        underCards = underCards.view(underCards.size(0), -1)
        # print(seeingCards.shape)
        x = torch.cat((hc, handCards, seeingCards, underCards), dim=1)
        x = self.mlp_base(x)
        # print(x.shape)
        return x
    def forward_bin(self,x):#二分类,x是forward_base的结果
        x = self.forward_base(seeingCards, handCards, underCards, epochCards)
        x=self.mlp_act1(x)
        x=F.log_softmax(x, dim=1)
        return x

    def forward_act(self, x,act):  # actor网络，x是forward_base的结果
        _,(act,c)=self.lstm2(act)
        x = torch.cat((x, act), dim=1)
        x = self.mlp_act2(x)
        return x
    def forward_q(self,x,act):#估计q，x是forward_base的结果
        x = torch.cat((x, act),dim=1)
        x=self.mlp_q(x)
        return x

class AmbushNet(nn.Module):#埋伏牌的网络
    def __init__(self,in_fea):
        super().__init__()
        self.mlp_ambush_q= nn.Sequential(
            nn.Linear(in_fea, 512),#512是base网络的输出，128是单张牌之间lstm的输出
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.mlp_ambush_action =  nn.Sequential(
            nn.Linear(in_fea, 512),#512是base网络的输出，128是单张牌之间lstm的输出
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    def forward_act(self,x,act):# actor网络，x是forward_base的结果
        x = torch.cat((x, act), dim=1)
        x = self.mlp_ambush_action(x)
        return x
    def forward_q(self,x,act):#估计q，x是forward_base的结果
        x = torch.cat((x, act), dim=1)
        x = self.mlp_ambush_q(x)
        return x
net=AgentNet()
anet=AmbushNet()
seeingCards, handCards, underCards, epochCards=torch.ones((1,14*6)),torch.ones((1,4,14*6)),torch.ones((1,14*6)),torch.ones((1,3,14*6))
basex=net.forward_base(seeingCards, handCards, underCards, epochCards)
act=torch.ones((3,14*6))
act1=torch.ones((1,14*6))
x=net.forward_act(basex,act)
x1=anet.forward_act(basex,act1)
print(x.shape,x1.shape)
# lstm1 = nn.LSTM(input_size=14*6, hidden_size=128, batch_first=True)
# x=torch.ones((10,3,14*6))
# x,(h,c)=lstm1(x)
# h=h.squeeze(dim=0)
# print(x.shape,h.shape)