import copy
import torch
import torch.nn.functional as F
import numpy as np

import torch
from torch import nn
mainfea=512
def orthogonal_init(layer, gain=1.0):
    for m in layer.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            nn.init.orthogonal_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
class ResBlock(nn.Module):#跳2层
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, out_channels),
        )
        self.fc1=nn.Linear(out_channels, out_channels)
        self.tanh= nn.Tanh()
        orthogonal_init(self.fc)
        orthogonal_init(self.fc1)
    def forward(self, x):
        identity = x
        out = self.fc(x)
        out += identity
        out =self.tanh(out)
        out1= self.fc1(out)
        out1 += identity
        out1 += out
        out1 = self.tanh(out1)
        return out1
class AgentCriNet(nn.Module):#
    def __init__(self,in_fea=302+55):#350
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=59, hidden_size=64, num_layers=2,batch_first=True)
        self.lstm2 = nn.LSTM(input_size=59, hidden_size=128,num_layers=2,  batch_first=True)#本LSTM接受出牌的动作序列，可能只有1，也可能很多。
        self.mlp_base = nn.Sequential(  # 求动作
            nn.Linear(in_fea+64+128, mainfea),
            nn.Tanh(),
            ResBlock(mainfea,mainfea),
            # nn.Linear(1024, 512),
            # ResBlock(512,512),
            nn.Linear(mainfea, 1),
        )
        # orthogonal_init(self.lstm1)
        # orthogonal_init(self.lstm2)
        orthogonal_init(self.mlp_base)

    def forward(self, baseFea,actFea,hisActFea):  # 估计q，x是forward_base的结果
        x1, (hc1, hn) = self.lstm1(actFea)
        # hc1 = hc1.squeeze(dim=0)
        x2, (hc2, hn) = self.lstm2(hisActFea)
        x1=x1[:,-1,:]
        x2=x2[:,-1,:]
        # print(hc.shape,baseFea.shape)
        x = torch.cat((baseFea, x1,x2), dim=1)
        x = self.mlp_base(x)
        return x

    def printGrad(self):
        print("criticNetGrad:")
        i=0
        for m in self.mlp_base.modules():
            if isinstance(m, nn.Linear):
                print("mlp_base:" + str(i), end=" ")
                i += 1
                print(m.weight.grad)
class AgentNet(nn.Module):#
    def __init__(self,in_fea=302+55):#350
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=59, hidden_size=64, num_layers=2,batch_first=True)
        self.lstm2 = nn.LSTM(input_size=59, hidden_size=128,num_layers=2,  batch_first=True)#本LSTM接受出牌的动作序列，可能只有1，也可能很多。
        self.mlp_base = nn.Sequential(  # 求动作
            nn.Linear(in_fea+64+128, mainfea),
            nn.Tanh(),
            ResBlock(mainfea, mainfea),
        )
        self.mlp_act1 = nn.Sequential(  # 求是否管上前面的人，二分类问题，是actor网络
            nn.Linear(mainfea, 2),
            # nn.Sigmoid(),
        )
        self.mlp_act2 = nn.Sequential(  # 求动作平分,是actor网络
            nn.Linear(mainfea +59, mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, 1),#512是base网络的输出，128是单张牌之间lstm的输出
            # nn.Sigmoid(),
        )
        # orthogonal_init(self.lstm1)
        # orthogonal_init(self.lstm2)
        orthogonal_init(self.mlp_base)
        orthogonal_init(self.mlp_act1)
        orthogonal_init(self.mlp_act2)
        # self.mlp_q = nn.Sequential(# critic网络,估计Q
        #     nn.Linear(512, 512),#512是base网络的输出，128是单张牌之间lstm的输出
        #     nn.Tanh(),
        #     nn.Linear(512, 1),
        # )

    def forward_base(self, baseFea,actFea,hisActFea):
        # seeingCards,handCards,underCards,epochCards：每个玩家已经出的牌，自己手牌,庄家扣的底牌,本轮其他玩家出的牌,分数
        # epochCards = epochCards.view(epochCards.size(0), -1).unsqueeze(dim=0)
        # print(baseFea.shape)
        x1, (hc1, hn) = self.lstm1(actFea)
        # hc1 = hc1.squeeze(dim=0)
        x2, (hc2, hn) = self.lstm2(hisActFea)
        x1=x1[:,-1,:]
        x2=x2[:,-1,:]
        # print(x1.shape,x2.shape,hc1[1].shape)
        # hc2 = hc2.view(-1)
        # hc2 = hc2.squeeze(dim=0)
        # print(hc2.shape)
        # print(hc.shape,baseFea.shape)
        x = torch.cat((baseFea, x1,x2), dim=1)
        x = self.mlp_base(x)
        # print(x.shape)
        return x
    def forward_bin(self,x):#二分类,x是forward_base的结果
        # x = self.forward_base(x)
        x=self.mlp_act1(x)
        x=F.softmax(x, dim=1)
        return x

    def forward_act(self, x,act):  # actor网络，x是forward_base的结果
        # _,(act,c)=self.lstm2(act)
        # act = act.squeeze(dim=0)
        x = torch.cat((x, act), dim=1)
        # print(x)
        x = self.mlp_act2(x)
        return x

    def printGrad(self):
        print("actorNetGrad:")
        i=0
        for m in self.mlp_base.modules():
            if isinstance(m, nn.Linear):
                print("mlp_base:"+str(i),end=" ")
                i+=1
                print(m.weight.grad)
        i = 0
        for m in self.mlp_act1.modules():
            if isinstance(m, nn.Linear):
                print("mlp_act1:" + str(i), end=" ")
                i += 1
                print(m.weight.grad)
        i = 0
        for m in self.mlp_act2.modules():
            if isinstance(m, nn.Linear):
                print("mlp_act2:" + str(i), end=" ")
                i += 1
                print(m.weight.grad)
class AmbushNet(nn.Module):#埋伏牌的网络
    def __init__(self,in_fea):
        super().__init__()
        self.mlp_ambush_q= nn.Sequential(
            nn.Linear(in_fea, 512),#512是base网络的输出，128是单张牌之间lstm的输出
            nn.Tanh(),
            nn.Linear(512, 1),
        )
        self.mlp_ambush_action =  nn.Sequential(
            nn.Linear(in_fea, 512),#512是base网络的输出，128是单张牌之间lstm的输出
            nn.Tanh(),
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

# net=AgentNet()
# # anet=AmbushNet()
# x,act=torch.ones((1,14*6)),torch.ones((1,4,14*6)),torch.ones((1,14*6)),torch.ones((1,3,14*6))
# basex=net.forward_base(seeingCards, handCards, underCards, epochCards)
# act=torch.ones((3,14*6))
# act1=torch.ones((1,14*6))
# x=net.forward_act(basex,act)
# x1=anet.forward_act(basex,act1)
# print(x.shape,x1.shape)
# lstm1 = nn.LSTM(input_size=14*6, hidden_size=128, batch_first=True)
# x=torch.ones((32,3,14*6))
# x,(h,c)=lstm1(x)
# h=h.squeeze(dim=0)
# print(x.shape,h.shape)