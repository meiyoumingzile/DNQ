import copy
import math

import torch
import torch.nn.functional as F
import numpy as np

import torch
from torch import nn
from doudizhu_codeParameter import INFEA,mainfea,lstmfea,lstmInfea
def par_init(layer, gain=1.0):
    with torch.no_grad():
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
            elif isinstance(m, nn.LSTM):
                nn.init.orthogonal_(m.weight_ih_l0)
                nn.init.orthogonal_(m.weight_hh_l0)

class ResBlock(nn.Module):#跳2层
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        # self.fc1=nn.Linear(out_channels, out_channels)
        par_init(self.fc)
        # par_init(self.fc1)
    def forward(self, x):
        identity = x
        out = self.fc(x)
        out += identity
        return out
class AgentCriNet(nn.Module):#
    def __init__(self,in_fea):#330
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=lstmInfea, hidden_size=lstmfea,num_layers=1,  batch_first=True)#本LSTM接受出牌的动作序列，可能只有1，也可能很多。

        self.mlp_base = nn.Sequential(  # 求动作
            nn.Linear(in_fea+lstmfea, mainfea),
            # nn.LayerNorm(mainfea),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.ReLU(),
        )
        self.fc=nn.Linear(mainfea, 1)
        # par_init(self.lstm1)
        par_init(self.fc)
        par_init(self.mlp_base)

    def forward_base(self, baseFea,hisActFea):  # 估计q，x是forward_base的结果
        x1, (hc1, hn) = self.lstm1(hisActFea)
        x1=x1[:,-1,:]
        # print(hc.shape,baseFea.shape)
        x = torch.cat((baseFea, x1), dim=1)
        x = self.mlp_base(x)
        return x
    def forward(self, baseFea,hisActFea):  # 估计q，x是forward_base的结果
        x = self.forward_base(baseFea,hisActFea)
        x=self.fc(x)
        return x
    def printGrad(self):
        print("criticNetGrad:")
        i=0
        for m in self.mlp_base.modules():
            if isinstance(m, nn.Linear):
                print("mlp_base:" + str(i), end=" ")
                i += 1
                print(m.weight.grad)
class PredictionNet(nn.Module):#预测牌的网络
    def __init__(self,in_fea):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=lstmInfea, hidden_size=lstmfea, num_layers=1, batch_first=True)
        self.mlp= nn.Sequential(#一个监督学习网络，预测下一个人手里有某张牌的概率
            nn.Linear(in_fea, mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, 54),
            nn.Sigmoid()
        )
        par_init(self.mlp)
    def forward(self,baseFea,hisActFea):#估计q，x是forward_base的结果
        x1, (hc1, hn) = self.lstm1(hisActFea)
        x1 = x1[:, -1, :]
        x = torch.cat((baseFea, x1), dim=1)
        x = self.mlp(x)
        return x
class AgentNet(nn.Module):#
    def __init__(self,in_fea):#330
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=lstmInfea, hidden_size=lstmfea, num_layers=1,batch_first=True)
        self.lstm2 = nn.LSTM(input_size=lstmInfea, hidden_size=lstmfea, num_layers=1, batch_first=True)
        self.preNet=PredictionNet(in_fea)
        self.mlp_base = nn.Sequential()
        self.mlp_act1 = nn.Sequential(  # 求出哪类，是actor网络
            nn.Linear(in_fea + lstmfea, mainfea),
            # nn.LayerNorm(mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, 13),
        )
        self.mlp_act2 = nn.Sequential(  # 求动作平分,是actor网络
            nn.Linear(in_fea + lstmfea+57+14, mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, 1),#512是base网络的输出，128是单张牌之间lstm的输出
            # nn.Sigmoid(),
        )
        # par_init(self.lstm1)
        # par_init(self.lstm2)
        par_init(self.mlp_base)
        par_init(self.mlp_act1)
        par_init(self.mlp_act2)
    def forward_base(self, baseFea,hisActFea):

        x1, (hc1, hn) = self.lstm1(hisActFea)
        x1=x1[:,-1,:]

        x = torch.cat((baseFea, x1), dim=1)
        x = self.mlp_base(x)
        # print(x.shape)
        return x
    def forward_fp(self,baseFea,hisActFea):#二分类,x是forward_base的结果
        # x = self.forward_base(baseFea,hisActFea)
        # print(baseFea.shape)
        x1, (hc1, hn) = self.lstm1(hisActFea)
        x1 = x1[:, -1, :]
        x = torch.cat((baseFea, x1), dim=1)
        # x = self.mlp_base(x)
        x=self.mlp_act1(x)
        return x

    def forward_act(self, baseFea,hisActFea,actFea):  # actor网络，baseFea和hisActFea都是1*infea的大小
        # _,(act,c)=self.lstm2(act)
        # act = act.squeeze(dim=0)
        # print(x)

        batch_size=actFea.shape[0]
        baseFea=baseFea.repeat(batch_size, 1)
        hisActFea = hisActFea.repeat(batch_size, 1,1)
        x1, (hc1, hn) = self.lstm2(hisActFea)
        x1 = x1[:, -1, :]
        x = torch.cat((baseFea, x1,actFea), dim=1)
        x = self.mlp_act2(x)
        # print(x)
        x=F.softmax(x, dim=0)
        return x
    def forward_pre(self, baseFea,hisActFea):
        return self.preNet.forward(baseFea,hisActFea)
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
class BidNet(nn.Module):#
    def __init__(self):#330
        super().__init__()
        self.mlp = nn.Sequential(  #叫分网络,它连接critic网络
            nn.Linear(mainfea, mainfea),
            # nn.LayerNorm(mainfea),
            nn.ReLU(),
            nn.Linear(mainfea, 4),
        )
    def forward(self,x):
        x=self.mlp(x)

        return x
# net=AgentNet()
# lstm=nn.LSTM(input_size=57, hidden_size=lstmfea, num_layers=1, batch_first=True)
# x=torch.ones((0,57))
# x,_,_=lstm(torch.Tensor([[]]))·
# print(x.shape)
# print(x,x.shape)
# actid=torch.Tensor([[2]]).long()
# print(x.gather(0,actid))
# x=torch.Tensor([5,0,3,3,3])
# # x=maxMinNor(x)
# # print(x)
# x=F.softmax(x, dim=0)
# x=torch.zeros((2,55))
# x=torch.cat((torch.zeros(((2,55))),x),dim=0)
# print(x.shape)