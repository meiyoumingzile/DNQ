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

class AgentCriNet(nn.Module):#
    def __init__(self,in_fea):#330
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=lstmInfea, hidden_size=lstmfea, num_layers=1, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size=lstmInfea, hidden_size=lstmfea, num_layers=1, batch_first=True)
        self.bn = nn.LayerNorm(lstmfea)
        self.mlp_base = nn.Sequential(  # 求动作
            nn.LayerNorm(in_fea+lstmfea),
            nn.Linear(in_fea+lstmfea, mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.Tanh(),
        )
        self.fc=nn.Linear(mainfea, 1)
        # par_init(self.lstm1)
        par_init(self.fc)
        par_init(self.mlp_base)

    def forward_base(self,baseFea,hisGameActi,hisActFea):  # 估计q，x是forward_base的结果
        # x1, (hc1, hn) = self.lstm1(hisGameActi)
        # # x1=self.bn(x1)
        # x1=x1[:,-1,:]
        x2, (hc1, hn) = self.lstm1(hisActFea)
        # x1=self.bn(x1)
        x2=x2[:,-1,:]
        # print(hc.shape,baseFea.shape)
        x = torch.cat((baseFea,x2), dim=1)
        x = self.mlp_base(x)
        return x
    def forward(self, baseFea,hisGameActi,hisActFea):  # 估计q，x是forward_base的结果
        x = self.forward_base(baseFea,hisGameActi,hisActFea)
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
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.Tanh(),
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
        self.bn=nn.LayerNorm(lstmfea)
        self.mlp_base = nn.Sequential()

        self.mlp_act2 = nn.Sequential(  # 求动作平分,是actor网络
            nn.LayerNorm(in_fea + lstmfea+57+14),
            nn.Linear(in_fea + lstmfea+57+14, mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, 1),#512是base网络的输出，128是单张牌之间lstm的输出
        )
        # def print_grad(grad):
        #     print("lstm梯度：")
        #     print(grad)
        #
        # for p in self.lstm1.parameters():
        #     p.register_hook(print_grad)
        # for p in self.lstm2.parameters():
        #     p.register_hook(print_grad)
        # par_init(self.lstm1)
        # par_init(self.lstm2)
        par_init(self.mlp_base)
        par_init(self.mlp_act2)
    def forward_base(self, baseFea,hisActFea):

        x1, (hc1, hn) = self.lstm1(hisActFea)
        x1=x1[:,-1,:]

        x = torch.cat((baseFea, x1), dim=1)
        x = self.mlp_base(x)
        # print(x.shape)
        return x

    def forward_act(self, baseFea,hisActFea1,hisActFea2,actFea):  # actor网络，baseFea和hisActFea都是1*infea的大小
        # _,(act,c)=self.lstm2(act)
        # act = act.squeeze(dim=0)
        # print(x)
        batch_size=actFea.shape[0]
        baseFea=baseFea.repeat(batch_size, 1)

        # x1, (hc1, hn) = self.lstm1(hisActFea1)  # 上一局的动作
        # # x1 = self.bn(x1)
        # x1 = x1[:, -1, :]
        # x1 = x1.repeat(batch_size, 1)
        x2, (hc1, hn) = self.lstm2(hisActFea2)
        # x2 = self.bn(x2)
        x2 = x2[:, -1, :]
        x2 = x2.repeat(batch_size, 1)
        # print(x1.shape,x2.shape)
        # print(baseFea.shape,x2.shape,actFea.shape)
        x = torch.cat((baseFea,x2,actFea), dim=1)
        # print(x.shape)
        # print(self.mlp_act2)
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
            nn.Tanh(),
            nn.Linear(mainfea, 4),
        )
    def forward(self,x):
        x=self.mlp(x)

        return x
class shuffleNet(nn.Module):#
    def __init__(self):#330
        super().__init__()
        self.mlp_critic = nn.Sequential(  #叫分网络,它连接critic网络
            nn.Linear(54+1, 512),
            # nn.LayerNorm(mainfea),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.mlp_actor = nn.Sequential(  #叫分网络,它连接critic网络
            nn.Linear(54+1, 512),
            # nn.LayerNorm(mainfea),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 54),
        )
    def forward_val(self,x):
        x=self.mlp_critic(x)
        return x
    def forward_act(self,x):
        x=self.mlp_actor(x)
        x=F.softmax(x, dim=1)
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
# x=torch.Tensor([10000,100000,100000,0,-1000])
# x=F.softmax(x, dim=0)
# x=torch.zeros((2,55))
# x=torch.cat((torch.zeros(((2,55))),x),dim=0)
# print(x)