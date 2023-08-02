import copy
import math

import torch
import torch.nn.functional as F
import numpy as np

import torch
from torch import nn
# from torch_geometric.nn import GCNConv

from graphDou.graphDou_codeParameter import INFEA, mainfea, lstmfea, lstmInfea, GATINFEA


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
        # # self.lstm2 = nn.LSTM(input_size=lstmInfea, hidden_size=lstmfea, num_layers=1, batch_first=True)
        self.bn = nn.LayerNorm(lstmfea)
        self.conv=nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5,stride=2),
        )
        self.mlp_base = nn.Sequential(  # 求动作
            nn.LayerNorm(in_fea+lstmfea),
            nn.Linear(in_fea+lstmfea, mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.Tanh(),
            nn.Linear(mainfea, mainfea),
            nn.LayerNorm(mainfea),
            nn.Tanh(),
        )
        self.fc=nn.Linear(mainfea, 1)

        par_init(self.fc)
        par_init(self.mlp_base)
    def forward_g(self,x):
        return self.conv(x)

    def forward_base(self,x,graphFea,hisActFea):  # 估计q，x是forward_base的结果
        x1, (hc1, hn) = self.lstm1(hisActFea)
        # x1=self.bn(x1)
        x1 = x1[:, -1, :]
        # now=self.forward_g(graphFea.cuda(cudaId))
        graphFea = self.conv(graphFea.unsqueeze(dim=0))
        now=graphFea.view(graphFea.size(0), -1)
        # print(x.shape, now.shape)
        x = torch.cat((x,x1,now),dim=1)
        x = self.mlp_base(x)
        return x
    def forward(self, x,graphFea,hisActFea,cudaId):  # 估计q，x是forward_base的结果
        x = self.forward_base(x.cuda(cudaId),graphFea.cuda(cudaId),hisActFea.cuda(cudaId))
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

class AgentNet(nn.Module):#
    def __init__(self,in_fea):#330
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=lstmInfea, hidden_size=lstmfea, num_layers=1, batch_first=True)
        self.bn=nn.LayerNorm(lstmfea)
        self.conv=nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5,stride=2),
        )
        self.mlp_act2 = nn.Sequential(  # 求动作平分,是actor网络
            nn.LayerNorm(in_fea + lstmfea+18),
            nn.Linear(in_fea + lstmfea+18, mainfea),
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
            # nn.Sigmoid(),
        )
        par_init(self.mlp_act2)
    def forward_g(self,x):
        return self.conv(x)

    def forward_act(self, x,graphFea,hisActFea,actFea,cudaId):  # actor网络，baseFea和hisActFea都是1*infea的大小
        batch_size = actFea.shape[0]
        graphFea=self.conv(graphFea.unsqueeze(dim=0).cuda(cudaId))
        now = graphFea.view(graphFea.size(0), -1).cuda(cudaId)
        x1, (hc1, hn) = self.lstm1(hisActFea.cuda(cudaId))
        # x1=self.bn(x1)
        x1 = x1[:, -1, :]
        # print(x.shape,now.shape)
        x = torch.cat((now,x.cuda(cudaId),x1),dim=1)
        # print(x.shape)
        x = x.repeat(batch_size, 1)
        # print(x.shape,actFea.shape)
        x = torch.cat((x,actFea.cuda(cudaId)), dim=1)
        x = self.mlp_act2(x)
        x = F.softmax(x, dim=0)
        return x

    def printGrad(self):
        print("actorNetGrad:")
        i = 0
        for m in self.mlp_act2.modules():
            if isinstance(m, nn.Linear):
                print("mlp_act2:" + str(i), end=" ")
                i += 1
                print(m.weight.grad)

net=AgentNet(21)
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