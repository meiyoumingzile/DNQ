
import torch.nn as nn #

class NET_CRITIC(nn.Module):#评论家网络
    def __init__(self,in_fea,out_fea):
        super().__init__()
        # in_fea=490
        hdim=128
        # self.lstm=nn.LSTM(3,64,2)
        self.fc =nn.Sequential(
            nn.Linear(in_features=in_fea, out_features=1024,bias=True),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=hdim,bias=True),
            nn.Tanh(),
            nn.Linear(in_features=hdim, out_features=out_fea,bias=True)
        )
        for con in self.fc:
            if con=="<class 'torch.nn.modules.linear.Linear'>":
                nn.init.normal_(con.weight, std=0.01)
                nn.init.constant_(con.bias, 0.1)
    def forward(self,x):#重写前向传播算法，x是张量
        x = x.view(x.size(0), -1)
        x=self.fc(x)
        return x

class NET_ACTOR(nn.Module):#演员网络
    def __init__(self,in_fea,out_fea):
        super().__init__()
        # in_fea=490
        hdim=128
        # self.lstm=nn.LSTM(3,64,2)
        self.fc =nn.Sequential(
            nn.Linear(in_features=in_fea, out_features=1024,bias=True),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=hdim,bias=True),
            nn.Tanh(),
            nn.Linear(in_features=hdim, out_features=out_fea,bias=True)
        )
        for con in self.fc:
            if con=="<class 'torch.nn.modules.linear.Linear'>":
                nn.init.normal_(con.weight, std=0.01)
                nn.init.constant_(con.bias, 0.1)
    def forward(self,x):#重写前向传播算法，x是张量
        x = x.view(x.size(0), -1)
        x=self.fc(x)
        return x