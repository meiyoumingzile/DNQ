import time

import torch
from torch import nn


def epslBoltzmann(self, probList, mask, epsl=0.2):  # 修正玻尔兹曼,这里epsl是随机选取的概率
    # print(torch.isnan(probList))
    if torch.any(torch.isnan(probList)):
        print(probList, mask)
        print("have nan")
        exit()
    if mask != None:
        cnt = mask.sum().item()
        p = epsl / cnt
        ans = probList * (1 - epsl) + p
        for i in range(probList.shape[0]):
            if mask[i] == 0:
                ans[i] = 0
    else:
        cnt = probList.shape[0]
        # if cnt<1:
        #     print(probList,mask,probList.shape)
        p = epsl / cnt
        ans = probList * (1 - epsl) + p
    actid = torch.multinomial(ans, num_samples=1, replacement=True)[0].item()
    return actid, ans
class shuffleNet(nn.Module):#
    def __init__(self):#330
        super().__init__()
        self.tran =  nn.Transformer(
            d_model=57,
            nhead=3,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=57,
            batch_first=True,
            dropout=0.1)
        self.lstm1 = nn.LSTM(input_size=57, hidden_size=128, num_layers=1, batch_first=True)
    def forward(self,x):
        start_time0 = time.time()
        z=self.tran(x,x)
        start_time1 = time.time()
        y,(h0,c0)=self.lstm1(x)
        start_time2 = time.time()
        print(start_time2-start_time1,start_time1-start_time0)
        return z

net=shuffleNet()
x=torch.ones((1,111,57))
net.forward(x)
print(x.shape)