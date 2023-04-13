import datetime
import os
from collections import deque

import matplotlib
import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# This is from the https://github.com/alexis-jacq/Pytorch-DPPO/blob/master/utils.py#L9

# this is to make sure if the workers could pass gradient to the chief...
def initInfo(s,f="info.txt"):
    if os.path.exists(f):
        os.remove(f)
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")
def wInfo(s,f="info.txt"):
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")

class MACD:
    def __init__(self,k,maxLen=10000):
        self.que=deque()
        self.sumQue=deque(maxlen=maxLen)
        self.sum=0
        self.k=k
    def add(self,a):
        self.sum += a
        self.que.append(a)
        if len(self.que) >self.k:
            self.sum-=self.que.popleft()
        self.sumQue.append(self.sum/self.k)
    def getAvg(self):
        return self.sum / self.k
class TrafficLight:
    def __init__(self):
        self.val = mp.Value("b", False)
        self.lock = mp.Lock()

    def get(self):
        with self.lock:
            return self.val.value


    def switch(self):
        with self.lock:
            self.val.value = (not self.val.value)

# this is used to decide when the chief could update the network...
class Counter:
    def __init__(self):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()
        self.up= mp.Value("i", 0)
    def getUp(self):
        with self.lock:
            return self.up.value
    def addUp(self):
        with self.lock:
            self.up.value+=1
    def subUp(self):
        with self.lock:
            self.up.value-=1
    def resetUp(self):
        with self.lock:
            self.up.value =0
    def get(self):
        with self.lock:
            return self.val.value

    def increment(self):
        with self.lock:
            self.val.value += 1

    def reset(self):
        with self.lock:
            self.val.value =0

# this is used to record the reward each worker achieved...
class RewardCounter:
    def __init__(self):
        self.val = mp.Value('f', 0)
        self.lock = mp.Lock()
    def add(self, reward):
        with self.lock:
            self.val.value += reward
    def get(self):
        with self.lock:
            return self.val.value
    def reset(self):
        with self.lock:
            self.val.value = 0

# this is used to accumulate the gradients
class Shared_grad_buffers:
    def __init__(self, models):
        self.lock = mp.Lock()
        self.grads = {}
        for name, p in models.named_parameters():
            self.grads[name] = torch.zeros(p.size()).share_memory_()

    def add_gradient(self, models):
        with self.lock:
            for name, p in models.named_parameters():
                if p.grad!=None:
                    self.grads[name] += p.grad.data.cpu()

    def add_gradient_tar(self, models,tarname):
        with self.lock:
            n=len(tarname)
            for name, p in models.named_parameters():
                # print(name)
                if p.grad != None and name[0:n]==tarname:
                    # print(name)
                    self.grads[name] += p.grad.data.cpu()

    def reset(self):
        with self.lock:
            for name, grad in self.grads.items():
                self.grads[name].fill_(0)


# running mean filter, used to normalize the state of mujoco environment
class Running_mean_filter:
    def __init__(self, num_inputs):
        self.lock = mp.Lock()
        self.n = torch.zeros(num_inputs).share_memory_()
        self.mean = torch.zeros(num_inputs).share_memory_()
        self.s = torch.zeros(num_inputs).share_memory_()
        self.var = torch.zeros(num_inputs).share_memory_()
    # start to normalize the states...
    def normalize(self, x):
        with self.lock:
            obs = x.copy()
            obs = torch.Tensor(obs)
            self.n += 1
            if self.n[0] == 1:
                self.mean[...] = obs
                self.var[...] = self.mean.pow(2)
            else:
                old_mean = self.mean.clone()
                self.mean[...] = old_mean + (obs - old_mean) / self.n
                self.s[...] = self.s + (obs - old_mean) * (obs - self.mean)
                self.var[...] = self.s / (self.n - 1)
            mean_clip = self.mean.numpy().copy()
            var_clip = self.var.numpy().copy()
            std = np.sqrt(var_clip)
            x = (x - mean_clip) / (std + 1e-8)
            x = np.clip(x, -5.0, 5.0)
            return x
    # start to get the results...
    def get_results(self):
        with self.lock:
            var_clip = self.var.numpy().copy()
            return (self.mean.numpy().copy(), np.sqrt(var_clip))
class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape)
        self.S = torch.zeros(shape)
        self.std = torch.sqrt(self.S)

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n )


def binary_conversion(var: int):
    """
    二进制单位转换
    :param var: 需要计算的变量，bytes值
    :return: 单位转换后的变量，kb 或 mb
    """
    assert isinstance(var, int)
    if var <= 1024:
        return f'占用 {round(var / 1024, 2)} KB内存'
    else:
        return f'占用 {round(var / (1024 ** 2), 2)} MB内存'
def printSize(x):
    memory_size = x.numel() * x.element_size()
    binary_conversion(memory_size)
def getNowTimePath():
    now = datetime.datetime.now()
    now = str(now.strftime("%Y-%m-%d-%H:%M"))
    now = now.replace(":", "-")
    path = "mod/" + now + "/"
    return path

def getMacdList(yList,k=5):
    sum = 0
    n=len(yList)
    macd = [0] * n
    for i in range(0,min(n,k)):
        sum+=yList[i]
    for i in range(k,n):
        macd[i] =sum/k
        sum=sum+yList[i]-yList[i-k]
    return macd[k:]
def drawMacd(xList,yList,k,color):
    if len(yList)>=k:
        ma5=getMacdList(yList,k)
        plt.plot(xList[k:], ma5,color=color,label="ma"+str(k))
def drawBrokenLine(xList,yList,name="aaa",xLabel="",yLabel="",isClear=True):
    plt.figure(figsize=(20, 10), dpi=100)
# game = ['1-G1', '1-G2', '1-G3', '1-G4', '1-G5', '2-G1', '2-G2', '2-G3', '2-G4', '2-G5', '3-G1', '3-G2', '3-G3',
#         '3-G4', '3-G5', '452G1', '4554-G2', '54G3', '54-G4', '54G5', '54-G6']
# scores = [23, 10, 38, 30, 36, 20, 28, 36, 16, 29, 15, 26, 30, 26, 38, 34, 33, 25, 28, 40, 28]
    plt.plot(xList, yList,color='blue',label="score")
    drawMacd(xList,yList,5,"yellow")
    drawMacd(xList, yList, 30, "red")
    plt.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95),
               ncol=3, shadow=True,fancybox=True)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(name+".jpg")
    if isClear:
        plt.clf()

# print(getNowTimePath())