# import math
import math
import random
import sys
import time
import platform
import torchvision
import cv2
# import gym
import os
from collections import defaultdict
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.optim as optim
import torch.nn.functional as F#激活函数
from torchvision import transforms#图像处理的工具类
from torchvision import datasets#数据集
from torch.utils.data import DataLoader#数据集加载
import ctypes
# import commands
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# li=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# act=torch.tensor([[0],[2],[1]])
# print(li)
# print(li.gather(1, act))
# probs=torch.tensor([[0.1,0.1,0.8],[0.2,0.1,0.7],[0.2,0.2,0.6]])
# for i in range(len(probs)):
#     ran=np.random.uniform(0, 1)
#     j=0
#     while(ran>0):
#         ran-=probs[i][j]
#         j+=1
#     probs[i]=torch.tensor([j])
# print(probs)

import pynvml
import time
# import ygo
pynvml.nvmlInit()
# incoord = lambda m,n: m in range(0,16) and n in range(0,16)
def sig(x):
    return 1 / (1 + math.exp(-x + 1))
# print(sig(-3.8))
# a=np.array(math.exp(-math.inf))
# print( np.expand_dims(a,axis=0))
# arr=np.random.choice(10, 5)
# print(arr)
def script_path():
    import inspect, os
    caller_file = inspect.stack()[1][1]         # caller's filename
    print(os.path.dirname(caller_file))
    return os.path.abspath(os.path.dirname(caller_file))# path

print(script_path())
def printNvidiaGPU(gpu_id):
    # get GPU temperature
    gpu_device = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

    temperature = pynvml.nvmlDeviceGetTemperature(gpu_device, pynvml.NVML_TEMPERATURE_GPU)
    # get GPU memory total
    totalMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).total
    # get GPU memory used
    usedMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).used

    performance = pynvml.nvmlDeviceGetPerformanceState(gpu_device)

    powerUsage = pynvml.nvmlDeviceGetPowerUsage(gpu_device)
    powerState = pynvml.nvmlDeviceGetPowerState(gpu_device)
    FanSpeed = pynvml.nvmlDeviceGetFanSpeed(gpu_device)
    PersistenceMode = pynvml.nvmlDeviceGetPersistenceMode(gpu_device)
    UtilizationRates = pynvml.nvmlDeviceGetUtilizationRates(gpu_device)

    print("MemoryInfo：{0}M/{1}M，使用率：{2}%".format("%.1f" % (usedMemory / 1024 / 1024),
                                                 "%.1f" % (totalMemory / 1024 / 1024),
                                                 "%.1f" % (usedMemory / totalMemory * 100)))
    print("Temperature：{0}摄氏度".format(temperature))
    print("Performance：{0}".format(performance))
    print("PowerState: {0}".format(powerState))
    print("PowerUsage: {0}".format(powerUsage / 1000))
    print("FanSpeed: {0}".format(FanSpeed))
    print("PersistenceMode: {0}".format(PersistenceMode))
    print("UtilizationRates: {0}".format(UtilizationRates.gpu))
    time.sleep(1)
list_numpy=np.array([2,3,5,4,3,2,1])
print(np.where(list_numpy == 3))

# pic=np.array([[11,22,55],[33,44,66]])
# a=np.array([[1,2,3]])
# pic=np.concatenate((pic, a), axis=0)
# mat=np.expand_dims(mat, axis=0)
# print(pic)

# import matplotlib.pyplot as plt
#
# x = [i for i in range(100) ]
# y_1 = [i * 2 for i in x]
# y_2 = [i * 4 for i in x]
#
# print(x)
# print(y_1)
# print(y_2)
#
# # 创建画布
# plt.figure()
#
# '''绘制第一条数据线
# 1、节点为圆圈
# 2、线颜色为红色
# 3、标签名字为y1-data
# '''
# plt.plot(x, y_1, marker='o', color='r', label='y1-data')
#
# '''绘制第二条数据线
# 1、节点为五角星
# 2、线颜色为蓝色
# 3、标签名字为y2-data
# '''
# plt.plot(x, y_2, marker='*', color='b', label='y2-data')
#
# # 显示图例（使绘制生效）
# plt.legend()
#
# # 横坐标名称
# plt.xlabel('x_label')
#
# # 纵坐标名称
# plt.ylabel('y_label')
#
# # 保存图片到本地
# plt.savefig('pci.png')
#
# # 显示图片
# # plt.show()






# while (1):
#     gpu_device1 = pynvml.nvmlDeviceGetHandleByIndex(0)
#     temperature1 = pynvml.nvmlDeviceGetTemperature(gpu_device1, pynvml.NVML_TEMPERATURE_GPU)
#     gpu_device2 = pynvml.nvmlDeviceGetHandleByIndex(1)
#     temperature2 = pynvml.nvmlDeviceGetTemperature(gpu_device2, pynvml.NVML_TEMPERATURE_GPU)
#     if temperature1 >50 or temperature2 > 80:
#         print(str(temperature1)+"  "+str(temperature2)+" exit!!!")
#         sys.exit(0)


