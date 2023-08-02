import argparse
import os
import sys
import numpy as np
from graphDou.graphDou_ppo import DppoWorkers
def loadnpy(args,path):
    if os.path.exists(path):
        args.deckDataNpy = np.load(path)
        args.deckDataNpy_len=args.deckDataNpy.shape[0]
    else:
        args.deckDataNpy=None
        args.deckDataNpy_len=0
        print("没有数据集文件")
parser = argparse.ArgumentParser()
parser.add_argument('--isInitPar', type=bool, default=False)
parser.add_argument('--cudaIDList', type=list, default=[1])
parser.add_argument('--value_lr', type=float, default=0.0003)
parser.add_argument('--policy_lr', type=float, default=0.0003)
parser.add_argument('--entropy_coef', type=float, default=0.001)
parser.add_argument('--learn_cstep', type=int, default=300)
parser.add_argument('--learn_astep', type=int, default=1000)
parser.add_argument('--collection_length', type=int, default=1)
parser.add_argument('--dataUseCnt', type=int, default=1)
parser.add_argument('--istrain', type=list, default=[1,0])#1代表训练
parser.add_argument('--policykind', type=list, default=[11,-2])
parser.add_argument('--shape', type=tuple, default=(1,350))#初始时训练那个人
parser.add_argument('--gama', type=float, default=1)
parser.add_argument('--lanta', type=float, default=0.95)
parser.add_argument('--rewardFun', type=str, default="setReward_Sparse")#setReward_Sparse
parser.add_argument('--picName', type=str, default="ppogae")
parser.add_argument('--epoch_size', type=int, default=102)
parser.add_argument('--pltDataLen', type=int, default=200)
parser.add_argument('--scthreshold', type=float, default=0.5)
parser.add_argument('--seeType', type=int, default=0)#1是能看见
parser.add_argument('--partialRewards', type=float, default=0.01)#1是能看见
args = parser.parse_args()
loadnpy(args,'eval_data.npy')#eval_data.npy
print("cuda:"+str(args.cudaIDList[0]))
worker=DppoWorkers(0,args)
worker.train_agent("")#2023-04-13-23-44，2023-04-13-18-47

#istrain代表是否训练
'''
policykind是智能体策略：-2是douzero策略，-1是我的贪心推理策略，1代表我的网络策略,2是贪心策略，3是随机策略
11是没有动作划分的网络策略
'''

