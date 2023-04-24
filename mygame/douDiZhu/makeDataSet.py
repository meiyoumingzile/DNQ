import argparse
import pickle
import numpy as np
from doudizhu_game import Player,getNum,Card,getDecor,Action,Doudizhu

parser = argparse.ArgumentParser(description='random data generator')
parser.add_argument('--output', default='eval_data', type=str)
parser.add_argument('--num_games', default=10000, type=int)
args = parser.parse_args()

env=Doudizhu()
def saveData():
    li=[]
    for i in range(args.num_games):
        env.reset()
        env.dealCards(None, 0)
        li.append(env.deck)
    ans=np.array(li)
    np.save(args.output, ans)
    print("数组占用的MB数：", ans.nbytes/(1024*1024))
saveData()

