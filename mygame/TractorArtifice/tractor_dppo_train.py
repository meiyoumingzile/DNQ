import torch
import argparse
import torch.multiprocessing as mp
from tractor_network_res import AgentNet, AgentCriNet
from tractor_dppo_agent import DppoWorkers
import tractor_utils as utils
import tractor_game
from tractor_game import CC
from tractor_dppo_chief import chief_worker
import os

# start the main function...
parser = argparse.ArgumentParser()
parser.add_argument('--initMod', type=str, default="base")
parser.add_argument('--cudaIDList', type=list, default=[0,1])
# parser.add_argument('--num_of_workers', type=int, default=8)
parser.add_argument('--pltDataLen', type=int, default=50)
parser.add_argument('--saveDataLen', type=int, default=50)
parser.add_argument('--value_lr', type=float, default=0.00002)
parser.add_argument('--policy_lr', type=float, default=0.00002)
parser.add_argument('--entropy_coef', type=float, default=0.00001)
parser.add_argument('--CLIP_EPSL', type=int, default=0.2)
parser.add_argument('--learn_step', type=int, default=50)
parser.add_argument('--collection_length', type=int, default=1)
parser.add_argument('--dataUseCnt', type=int, default=1)
parser.add_argument('--env_name', type=str, default="tractor")
parser.add_argument('--trainlist', type=list, default=[1,0,1,0])#初始时训练时策略类型
parser.add_argument('--infea', type=int, default=302)#特征维数
parser.add_argument('--gama', type=float, default=1)
parser.add_argument('--lanta', type=float, default=1)
parser.add_argument('--rewardInterval', type=int, default=1)
parser.add_argument('--rewardFun', type=str, default="setReward")#setReward_Sparse
parser.add_argument('--picName', type=str, default="dppo_sp")

args = parser.parse_args()
utils.initInfo("begin!!")
# print("cuda"+str(args.cudaIDList[0]))
# worker=DppoWorkers(0,None,args)
# worker.train_agent("2023-03-28-11-17")#2023-03-24-23-30

#trainlist[i]=2代表固定策略，0代表初始神经网络，1代表训练的神经网络。
#共用critic，但base分开
def initOpt(critic_net,netList,args):
    optimizerList=[]
    for i in range(4):
        baseNet=list(netList[i].mlp_base.parameters())+list(netList[i].lstm1.parameters())+list(netList[i].lstm2.parameters())
        optimizerList.append((torch.optim.Adam(critic_net.parameters(), lr=args.value_lr,eps=1e-5),
                          torch.optim.Adam(list(netList[i].mlp_act1.parameters())+baseNet, lr=args.policy_lr,eps=1e-5),
        torch.optim.Adam(list(netList[i].mlp_act2.parameters())+baseNet,lr=args.policy_lr, eps=1e-5)
        ))
    return optimizerList
if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    # env =CC()
    args.savePath= utils.getNowTimePath()
    num_of_workers = len(args.cudaIDList)
    cudaID=args.cudaIDList[0]#主网络所在位置
    args.cudaID0=cudaID
    # cudaid=0
    # for i in range(len(args.cudaIDList),num_of_workers):#补全
    #     args.cudaIDList.append(cudaid)
    #     cudaid=(cudaid+1)%2
    print("进程数量："+str(num_of_workers))
    print("最大进程数量"+str(mp.cpu_count() - 2))
    if num_of_workers>=mp.cpu_count() - 1:
        print("erro")
        exit()
    # 初始化显卡和进程数

    #接下来是初始化模型
    critic_net= AgentCriNet(args.infea).cuda(cudaID)
    netList =[AgentNet(args.infea).cuda(cudaID) for i in range(4)]
    if args.initMod!=None and len(args.initMod)>0:#加载模型
        critic_net= torch.load("mod/"+args.initMod+"/agentCriticNet.pt").cuda(cudaID)  #
        for i in range(4):
            netList[i] = torch.load("mod/"+args.initMod+"/agentNet"+str(i)+".pt").cuda(cudaID)  #
    critic_net.train()
    critic_net.share_memory()  # 共线内存，意味着把critic_shared_model赋值给别人时，那个值就是自己的值
    for i in range(4):
        netList[i].train()
        netList[i].share_memory()#共线内存，意味着把critic_shared_model赋值给别人时，那个值就是自己的值


    # define the traffic signal...
    traffic_signal = [utils.TrafficLight() for i in range(5)]
    # define the counter
    counter = [utils.Counter() for i in range(5)]
    # define the shared gradient buffer...
    graBuf_crinet=utils.Shared_grad_buffers(critic_net)
    graBuf_netList=[utils.Shared_grad_buffers(netList[i//2]) for i in range(8)]
    # [utils.Counter() for i in range(4)]
    # define shared observation state...
    # shared_obs_state = utils.Running_mean_filter(num_inputs)
    # define shared reward...
    # shared_reward = utils.RewardCounter()
    # define the optimizer...
    shared_info={}
    optimizerList = initOpt(critic_net,netList,args)

    # find how many processor is available...
    # num_of_workers = mp.cpu_count() - 1

    processor = []
    workers = []
    # print("cpu进程数",num_of_workers)
    p = mp.Process(target=chief_worker, args=(args,num_of_workers,graBuf_netList, graBuf_crinet, netList,
                                              critic_net, counter, traffic_signal,shared_info,optimizerList))#待完成

    processor.append(p)
    for i in range(num_of_workers):
        workers.append(DppoWorkers(i,args,graBuf_netList, graBuf_crinet, netList, critic_net, counter, traffic_signal,shared_info))
    for worker in workers:
        p = mp.Process(target=worker.train_agent, args=())
        processor.append(p)
    for p in processor:
        p.start()
    for p in processor:
        p.join()#阻塞


