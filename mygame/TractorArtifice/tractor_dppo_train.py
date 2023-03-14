import torch
import argparse
import torch.multiprocessing as mp
import tractor_network
from tractor_dppo_agent import dppo_workers
import tractor_utils as utils
import tractor_game
from tractor_game import CC
from tractor_dppo_chief import chief_worker
import os

# start the main function...
parser = argparse.ArgumentParser()
parser.add_argument('--isInitPar', type=bool, default=False)
parser.add_argument('--cudaID', type=int, default=0)
parser.add_argument('--value_lr', type=float, default=0.00001)
parser.add_argument('--policy_lr', type=float, default=0.00001)
parser.add_argument('--learn_step', type=int, default=50)
parser.add_argument('--collection_length', type=int, default=1)
parser.add_argument('--value_step', type=int, default=5)
parser.add_argument('--policy_step', type=int, default=5)
parser.add_argument('--env_name', type=str, default="snake_v0")

if __name__ == '__main__':
    # get the arguments...
    # print(mp.get_start_method())
    # if mp.get_start_method()=="fork":
    #     mp.set_start_method('spawn')
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    args = parser.parse_args()
    # build up the environment and extract some informations...
    env =CC()
    num_inputs = 490#自己写的
    num_actions = 100#自己填
    # define the global network...
    shared_player_model =[tractor_network.AgentNet(num_inputs).cuda(args.cudaID) for i in range(4)]
    if args.isInitPar:
        for i in range(4):
            shared_player_model[i] = torch.load("mod/agent_net"+str(i)+".pt")  #
    shared_player_model.train()
    shared_player_model.share_memory()#共线内存，意味着把critic_shared_model赋值给别人时，那个值就是自己的值


    # define the traffic signal...
    traffic_signal = utils.TrafficLight()
    # define the counter
    critic_counter = utils.Counter()
    actor_counter = utils.Counter()
    # define the shared gradient buffer...
    shared_grad_buffer = utils.Shared_grad_buffers(shared_player_model)
    # define shared observation state...
    shared_obs_state = utils.Running_mean_filter(num_inputs)
    # define shared reward...
    shared_reward = utils.RewardCounter()
    # define the optimizer...
    critic_optimizer = torch.optim.Adam(critic_shared_model.parameters(), lr=args.value_lr)
    actor_optimizer = torch.optim.Adam(actor_shared_model.parameters(), lr=args.policy_lr)

    # find how many processor is available...
    # num_of_workers = mp.cpu_count() - 1
    num_of_workers = 2
    processor = []
    workers = []
    print("cpu进程数",num_of_workers)
    p = mp.Process(target=chief_worker, args=(args,num_of_workers, traffic_signal, critic_counter, actor_counter,
        critic_shared_model, actor_shared_model, critic_shared_grad_buffer, actor_shared_grad_buffer,
        critic_optimizer, actor_optimizer, shared_reward, shared_obs_state, args.policy_step, args.env_name))
    processor.append(p)
    for idx in range(num_of_workers):
        workers.append(dppo_workers(idx,shared_obs_state,args))
    for worker in workers:
        p = mp.Process(target=worker.train_network, args=(traffic_signal, critic_counter, actor_counter,
            critic_shared_model, actor_shared_model, shared_obs_state, critic_shared_grad_buffer, actor_shared_grad_buffer, shared_reward))
        processor.append(p)
    for p in processor:
        p.start()
    for p in processor:
        p.join()#阻塞


