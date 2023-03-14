import torch
import argparse
import torch.multiprocessing as mp
import snake_agent
from snake_agent import dppo_workers
import snake_utils as utils
import snake_network
import snake_game
from snake_game import CC
from snake_chief import chief_worker
import os

# start the main function...
parser = argparse.ArgumentParser()
parser.add_argument('--isInitPar', type=bool, default=False)
parser.add_argument('--cudaID', type=int, default=1)
parser.add_argument('--value_lr', type=float, default=0.0001)
parser.add_argument('--policy_lr', type=float, default=0.0001)
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
    num_actions = snake_game.ACTION_SPACE_MAX
    # define the global network...
    critic_shared_model = snake_network.NET_CRITIC(num_inputs, 1).cuda(args.cudaID)
    actor_shared_model = snake_network.NET_ACTOR(num_inputs, num_actions).cuda(args.cudaID)
    if args.isInitPar:
        critic_shared_model = torch.load("mod/critic_net.pt")  #
        actor_shared_model = torch.load("mod/actor_net.pt")  #
    # critic_shared_model.train()
    critic_shared_model.share_memory()#共线内存，意味着把critic_shared_model赋值给别人时，那个值就是自己的值
    # print(critic_shared_model.gard)
    # actor_shared_model.train()
    actor_shared_model.share_memory()


    # define the traffic signal...
    traffic_signal = utils.TrafficLight()
    # define the counter
    critic_counter = utils.Counter()
    actor_counter = utils.Counter()
    # define the shared gradient buffer...
    critic_shared_grad_buffer = utils.Shared_grad_buffers(critic_shared_model)
    actor_shared_grad_buffer = utils.Shared_grad_buffers(actor_shared_model)
    # define shared observation state...
    shared_obs_state = utils.Running_mean_filter(num_inputs)
    # define shared reward...
    shared_reward = utils.RewardCounter()
    # define the optimizer...
    critic_optimizer = torch.optim.Adam(critic_shared_model.parameters(), lr=args.value_lr)
    actor_optimizer = torch.optim.Adam(actor_shared_model.parameters(), lr=args.policy_lr)

    # find how many processor is available...
    # num_of_workers = mp.cpu_count() - 1
    num_of_workers = 10
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


