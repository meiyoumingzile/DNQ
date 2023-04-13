import torch
import time
from torch.autograd import Variable


def chief_worker(args,num_of_workers,graBuf_netList:list, graBuf_crinet, shared_netList:list, shared_crinet, counter:list, traffic_signal:list,shared_info,optimizerList):
    num_iteration = 1
    # gloCounter=counter[4]
    # gloTraffic_signal = traffic_signal[4]
    while True:
        # time.sleep(0.5)
        for i in range(4):
            if args.trainlist[i]!=1:
                continue
            # print(counter[i].getBegin())
            if counter[i].get() >= num_of_workers:
                # num_iteration+=1
                shared_crinet.zero_grad()
                for n, p in shared_crinet.named_parameters():
                    p.grad = graBuf_crinet.grads[n].cuda(args.cudaID0)

                # start to update the critic network
                optimizerList[i][0].step()
                # clean the buffer....
                graBuf_crinet.reset()

                shared_netList[i].zero_grad()
                k=i*2
                for n, p in shared_netList[i].named_parameters():#更新act1
                    p.grad = graBuf_netList[k].grads[n].cuda(args.cudaID0)
                optimizerList[i][1].step()
                graBuf_netList[k].reset()


                shared_netList[i].zero_grad()
                k = i * 2+1
                for n, p in shared_netList[i].named_parameters():  # 更新act1
                    p.grad = graBuf_netList[k].grads[n].cuda(args.cudaID0)
                optimizerList[i][2].step()
                graBuf_netList[k].reset()


                # get the reward...
                # if num_iteration % update_step == 0:
                #     reward_batch = shared_reward.get()
                #     reward_batch /= num_workers
                #     shared_reward.reset()
                #     print('The iteration is ' + str(int(num_iteration / update_step)) + ' and the reward mean is ' + str(
                #         reward_batch))
                #
                # if num_iteration % (update_step * 10) == 0:
                #     save_path = 'saved_models/' + name + '/models_' + str(int(num_iteration / update_step)) + '.pt'
                #     torch.save([actor_shared_model.state_dict(), shared_obs_state.get_results()], save_path)
                # num_iteration += 1

                counter[i].reset()
                traffic_signal[i].switch()
