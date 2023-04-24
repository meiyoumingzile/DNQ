def chooseActFromNet(self, roundId, env, maxAgent, preActQue, allActionList: list):  # ansUp代表较大的动作，ansDown代表较小的动作
    # return self.chooseActFromNet_withDiv(roundId,env,maxAgent,preActQue,allActionList)
    baseFea = getBaseFea(env, self.player.id, preActQue).cuda(self.worker.cudaID)  # self.id
    allAct_kind = env.classActKind(allActionList)

    # print(self.worker.cudaID,next(self.net.parameters()).device)
    hisActFea = self.hisActTensor
    probList1 = self.net.forward_fp(baseFea, hisActFea).squeeze(dim=0)  # base的特征向量
    for i in range(len(allAct_kind)):
        if len(allAct_kind[i]) == 0:
            probList1[i] = -math.inf
    mask = (probList1 != -math.inf)
    probList1 = torch.softmax(probList1, dim=0)

    actid_1, probList1 = self.epslBoltzmann(probList1, mask, self.worker.actepsl)
    actid_1_prob = probList1[actid_1].item()

    if len(allAct_kind[actid_1]) <= 1:
        actid_2 = 0
        actid_2_prob = 1
    else:
        allActFea = getAllActionFea(self.env, allAct_kind[actid_1]).cuda(self.worker.cudaID)
        probList2 = self.net.forward_act(baseFea, hisActFea, allActFea).squeeze(dim=1)
        actid_2, probList2 = self.epslBoltzmann(probList2, None, self.worker.actepsl)
        actid_2_prob = probList2[actid_2].item()
    act = allAct_kind[actid_1][actid_2]
    self.pushMemory(roundId, mask, allAct_kind[actid_1], baseFea, hisActFea, 0, (actid_1, actid_1_prob),
                    (actid_2, actid_2_prob))
    self.player.useAction(act, self.updateINF)
    return act