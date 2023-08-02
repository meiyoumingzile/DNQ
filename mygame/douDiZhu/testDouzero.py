import math
from collections import deque

import torch

from baseline import douzero_encoder
from otherPolicyBaseline import getModel
from DNQ.mygame.douDiZhu.doudizhu_encoder import getBaseFea, getAllActionFea
from doudizhu_game import Doudizhu, dfsPrintActList, Action, baselinePolicy, baselinePolicy_first, INF, \
    printDeckListForId
from doudizhu_cheat import mkDeck, cheat1, setHandCards
import random

pid=0
env=Doudizhu()
env.dealCards(None,0)
setHandCards(env.players[pid],["10","10","10","J","J","J","A","A"])

obs = douzero_encoder.toObs(playerTag, encoderPyshell.Infoset(env, self.player.id, allActionList))
env.players[pid].getHandList()
# print(env.players[pid].getDouCnt(False))
act=Action([6,6,6,7,7,7],[2,2,4,4])
ans = env.players[pid].getAllAction(act)