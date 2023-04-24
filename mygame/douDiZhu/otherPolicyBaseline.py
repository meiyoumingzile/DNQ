import os

import torch
import numpy as np
from torch import nn
import onnx
import onnxruntime as ort
from onnx import helper
import google.protobuf as pb
from google.protobuf import text_format

import baseline.perfectdou_encoder as perfectdou_encoder
import baseline.douzero_encoder as douzero_encoder
import baseline.douzeroresnet_encoder as douzeroresnet_encoder


class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)
def getModel(kind,cudaId):
    models=None
    if kind=="douzero":
        models=[douzero_encoder.load_model("landlord","baseline/douzero/douzero_ADP/landlord.ckpt", cudaId),
                douzero_encoder.load_model("landlord_down","baseline/douzero/douzero_ADP/landlord_down.ckpt", cudaId),
                douzero_encoder.load_model("landlord_up","baseline/douzero/douzero_ADP/landlord_up.ckpt", cudaId),
        ]
    elif kind == "douzeroresnet":
        models = [douzeroresnet_encoder.load_model("landlord", "baseline/douzero/douzero_resnet/resnet_landlord.ckpt", cudaId),
                  douzeroresnet_encoder.load_model("landlord_down", "baseline/douzero/douzero_resnet/resnet_landlord_down.ckpt",
                                              cudaId),
                  douzeroresnet_encoder.load_model("landlord_up", "baseline/douzero/douzero_resnet/resnet_landlord_up.ckpt", cudaId),
                  ]
    elif kind=="perfectdou":
        models = [perfectdou_encoder.load_model("landlord", "baseline/perfectdou/landlord.onnx",cudaId),
                  perfectdou_encoder.load_model("landlord_down","baseline/perfectdou/landlord_down.onnx",cudaId),
                  perfectdou_encoder.load_model("landlord_up","baseline/perfectdou/landlord_up.onnx", cudaId),
                  ]
    return models

# from baseline.perfectdou.env.encode import (
#     encode_obs_landlord,
#     encode_obs_peasant,
#     _decode_action,
# )
# def act(model,obs):
#     input_name = model.get_inputs()[0].name
#     input_data = np.concatenate(
#         [obs["x_no_action"].flatten(), obs["legal_actions_arr"].flatten()]
#     ).reshape(1, -1)
#     logit = model.run(["action_logit"], {input_name: input_data})
#     action_id = np.argmax(logit)
#     action = _decode_action(action_id, obs["current_hand"], obs["actions"])
#     return action

# from DNQ.mygame.douDiZhu.doudizhu_game import Doudizhu, Action
# from DNQ.mygame.douDiZhu.doudizhu_cheat import setHandCards
# env=Doudizhu()
# env.reset()
# env.dealCards(None,0)
# setHandCards(env.players[0],["大王","小王","10","J","J","J","2","A"])
# allActLists=env.players[0].getAllFirstAction(all=True)
# model = getModel("perfectdou",0)
# a=act(model[0],perfectdou_encoder.toObs(0,perfectdou_encoder.Infoset(env,0,allActLists)))
# id,act=perfectdou_encoder.strToMyAction(a,allActLists)
# print(id,act)