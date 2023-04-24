from collections import Counter#计数
import numpy as np
import torch
from doudizhu_game import Doudizhu, getNum,Action,cardIdListlistToNum
import onnx
import onnxruntime as ort
# from baseline.perfectdou.env.encode import (#python3.7才能用，3.8就不行
#     encode_obs_landlord,
#     encode_obs_peasant,
#     _decode_action,
# )

myCardToEnvCard = {3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                    8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                    13: 13, 1: 14, 2: 17,14: 20, 15: 30}
cardnameTomyCard= {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                     'K': 13, 'A': 1, '2': 2,'B': 14, 'R': 15}

posList=["landlord","landlord_down","landlord_up"]
Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}
myCardTodouzeroCode = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 1: 11, 2: 12}
NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}
def toPerfectdouList(cards):
    # print(isinstance(cards, Action))
    if isinstance(cards, list):
        cards=cards.copy()
    else:
        cards=cards.tolist()
    for i in range(len(cards)):
        cards[i]=myCardToEnvCard[getNum(cards[i])]
    return cards
class Infoset():
    def __init__(self,env:Doudizhu,pid,allActLists):
        self.player_position=posList[(pid-env.dealer+3)%3]
        self.legal_actions=[]
        for cards in allActLists:
            li=toPerfectdouList(cards)
            self.legal_actions.append(li)

        self.player_hand_cards = toPerfectdouList(env.players[pid].cards)
        self.other_hand_cards=[]
        for pos in [env.dealer, (env.dealer+2)%3, (env.dealer+1)%2]:
            if pos != pid:
                self.other_hand_cards+=env.players[pos].cards
        self.other_hand_cards=toPerfectdouList(self.other_hand_cards)
        self.last_move=[]
        if len(env.historyAct)!=0:
            if env.historyAct[-1].isPass():
                self.last_move=env.historyAct[-2].tolist()
            else:
                self.last_move = env.historyAct[-1].tolist()
        self.last_move = toPerfectdouList(self.last_move)
        self.all_handcards={}
        self.all_handcards["landlord"]=toPerfectdouList(env.players[(env.dealer+0)%3].cards)
        self.all_handcards["landlord_up"] = toPerfectdouList(env.players[(env.dealer + 2) % 3].cards)
        self.all_handcards["landlord_down"] = toPerfectdouList(env.players[(env.dealer + 1) % 3].cards)
        self.num_cards_left_dict={}
        self.num_cards_left_dict["landlord"]=len(env.players[(env.dealer+0)%3].cards)
        self.num_cards_left_dict["landlord_up"] = len(env.players[(env.dealer + 2) % 3].cards)
        self.num_cards_left_dict["landlord_down"] = len(env.players[(env.dealer + 1) % 3].cards)
        self.played_cards={}
        self.played_cards["landlord"] = toPerfectdouList(env.players[(env.dealer + 0) % 3].uncards)
        self.played_cards["landlord_up"] = toPerfectdouList(env.players[(env.dealer + 2) % 3].uncards)
        self.played_cards["landlord_down"] = toPerfectdouList(env.players[(env.dealer + 1) % 3].uncards)

        self.last_pid='landlord'
        lastact=env.getLastHistoryAct()
        if lastact!=None:
            self.last_pid=posList[(lastact.playerId-env.dealer+3)%3]
        self.bomb_num=env.bomb_num
        self.card_play_action_seq=[]
        for act in env.historyAct:
            self.card_play_action_seq.append(toPerfectdouList(act))
        self.last_move_dict={}
        self.last_move_dict['landlord']=toPerfectdouList(env.players[(env.dealer+0)%3].last_move)
        self.last_move_dict['landlord_up'] = toPerfectdouList(env.players[(env.dealer + 2) % 3].last_move)
        self.last_move_dict['landlord_down'] = toPerfectdouList(env.players[(env.dealer + 1) % 3].last_move)
        self.last_move=[]
        if len(env.historyAct)!=0:
            if env.historyAct[-1].isPass():
                self.last_move=env.historyAct[-2].tolist()
            else:
                self.last_move = env.historyAct[-1].tolist()
        self.last_move =toPerfectdouList( self.last_move)
        self.last_two_moves = [[], []]
        for card in self.card_play_action_seq[-2:]:
            self.last_two_moves.insert(0, card)
            self.last_two_moves = self.last_two_moves[:2]

        self.three_landlord_cards = toPerfectdouList(env.underCards)
def strToMyAction(cards_str:str,allActionList):
    if cards_str=="pass":
        # print(cards_str, allActionList[0].tolist())
        return 0, allActionList[0]
    cards = list(cards_str)
    n = len(cards)
    for i in range(len(cards)):
        cards[i]=cardnameTomyCard[cards[i]]
    cards.sort()
    for i in range(len(allActionList)):
        act=allActionList[i]
        li=act.tolist()
        cardIdListlistToNum(li)
        li.sort()
        # print(cards,li)
        if act.len==n and all(x == y for x, y in zip(cards, li)):
            return i,act
    return None,None

def toObs(kind,info):
    if kind==0 or kind=="landlord":
        return encode_obs_landlord(info)
    elif kind==2 or kind=="landlord_up":
        return encode_obs_peasant(info)
    elif kind==1 or kind == "landlord_down":
        return encode_obs_peasant(info)
def getAct(model,obs):
    input_name = model.get_inputs()[0].name
    input_data = np.concatenate(
        [obs["x_no_action"].flatten(), obs["legal_actions_arr"].flatten()]
    ).reshape(1, -1)
    logit = model.run(["action_logit"], {input_name: input_data})
    action_id = np.argmax(logit)
    action = _decode_action(action_id, obs["current_hand"], obs["actions"])
    return action
def _load_model(model_path):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = 1
    sess_options.log_severity_level = 3
    return ort.InferenceSession(model_path, sess_options)

def load_model(position, model_path,cudaId):
    return _load_model(model_path)
