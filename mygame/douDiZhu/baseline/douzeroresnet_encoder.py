from collections import Counter#计数
import numpy as np

import baseline.douzero.model_resnet as douzero_models
import torch
from doudizhu_game import Doudizhu, getNum,Action

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}
myCardTodouzeroCode = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 1: 11, 2: 12}
NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}
class Infoset():
    def __init__(self,env:Doudizhu,pid,allActLists):
        self.legal_actions=allActLists
        self.player_hand_cards = env.players[pid].cards
        self.other_hand_cards=[]
        for pos in [env.dealer, (env.dealer+2)%3, (env.dealer+1)%2]:
            if pos != pid:
                self.other_hand_cards+=env.players[pos].cards
        self.last_move=[]
        if len(env.historyAct)!=0:
            if env.historyAct[-1].isPass():
                self.last_move=env.historyAct[-2].tolist()
            else:
                self.last_move = env.historyAct[-1].tolist()
        self.all_handcards={}
        self.all_handcards["landlord"]=env.players[(env.dealer+0)%3].cards
        self.all_handcards["landlord_up"] = env.players[(env.dealer + 2) % 3].cards
        self.all_handcards["landlord_down"] = env.players[(env.dealer + 1) % 3].cards
        self.num_cards_left_dict={}
        self.num_cards_left_dict["landlord"]=len(env.players[(env.dealer+0)%3].cards)
        self.num_cards_left_dict["landlord_up"] = len(env.players[(env.dealer + 2) % 3].cards)
        self.num_cards_left_dict["landlord_down"] = len(env.players[(env.dealer + 1) % 3].cards)
        self.played_cards={}
        self.played_cards["landlord"] = env.players[(env.dealer + 0) % 3].uncards
        self.played_cards["landlord_up"] = env.players[(env.dealer + 2) % 3].uncards
        self.played_cards["landlord_down"] = env.players[(env.dealer + 1) % 3].uncards
        self.bomb_num=env.bomb_num
        self.card_play_action_seq=env.historyAct
        self.last_move_dict={}
        self.last_move_dict['landlord']=env.players[(env.dealer+0)%3].last_move.tolist()
        self.last_move_dict['landlord_up'] = env.players[(env.dealer + 2) % 3].last_move.tolist()
        self.last_move_dict['landlord_down'] = env.players[(env.dealer + 1) % 3].last_move.tolist()
        self.three_landlord_cards=env.underCards
        self.bid_info = np.array([
            [1,0.5,1],
            [1,1,1],
            [1,1,-4],
            [1,1,1]])
        # Fixed score info for resnet model.
        self.multiply_info = [1, 1, 1]
def _cards2array_1(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))
def _cards2array(cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if isinstance(cards, Action):
        cards=cards.tolist()
    ans=np.zeros(54, dtype=np.int8)
    # print(cards)
    if len(cards) == 0:
        return ans
    cnt = [0]*15
    for card in cards:
        a = getNum(card)
        cnt[a - 1] += 1
    for i in range(13):
        a=myCardTodouzeroCode[i+1]
        ans[a*4:a*4+4]=NumOnes2Array[cnt[i]]
    ans[52] = cnt[13]
    ans[53] = cnt[14]
    return ans
def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot
def _get_one_hot_bomb(bomb_num):
    """
    A utility function to encode the number of bombs
    into one-hot representation.
    """
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot
def _process_action_seq(historyActList, length=15,model_type="mlp"):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """

    sequence = []
    hislen=len(historyActList)
    for i in range(max(0,hislen-length),hislen):
        sequence.append(historyActList[i].tolist())
    if model_type == "resnet":
        sequence = sequence[::-1]
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _action_seq_list2array(action_seq_list, seq_len=0,  model_type="mlp"):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5x162 matrix, which will be fed
    into LSTM for encoding.
    """
    if model_type == "resnet":
        action_seq_array = np.ones((len(action_seq_list), 54)) * -1  # Default Value -1 for not using area
        empty_len = len(action_seq_list) - seq_len
        for row, list_cards in enumerate(action_seq_list):
            if list_cards or row >= empty_len:
                action_seq_array[row, :] = _cards2array(list_cards)
    else:
        action_seq_array = np.zeros((len(action_seq_list), 54))
        for row, list_cards in enumerate(action_seq_list):
            action_seq_array[row, :] = _cards2array(list_cards)
        action_seq_array = action_seq_array.reshape(5, 162)
    return action_seq_array

def _get_obs_resnet(infoset, position):
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    position_map = {
        "landlord": [1, 0, 0],
        "landlord_up": [0, 1, 0],
        "landlord_down": [0, 0, 1]
    }
    position_info = np.array(position_map[position])
    position_info_batch = np.repeat(position_info[np.newaxis, :],
                                    num_legal_actions, axis=0)

    bid_info = np.array(infoset.bid_info).flatten()
    bid_info_batch = np.repeat(bid_info[np.newaxis, :],
                               num_legal_actions, axis=0)

    multiply_info = np.array(infoset.multiply_info)
    multiply_info_batch = np.repeat(multiply_info[np.newaxis, :],
                                    num_legal_actions, axis=0)

    three_landlord_cards = _cards2array(infoset.three_landlord_cards)
    three_landlord_cards_batch = np.repeat(three_landlord_cards[np.newaxis, :],
                                           num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action.tolist())

    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    other_handcards_left_list = []
    for pos in ["landlord", "landlord_up", "landlord_up"]:
        if pos != position:
            other_handcards_left_list.extend(infoset.all_handcards[pos])

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)
    num_cards_left = np.hstack((
        landlord_num_cards_left,  # 20
        landlord_up_num_cards_left,  # 17
        landlord_down_num_cards_left))

    x_batch = np.hstack((
        bid_info_batch,  # 12
        multiply_info_batch))  # 3
    x_no_action = np.hstack((
        bid_info,
        multiply_info))
    z =np.vstack((
        num_cards_left,
        my_handcards,  # 54
        other_handcards,  # 54
        three_landlord_cards,  # 54
        landlord_played_cards,  # 54
        landlord_up_played_cards,  # 54
        landlord_down_played_cards,  # 54
        _action_seq_list2array(
            _process_action_seq(
                infoset.card_play_action_seq, 32, model_type="resnet"
            ),
            seq_len=len(infoset.card_play_action_seq),
            model_type="resnet"
        )
    ))

    _z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    my_action_batch = my_action_batch[:,np.newaxis,:]
    z_batch = np.zeros([len(_z_batch),40,54],int)
    for i in range(0,len(_z_batch)):
        z_batch[i] = np.vstack((my_action_batch[i],_z_batch[i]))
    obs = {
        'position': position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs

def toObs(kind,info):
    if kind==0 or kind=="landlord":
        return _get_obs_resnet(info,"landlord")
    elif kind==2 or kind=="landlord_up":
        return _get_obs_resnet(info,"landlord_up")
    elif kind==1 or kind == "landlord_down":
        return _get_obs_resnet(info,"landlord_down")

def load_model(position, model_path,cudaId):
    model = douzero_models.model_dict_resnet[position]().cuda(cudaId)
    # print(model)
    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:'+str(cudaId))
    else:
        print("没有cuda")
        exit()
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda(cudaId)
    model.eval()
    return model