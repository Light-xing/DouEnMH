import torch
import numpy as np
import os
from douzero.env.env import get_obs


def _load_model(position, model_path):
    from douzero.dmc.models import model_dict, GeneralModelTransformer

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        device = torch.device('cpu')
        pretrained = torch.load(model_path, map_location='cpu')

    if position in ('first', 'second', 'third'):
        model = model_dict[position]()
    else:
        # Auto-detect Transformer architecture from state_dict
        embed_key = 'state_embed.weight'
        embed_weight = pretrained.get(embed_key)
        d_model = embed_weight.shape[0]

        num_layers = 0
        while f'state_encoder.layers.{num_layers}.self_attn.in_proj_weight' in pretrained:
            num_layers += 1

        ff_weight = pretrained['state_encoder.layers.0.linear1.weight']
        dim_feedforward = ff_weight.shape[0]
        nhead = 8  # default, cannot be inferred from weights

        model = GeneralModelTransformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward)

    model.load_state_dict(pretrained)
    model.to(device)
    model.eval()
    return model


class DeepAgent:

    def __init__(self, position, model_path):
        self.model = _load_model(position, model_path)
        self.EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                            8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                            13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

    def act(self, infoset):
        obs = get_obs(infoset, infoset.bid_over)

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        result = self.model.forward(z_batch, x_batch, return_value=True)
        win_rate, win, lose = result['values']
        _win_rate = (win_rate + 1) / 2
        y_pred = _win_rate * win + (1. - _win_rate) * lose
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        return best_action


class SupervisedModel:

    def __init__(self):
        from baseline.SLModel.BidModel import Net2 as Net
        self.net = Net()
        self.net.eval()
        if torch.cuda.is_available():
            self.gpu = True
        else:
            self.gpu = False
        if self.gpu:
            self.net = self.net.cuda()
        if os.path.exists("baseline/SLModel/bid_weights_new.pkl"):
            if torch.cuda.is_available():
                self.net.load_state_dict(torch.load('baseline/SLModel/bid_weights_new.pkl'))
            else:
                self.net.load_state_dict(torch.load('baseline/SLModel/bid_weights_new.pkl', map_location=torch.device("cpu")))

    def RealToOnehot(self, cards):
        Onehot = torch.zeros((4, 15))
        m = 0
        for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30]:
            Onehot[:cards.count(i), m] = 1
            m += 1
        return Onehot

    def predict_score(self, cards):
        input = self.RealToOnehot(cards)
        input = torch.flatten(input)
        input = input.unsqueeze(0)
        result = self.net(input)
        return result[0].item()

    def act(self, infoset):
        legal_action = infoset.legal_actions
        obs = torch.flatten(self.RealToOnehot(infoset.player_hand_cards))
        if self.gpu:
            obs = obs.cuda()
        predict = self.net.forward(obs.unsqueeze(0))
        one = -0.1
        two = 0
        three = 0.1
        if predict > three and ([3] in legal_action):
            return [3]
        elif predict > two and ([2] in legal_action):
            return [2]
        elif predict > one and ([1] in legal_action):
            return [1]
        else:
            return [0]


def RealToOnehot(cards):
    RealCard2EnvCard = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
                        '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9,
                        'K': 10, 'A': 11, '2': 12, 'X': 13, 'D': 14}
    cards = [RealCard2EnvCard[c] for c in cards]
    Onehot = torch.zeros((4,15))
    for i in range(0, 15):
        Onehot[:cards.count(i),i] = 1
    return Onehot
