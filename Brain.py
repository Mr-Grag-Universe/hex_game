import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import *
from bisect import bisect_left

from utils import ij2xy_obs, xy2ij_obs

class PreNet(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.h = h
        self.w = w
        self.c = c

        self.conv1_1 = nn.Conv2d(c, 64, 3, padding=1)
        # self.conv1_2 = nn.Conv2d(c, 64, 5, padding=2)
        self.conv1_2 = nn.Conv2d(c, 64, 7, padding=3)
        # self.conv1_4 = nn.Conv2d(c, 64, 9, padding=4)
        
        self.conv_1x1 = nn.Conv2d(2*64, 256, 1)

        self.memory_proj = nn.Linear(256*h*w, 256)
        self.memory = nn.GRUCell(256, 256, True)

        # self.conv2_1 = nn.Conv2d(256, 256, 3, padding=1)
        # self.conv2_2 = nn.Conv2d(256, 256, 5, padding=2)
        # self.conv2_3 = nn.Conv2d(256, 256, 7, padding=3)
        # self.conv2_4 = nn.Conv2d(256, 256, 9, padding=4)

        # self.fc_inner = nn.Linear(256*self.h*self.w*4, 4)

        self.initialize_weights()

    def initialize_weights(self):
        # Инициализация весов для всех слоёв
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.LazyConv2d):
                nn.init.kaiming_uniform_(layer.weight, a=0.1)  # Kaiming (He) инициализация
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Инициализация смещений нулями
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier инициализация для полносвязных слоёв
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x, mem):
        out_1 = self.conv1_1(x)
        out_2 = self.conv1_2(x)
        # out_3 = self.conv1_3(x)
        # out_4 = self.conv1_4(x)
        out = torch.selu(torch.cat([out_1, out_2], dim=int(len(x.shape) > 3)))
        out = torch.selu(self.conv_1x1(out))

        proj = self.memory_proj(nn.Flatten(start_dim=-3)(out))
        # print(proj.shape, mem.shape)
        mem = self.memory(proj, mem)
        
        # out_1 = self.conv2_1(out)
        # out_2 = self.conv2_2(out)
        # out_3 = self.conv2_3(out)
        # out_4 = self.conv2_4(out)
        # out = torch.cat([out_1, out_2, out_3, out_4], dim=int(len(x.shape) > 3))
        # out = nn.Flatten()(out.view(-1, 1024, self.h, self.w))
        # p = (torch.tanh(torch.selu(self.fc_inner(out)))+1) / 2
        # p = p.view(-1, 4)
        
        # out =   p[:, 0].view(-1,1,1,1) * out_1.view(-1, 256, self.h, self.w) + \
        #         p[:, 1].view(-1,1,1,1) * out_2.view(-1, 256, self.h, self.w) + \
        #         p[:, 2].view(-1,1,1,1) * out_3.view(-1, 256, self.h, self.w) + \
        #         p[:, 3].view(-1,1,1,1) * out_4.view(-1, 256, self.h, self.w)
        if out.shape[0] == 1:
            out = out.squeeze(0)
        # print("out: ", out.shape)
        return out, mem

class ValueNet(nn.Module):
    def __init__(self, size=None, prenet : Optional[PreNet] = None):
        super().__init__()
        if size is not None:
            c, h, w = size
        else:
            c, h, w = prenet.c, prenet.h, prenet.w
        self.h = h
        self.w = w
        self.c = c
        self.value_head = nn.Sequential(nn.Flatten(), nn.Linear(256*h*w, 64), nn.Linear(64, 1))
        self.prenet = prenet
        if prenet is None:
            self.prenet = PreNet(c, h, w)
    
    def forward(self, state : torch.Tensor, mem : torch.Tensor):
        B = len(state) if state.ndim == 4 else 1
        out, mem = self.prenet(state, mem)
        # print(out.shape, (256, self.h, self.w), nn.Flatten()(out).shape)
        return self.value_head(out.view(B, 256, self.h, self.w)), mem

class Distribution:
    def __init__(self, dist : dict):
        '''
            dist = {'move' : {'proba' : p, 'heat__map' : H}, 'attack' : {'proba' : p, 'heat__map' : H}}
        '''
        # print(dist)
        self.use_batch = len(dist['move']['heat_map'].shape) > 2
        self.dist = dist
        self.actions = list(dist.keys())
        self.actions_offset = []
        self.shapes = []
        offset = 0
        for action in self.actions:
            self.actions_offset.append(offset)
            shape = dist[action]['heat_map'].shape[-2:]
            offset += shape[-2]*shape[-1]
            self.shapes.append(shape)
        
        hm_vectors = []
        for action in self.actions:
            hm = dist[action]['heat_map'].flatten(start_dim=int(self.use_batch))
            p = dist[action]['p']
            if self.use_batch:
                p = p.view(-1, 1)
            d = hm*p
            hm_vectors.append(d)
        self.all_proba_vector = torch.cat(hm_vectors, dim=int(self.use_batch))
        # print(self.all_proba_vector.sum(), self.all_proba_vector)
        self.m_dist = torch.distributions.Multinomial(1, probs=self.all_proba_vector)
    
    def sample(self, *args, **kwargs):
        return self.m_dist.sample(*args, **kwargs)

    def log_prob(self, value):
        return self.m_dist.log_prob(value)
    
    def entropy(self):
        return self.m_dist.entropy()

    def decode_action(self, action_ind):
        i = bisect_left(self.actions_offset, action_ind)-1
        action = self.actions[i]
        shape = self.shapes[i]
        ind = action_ind-self.actions_offset[i]
        pos = (ind // shape[1], ind % shape[1])

        return {'type' : action, 'params' : {'pos' : ij2xy_obs(*pos)}}

    def encode_action(self, action):
        action, pos = action['type'], action['params']['pos']
        pos = xy2ij_obs(*pos)
        i = self.actions.index(action)
        base = self.actions_offset[i]
        matrix_offset = pos[0]*self.shapes[i][1]+pos[1]
        return base+matrix_offset

class ArcherBrain(nn.Module):
    def __init__(self, size=None, prenet : Optional[PreNet] = None, return_dist=False):
        super().__init__()
        assert size is not None or prenet is not None, RuntimeError("you should pass size or prenet")
        if size is not None:
            c, h, w = size
        else:
            c, h, w = prenet.c, prenet.h, prenet.w
        self.h = h
        self.w = w
        self.c = c
        self.return_dist = return_dist
        self.actions = ['move', 'attack', 'none']
        self.chose_action_head = nn.Sequential(nn.Linear(256*h*w, len(self.actions)), nn.Softmax(dim=-1))
        self.actions_heads_move = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.SELU(), nn.Conv2d(64, 16, 3, padding=1), nn.SELU(), nn.Conv2d(16, 1, 3, padding=1), nn.Softmax(dim=-1))
        self.actions_heads_attack = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.SELU(), nn.Conv2d(64, 16, 3, padding=1), nn.SELU(), nn.Conv2d(16, 1, 3, padding=1), nn.Softmax(dim=-1))
        self.prenet = prenet
        if prenet is None:
            self.prenet = PreNet(c, h, w)

        self.initialize_weights()

    def initialize_weights(self):
        # Инициализация весов для всех слоёв
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.LazyConv2d):
                nn.init.kaiming_uniform_(layer.weight, a=0.1)  # Kaiming (He) инициализация
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Инициализация смещений нулями
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier инициализация для полносвязных слоёв
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, obs, mem):
        pred, mem = self.prenet(obs, mem)
        action_proba = self.chose_action_head(nn.Flatten(start_dim=int(len(pred.shape) > 3))(pred))
        if not self.return_dist:
            action = torch.argmax(input=action_proba)
            action = self.actions[action]
            
            params = None
            match action:
                case 'move':
                    heat_map = self.actions_heads_move(pred).view(self.h, self.w)
                    pos = torch.where(heat_map == heat_map.max())
                    pos = pos[0][0], pos[1][0]
                    params = {'pos_pt' : pos, 'pos' : tuple(map(float, pos)), 'heat_map' : heat_map}
                case 'attack':
                    heat_map = self.actions_heads_attack(pred).view(self.h, self.w)
                    pos = torch.where(heat_map == heat_map.max())
                    pos = pos[0][0], pos[1][0]
                    params = {'pos_pt' : pos, 'pos' : tuple(map(float, pos)), 'heat_map' : heat_map}
                case 'none':
                    params = {'heat_map' : torch.tensor([[0.]])}
            
            return {'type_proba' : action_proba, 'type' : action, 'params' : params}
        else:
            heat_map_1 = self.actions_heads_move(pred).view(-1, self.h, self.w)
            heat_map_2 = self.actions_heads_attack(pred).view(-1, self.h, self.w)
            heat_map_3 = torch.ones_like(heat_map_1[:, :1, :1])
            use_batch = (heat_map_1.shape[0] > 1)
            if not use_batch:
                heat_map_1 = heat_map_1.squeeze(0)
                heat_map_2 = heat_map_2.squeeze(0)
                heat_map_3 = heat_map_3.squeeze(0)
            p_1 = action_proba[:,0] if use_batch else action_proba[0]
            p_2 = action_proba[:,1] if use_batch else action_proba[1]
            p_3 = action_proba[:,2] if use_batch else action_proba[2]
            return Distribution({
                                'move'   : {'p' : p_1, 'heat_map' : heat_map_1}, 
                                'attack' : {'p' : p_2, 'heat_map' : heat_map_2},
                                'none'   : {'p' : p_3, 'heat_map' : heat_map_3}
                                }), mem

class KnightBrain(nn.Module):
    def __init__(self, size=None, prenet : Optional[PreNet] = None, return_dist=False):
        super().__init__()
        assert size is not None or prenet is not None, RuntimeError("you should pass size or prenet")
        if size is not None:
            c, h, w = size
        else:
            c, h, w = prenet.c, prenet.h, prenet.w
        self.h = h
        self.w = w
        self.c = c
        self.return_dist = return_dist
        self.actions = ['move', 'attack']
        self.chose_action_head = nn.Sequential(nn.Linear(256*h*w, len(self.actions)))
        self.actions_heads_move = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.SELU(), nn.Conv2d(64, 16, 3, padding=1), nn.SELU(), nn.Conv2d(16, 1, 3, padding=1))
        self.actions_heads_attack = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.SELU(), nn.Conv2d(64, 16, 3, padding=1), nn.SELU(), nn.Conv2d(16, 1, 3, padding=1))
        self.prenet = prenet
        if prenet is None:
            self.prenet = PreNet(h, w)

        self.initialize_weights()

            
    def initialize_weights(self):
        # Инициализация весов для всех слоёв
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.LazyConv2d):
                nn.init.kaiming_uniform_(layer.weight, a=0.1)  # Kaiming (He) инициализация
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Инициализация смещений нулями
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier инициализация для полносвязных слоёв
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, obs):
        pred = self.prenet(obs)
        action_proba = self.chose_action_head(nn.Flatten(start_dim=int(len(pred.shape) > 3))(pred))
        if not self.return_dist:
            action = torch.argmax(input=action_proba)
            action = self.actions[action]
            
            match action:
                case 'move':
                    heat_map = self.actions_heads_move(pred).view(self.h, self.w)
                    pos = torch.where(heat_map == heat_map.max())
                    pos = pos[0][0], pos[1][0]
                    params = {'pos_pt' : pos, 'pos' : tuple(map(float, pos)), 'heat_map' : heat_map}
                case 'attack':
                    heat_map = self.actions_heads_attack(pred).view(self.h, self.w)
                    pos = torch.where(heat_map == heat_map.max())
                    pos = pos[0][0], pos[1][0]
                    params = {'pos_pt' : pos, 'pos' : tuple(map(float, pos)), 'heat_map' : heat_map}
            
            return {'type_proba' : action_proba, 'type' : action, 'params' : params}
        else:
            heat_map_1 = self.actions_heads_move(pred).view(-1, self.h, self.w)
            heat_map_2 = self.actions_heads_attack(pred).view(-1, self.h, self.w)
            use_batch = (heat_map_1.shape[0] > 1)
            if not use_batch:
                heat_map_1 = heat_map_1.squeeze(0)
                heat_map_2 = heat_map_2.squeeze(0)
            p_1 = action_proba[:,0] if use_batch else action_proba[0]
            p_2 = action_proba[:,1] if use_batch else action_proba[1]
            return Distribution({
                                'move'   : {'p' : p_1, 'heat_map' : heat_map_1}, 
                                'attack' : {'p' : p_2, 'heat_map' : heat_map_2}
                                })
    