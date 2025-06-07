import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..Brain import load_brain, load_value

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyValueNetwork(nn.Module):
    def __init__(self, policy, value):
        super().__init__()
        # self.policy_archer = policy_archer
        # self.policy_knight = policy_knight
        self.policy = policy
        self.value = value
        self.class_map = {'knight' : 1, 'archer' : 2}

    def forward(self, states, mem_1, mem_2):
        # print(states.shape, mem_1.shape, mem_2.shape)
        distribution, mem_1 = self.policy(states, mem_1)
        value, mem_2 = self.value(states, mem_2)
        return distribution, mem_1, mem_2, value

def load_policy_value_net(dir_path, obs_shape, device=None):
    policy = load_brain(obs_shape, os.path.join(dir_path, 'policy'), 'archer', device)
    value = load_value(obs_shape, os.path.join(dir_path, 'value'), device)