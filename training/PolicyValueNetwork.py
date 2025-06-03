import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyValueNetwork(nn.Module):
    def __init__(self, policy, value):
        super().__init__()
        self.policy = policy
        self.value = value

    def forward(self, states):
        distribution = self.policy(states)
        value = self.value(states)
        return distribution, value