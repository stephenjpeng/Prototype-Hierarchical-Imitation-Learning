import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


import utils.pytorch_util as ptu

class DetectorAgent(nn.Module):
    """
        End-to-end agent that observes the state and outputs a value V_t as well as the
        policy. Structure inspired by Kujanpaa et al '23, which takes from Guez et al '19
    """
    def __init__(self, agent_params):
        super(DetectorAgent, self).__init__()
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']

        self.num_actions = agent_params['num_regimes'] + 1  # classify, or leave alone
        self.hidden_size = agent_params['hidden_size']
        
        self.vision = agent_params['vision_core']

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.prev_hidden = None
        self.prev_cell   = None

        self.mlp = ptu.build_mlp(
            agent_params['hidden_size'], # input from LSTM
            1 + self.num_actions,        # policy plus V
            agent_params['n_layers'],
            agent_params['hidden_size'],
            'tanh',                      # hidden activations
            'identity',                  # output activations
        )
        self.v_activation  = ptu._str_to_activation[agent_params['v_activation']]
        self.pi_activation = ptu._str_to_activation[agent_params['pi_activation']]


    def reset(self):
        self.vision.reset()
        self.prev_hidden = None
        self.prev_cell   = None


    def forward(self, x):
        x = self.vision(x)  # n, h
        if self.prev_hidden is None:
            h, c = self.lstm(x)  # n, h each
        else:
            h, c = self.lstm(x, (self.prev_hidden, self.prev_cell))
            self.prev_hidden, self.prev_cell = h, c

        output = self.mlp(h)  # n, a + 1
        value  = self.v_activation(output[:, 0])   # n, 1
        policy = self.pi_activation(output[:, 1:]) # n, a

        return value, policy
