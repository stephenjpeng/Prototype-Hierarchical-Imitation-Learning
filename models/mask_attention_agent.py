import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np      
import pickle


from models.attention import SpatialBasis, spatial_softmax, apply_alpha
import utils.pytorch_util as ptu


class MaskAttentionAgents(nn.Module):
    '''
    Code adapted from https://github.com/machine-perception-robotics-group/Mask-Attention_A3C
    '''
    def __init__(self, args):
        super(MaskAttentionAgents, self).__init__()
        # # critic
        # self.c_conv = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        # self.critic_linear = nn.Linear(3200, 1)
        
        self.hidden_size = args['hidden_size']
        self.device = args['device']
        self.num_actions = args['num_actions']

        self.num_agents = args['num_agents']
        self.num_policy_heads = args['num_policy_heads']
        self.num_queries_per_agent = args['num_queries_per_agent']
        if self.num_queries_per_agent > 1:
            print("Must have only one query per agent for simple mask attn!")
            raise NotImplementedError
        self.num_queries = self.num_agents * self.num_queries_per_agent

        # actor
        self.att_conv_a1 = nn.Conv2d(self.hidden_size, self.num_queries, 1, stride=1, padding=0)
        self.sigmoid_a = nn.Sigmoid()
        self.a_conv = nn.Conv2d(self.hidden_size, self.hidden_size // 2, 1, stride=1, padding=0)

        self.actor_linear = nn.Linear(self.hidden_size * 12 * 6, self.num_actions)
        self.actor_activation = nn.Sigmoid()

        self.apply(ptu.weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.a_conv.weight.data.mul_(relu_gain)
        # self.c_conv.weight.data.mul_(relu_gain)
        self.att_conv_a1.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = ptu.norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = ptu.norm_col_init(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

        self.train()

    def reset(self):
        return

    def forward(self, x, c, r_prev=None, a_prev=None):
        # # Critic
        # c_x = F.relu(self.c_conv(x))
        # c_x = c_x.view(c_x.size(0), -1)
 
        # Actor
        x = x.transpose(1, 3)
        a_x = F.relu(self.a_conv(x))

        # n, h, w, 1
        att_p_feature = self.att_conv_a1(x)[:, c, :, :].squeeze(1)
        self.att_p = self.sigmoid_a(att_p_feature) # mask-attention
        self.att_p_sig5 = self.sigmoid_a(att_p_feature * 5.0)
        a_mask_x = a_x * self.att_p # mask processing
        self.A = self.att_p.unsqueeze(1).permute(0, 3, 2, 1).detach()
        a_x = a_mask_x
        a_x = a_x.view(a_x.size(0), -1)
        actions = self.actor_activation(self.actor_linear(a_x))

        return actions, None # self.critic_linear(c_x), 

