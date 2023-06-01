import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np      
import pickle


from models.attention import SpatialBasis, spatial_softmax, apply_alpha
from models.rbf_mask import RBFMask
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
        self.h = args['vision_h']
        self.w = args['vision_w']
        self.ch = args['vision_ch']
        
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

        self.actor_linear = [nn.Linear(self.hidden_size * 12 * 6, self.num_actions)
                for _ in range(self.num_agents)]
        self.actor_activation = nn.Tanh()

        # attention limits
        self.limit_attention = args['limit_attention']
        self.rbf_limit = args['rbf_limit']
        self.base_weight = args['base_weight']
        if self.limit_attention:
            if self.rbf_limit:
                self.attention_kernels = nn.Parameter(
                    torch.rand(self.num_queries, 2)
                )
                self.rbf_mask = RBFMask(self.h, self.w, self.device)
            else:
                # (n, h, w, num_queries)
                self.attention_base = nn.Parameter(
                    torch.randn(1, self.h, self.w, self.num_queries)
                )

        # initialize
        self.apply(ptu.weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.a_conv.weight.data.mul_(relu_gain)
        # self.c_conv.weight.data.mul_(relu_gain)
        self.att_conv_a1.weight.data.mul_(relu_gain)

        for lin in self.actor_linear:
            lin.weight.data = ptu.norm_col_init(lin.weight.data, 0.01)
            lin.bias.data.fill_(0)
            lin.to(self.device)
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
        regime = torch.argmax(c)
        att_p_feature = self.att_conv_a1(x)[:, regime, :, :].squeeze(1)
        self.att_p = self.sigmoid_a(att_p_feature) # mask-attention
        self.att_p_sig5 = self.sigmoid_a(att_p_feature * 5.0)

        if self.limit_attention:
            if self.rbf_limit:
                attention_base_c = self.rbf_mask(self.attention_kernels.chunk(self.num_agents,
                    dim=0)[regime]).squeeze(-1)
                self.att_p = self.att_p * attention_base_c
            else:
                raise NotImplementedError
                # attention_base_c = self.attention_base.chunk(self.num_agents, dim=3)[regime]
                # self.att_p = ((1 - self.base_weight) * self.att_p + self.base_weight * spatial_softmax(attention_base_c))

        a_mask_x = a_x * self.att_p # mask processing


        self.A = self.att_p.unsqueeze(1).permute(0, 3, 2, 1).detach()
        a_x = a_mask_x
        a_x = a_x.view(a_x.size(0), -1)
        actions = self.actor_activation(self.actor_linear[regime](a_x))

        return actions, None # self.critic_linear(c_x), 

