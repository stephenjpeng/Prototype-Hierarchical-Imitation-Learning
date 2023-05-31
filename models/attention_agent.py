import time
import torch 
import torch.nn as nn
import numpy as np      
import pickle


from models.attention import SpatialBasis, spatial_softmax, apply_alpha
from models.rbf_mask import RBFMask
import utils.pytorch_util as ptu


class AttentionAgents(nn.Module):
    """
    Implement multiple context c-conditioned attention agents as n-headed self attention
    with each k = n/|c| heads corresponding to an agent.

    Takes as state the output of the vision core, maintained separately
    """
    def __init__(self, agent_params):
        super(AttentionAgents, self).__init__()
        self.agent_params = agent_params

        self.device = agent_params['device']
        self.num_actions = agent_params['num_actions']
        self.hidden_size = agent_params['lstm_hidden_size']
        
        self.c_k = agent_params['c_k']
        self.c_v = agent_params['c_v']
        self.c_s = agent_params['c_s']

        self.h = agent_params['vision_h']
        self.w = agent_params['vision_w']
        self.ch = agent_params['vision_ch']

        self.num_agents = agent_params['num_agents']
        self.num_policy_heads = agent_params['num_policy_heads']
        self.num_queries_per_agent = agent_params['num_queries_per_agent']
        self.num_queries = self.num_agents * self.num_queries_per_agent

        self.limit_attention = agent_params['limit_attention']
        self.rbf_limit = agent_params['rbf_limit']
        self.base_weight = agent_params['base_weight']
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

        self.spatial = SpatialBasis(self.h, self.w, self.c_s, int(np.sqrt(self.c_s)))

        self.answer_mlp = ptu.build_mlp(
            # queries + answers + action + reward
            self.num_queries_per_agent * (self.c_k + 2 * self.c_s + self.c_v) +
            self.num_actions + 1,
            self.hidden_size,
            agent_params['a_mlp_n_layers'],
            agent_params['a_mlp_size'],              # hidden size
            'leaky_relu',                                  # hidden activations
            'identity',                              # output activations
        )

        self.policy_core = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.prev_hidden = None
        self.prev_cell   = None

        self.q_mlp = ptu.build_mlp(
            self.hidden_size,                         # input from LSTM
            self.num_queries * (self.c_k + self.c_s), # N * (c_k + c_s) to be reshaped
            agent_params['q_mlp_n_layers'],
            agent_params['q_mlp_size'],               # hidden size
            'leaky_relu',                                   # hidden activations
            'identity',                               # output activations
        )

        self.policy_heads = [
                ptu.build_mlp(self.hidden_size, self.num_actions, 0, 0,
                        'leaky_relu', agent_params['policy_act'])
                for _ in range(self.num_policy_heads)]
        for policy_head in self.policy_heads:
            policy_head.to(self.device)
        self.values_head = ptu.build_mlp(self.hidden_size, self.num_actions, 0, 0, 'leaky_relu',
                agent_params['values_act'])

    def reset(self):
        self.prev_hidden = None
        self.prev_cell   = None

    def forward(self, x, c, r_prev=None, a_prev=None):
        # Setup
        n = x.shape[0]
        regime = torch.argmax(c)
        
        if r_prev is None:
            r_prev = torch.zeros(n, 1, 1).to(self.device)     # (n, 1, 1)
        else:
            r_prev = r_prev.to(self.device) if torch.is_tensor(r_prev) else torch.tensor(r_prev).to(self.device)
            r_prev = r_prev.reshape(n, 1, 1)  # (n, 1, 1)
        if a_prev is None:
            a_prev = torch.zeros(n, 1, self.num_actions).to(self.device)     # (n, 1, a)
        else:
            a_prev = a_prev.to(self.device) if torch.is_tensor(a_prev) else torch.tensor(a_prev).to(self.device)
            a_prev = a_prev.reshape(n, 1, self.num_actions)  # (n, 1, a)

        # Spatial
        # (n, h, w, c_k), (n, h, w, c_v)
        K, V = x.split([self.c_k, self.c_v], dim=3)
        # (n, h, w, c_k + c_s), (n, h, w, c_v + c_s)
        K, V = self.spatial(K), self.spatial(V)

        # Queries
        if self.prev_hidden is None:
            self.prev_hidden = torch.zeros(
                n, self.hidden_size, requires_grad=True
            ).to(self.device)

        Q = self.q_mlp(self.prev_hidden)  # (n, h, w, num_q * (c_k + c_s))
        Q = Q.reshape(n, self.num_queries, self.c_k + self.c_s)  # (n, num_queries, c_k + c_s)
        Q = Q.chunk(self.num_agents, dim=1)[regime]  # (n, num_queries_per_agent, c_k + c_s)

        # Answer
        A = torch.matmul(K, Q.transpose(2, 1).unsqueeze(1))  # (n, h, w, num_queries_per_agent)
        # NOTE: Scaled by sqrt(d)
        A = A / np.sqrt(self.c_k + self.c_s)
        # (n, h, w, num_queries_per_agent)
        A = spatial_softmax(A)
        if self.limit_attention:
            if self.rbf_limit:
                attention_base_c = self.rbf_mask(self.attention_kernels.chunk(self.num_agents, dim=0)[regime])
                A = A * spatial_softmax(attention_base_c)
            else:
                attention_base_c = self.attention_base.chunk(self.num_agents, dim=3)[regime]
                A = ((1 - self.base_weight) * A + self.base_weight * spatial_softmax(attention_base_c))
        self.A = A.clone().detach()
        # (n, 1, 1, num_queries_per_agent)
        a = apply_alpha(A, V)

        # (n, (c_v + c_s) * num_queries_per_agent + (c_k + c_s) * num_queries_per_agent + 1 + a)
        answer = torch.cat(
            torch.chunk(a, 4, dim=1)
            + torch.chunk(Q, 4, dim=1)
            + (r_prev.float(), a_prev.float()),
            dim=2,
        ).squeeze(1)
        # (n, hidden_size)
        answer = self.answer_mlp(answer)

        # Policy
        if self.prev_cell is None:
            h, c = self.policy_core(answer)
        else:
            h, c = self.policy_core(answer, (self.prev_hidden, self.prev_cell))
        self.prev_hidden, self.prev_cell = h, c
        # (n, hidden_size)
        output = h

        # Outputs
        # (n, num_actions)
        action = [policy_head(output) for policy_head in self.policy_heads]
        # (n, num_actions)
        values = self.values_head(output)
        return action, values

