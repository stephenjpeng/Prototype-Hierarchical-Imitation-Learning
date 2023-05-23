from torch import distributions as dist

from models.attention_agent import AttentionAgents


class CarBaseAgents(AttentionAgents):
    """
    AttentionAgents for the Car Racing domain. Defined with sensible initial values that
    can be updated by args
    """
    def __init__(self, num_agents, base_mlp_size, args={}):
        self.agent_params = {
                'num_actions': 2,
                'num_policy_heads': 1,      # estimate controls straight up
                'lstm_hidden_size': 128,    # paper: 256
                'c_k': 8,                   # paper: 8
                'c_v': 56,                 # paper: 120
                'c_s': 16,                  # paper: 64
                'vision_h': 12,
                'vision_w': 12,
                'vision_ch': 3,
                'num_agents': num_agents,
                'num_queries_per_agent': 2, # paper: 4
                'a_mlp_n_layers': 2,        # paper: 2
                'a_mlp_size': base_mlp_size, # 32,          # paper: 512 / 256
                'q_mlp_n_layers': 2,        # paper: 3
                'q_mlp_size': base_mlp_size, # 32,          # paper: 256 / 128 / 72 x 4
                'policy_act': 'sigmoid',   # paper: identity
                'values_act': 'identity',   # paper: identity
        }
        self.agent_params.update(args)
        super(CarBaseAgents, self).__init__(self.agent_params)

        self.values = None
        self.action = None

    def act(self, obs, c, r_prev=None, a_prev=None):
        action, values = self.forward(obs, c, r_prev, a_prev)
        self.values = values
        self.action = action[0]

        return self.action
