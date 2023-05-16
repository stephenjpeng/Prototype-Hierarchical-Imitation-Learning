from torch import distributions as dist

from models.attention_agent import AttentionAgents


class CarBaseAgents(AttentionAgents):
    """
    AttentionAgents for the Car Racing domain. Defined with sensible initial values that
    can be updated by args
    """
    def __init__(self, num_agents, args={}):
        self.agent_params = {
                'num_actions': 3,
                'num_policy_heads': 2,      # estimate alpha and beta for the controls
                'lstm_hidden_size': 128,    # paper: 256
                'c_k': 8,                   # paper: 8
                'c_v': 120,                 # paper: 120
                'c_s': 64,                  # paper: 64
                'vision_h': 96,
                'vision_w': 96,
                'vision_ch': 3,
                'num_agents': num_agents,
                'num_queries_per_agent': 2, # paper: 4
                'a_mlp_n_layers': 2,        # paper: 2
                'a_mlp_size': 128,          # paper: 512 / 256
                'q_mlp_n_layers': 2,        # paper: 3
                'q_mlp_size': 128,          # paper: 256 / 128 / 72 x 4
                'policy_act': 'softplus',   # paper: identity
                'values_act': 'identity',   # paper: identity
        }
        self.agent_params.update(args)
        super(CarBaseAgents, self).__init__(self.agent_params)

        self.values = None
        self.policy = None

    def act(self, obs, c, r_prev=None, a_prev=None):
        (alpha, beta), values = self.forward(obs, c, r_prev, a_prev)
        alpha = alpha + 1
        beta = beta + 1
        self.values = values
        self.policy = dist.Beta(alpha, beta)

        return self.policy


