import torch
import torch.distributions as dist
import torch.nn.functional as F


from models.detector import DetectorAgent


class CarDetectorAgent(DetectorAgent):
    """
    Detector Agent for the Car Racing domain. Defined with sensible initial values that
    can be updated by args
    """
    def __init__(self, args={}):
        self.agent_params = {
                'max_regimes': 5,
                'n_layers': 2,
                'hidden_size': 256,
                'v_activation': 'identity',
                'pi_activation': 'identity',
        }
        self.agent_params.update(args)
        self.value = 0
        self.policy = None
        self.device = args['device']
        super(CarDetectorAgent, self).__init__(self.agent_params)

    def act(self, obs, c, valid=None):
        """
        valid [np.ndarray or None]: mask of valid actions
        """
        value, policy = self.forward(obs, c)
        self.value = value
        policy = F.softmax(policy, dim=1)
        if valid is not None:
            valid = torch.tensor(valid).to(self.device)
            policy = policy * valid
            policy /= policy.sum(axis=1)
        self.policy = dist.Categorical(probs=policy)
        return self.policy
