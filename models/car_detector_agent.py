import torch.distributions as dist


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
        super(CarDetectorAgent, self).__init__(self.agent_params)

    def act(self, obs, c, valid=None):
        """
        valid [np.ndarray or None]: mask of valid actions
        """
        value, policy = self.forward(obs, c)
        self.value = value
        self.policy = dist.Categorical(logits=policy)
        # check unsqueeze is necessary
        import pdb; pdb.set_trace()
        if valid is not None:
            self.policy = self.policy * valid
        return self.policy
