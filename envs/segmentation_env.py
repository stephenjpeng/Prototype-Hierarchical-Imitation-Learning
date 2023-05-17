import gym
import numpy as np
import torch
import torchvision.transforms.functional as F


from PIL import Image
from PIL import ImageDraw


class OfflineEnv(gym.Env):
    def __init__(
            self,
            D,            # dataset of trajectories (N, 2, ep_len)
            env_params,   # parameters for our environment
            ):
        super(OfflineEnv, self).__init__()

        self.D = D
        self.shuffle = env_params['shuffle']
        self.rng = np.random.default_rng(env_params['seed'])
        if self.shuffle:
            self.rng.shuffle(self.D)
        self.N = len(D)

        self.t = 0
        self.last_reward_step = 0
        self.total_reward = 0
        self.n_episodes = 0

    def get_observation(self):
        return self.D[self.n_episodes % self.N][0][self.t]

    def get_true_action(self):
        labeled_action = self.D[self.n_episodes % self.N][1][self.t]

        # map [steer, accel, brake] back into [steer, speed] of [0, 1] range
        steering = (labeled_action[0] + 1) / 2
        speed = (labeled_action[1] - labeled_action[2] + 1) / 2
        return [steering, speed]

    def reset(self):
        self.t = 0
        self.last_reward_step = 0
        self.total_reward = 0
        self.n_episodes += 1

        if (self.n_episodes % self.N == 0) and self.shuffle:
            self.rng.shuffle(self.D)
        return self.get_observation()

    def step(self, policy):
        reward = policy.log_prob(torch.tensor(self.get_true_action())).sum()

        self.t += 1
        done = (self.t == len(self.D[self.n_episodes % self.N][0]))
        next_obs = None if done else self.get_observation()

        return next_obs, reward, done, {}

class SegmentationEnv(gym.Env):
    def __init__(
            self,
            base_env,     # base environment
            base_agent,   # base agent
            vision_core,  # vision core to process state
            online,       # whether to create an online version
            env_params,   # parameters for our environment
            ):
        super(SegmentationEnv, self).__init__()
        self.base_env = base_env
        self.vision_core = vision_core
        self.base_agent = base_agent
        self.max_regimes = env_params['max_regimes']
        self.max_seg_len = env_params['max_seg_len']
        self.online = online
        self.alpha = env_params['alpha']

        self.t = 0
        self.last_reward_step = 0
        self.total_reward = 0
        self.base_agent_cum_reward = 0
        self.base_agent_last_reward = 0
        self.n_episodes = 0

        self.c = self.max_regimes # will be forced to choose c at start
        self.segments = []
        self.ep_segments = []
        self.cs = []
        self.raw_state = self.base_env.reset()
        self.ep_states = [self.raw_state]

    def get_obs(self):
        return self.vision_core(torch.tensor(self.raw_state).unsqueeze(0))

    def get_valid_actions(self):
        # forced to choose a regime
        if self.t == 0 or (self.t - self.ep_segments[-1]) > self.max_seg_len:
            return np.array([0] + ([1] * self.max_regimes))
        else:
            return np.ones(self.max_regimes+1)

    def reset(self):
        self.t = 0
        self.last_reward_step = 0
        self.n_episodes += 1
        self.total_reward = 0

        self.base_agent_cum_reward = 0
        self.base_agent_last_reward = 0

        self.c = self.max_regimes # will be forced to choose c at start
        self.cs = []
        self.ep_segments = []
        self.segments.append(self.ep_segments)
        self.base_agent.reset()
        self.raw_state = self.base_env.reset()
        self.ep_states = [self.raw_state]

        return self.get_obs()

    def step(self, action):
        self.t += 1
        reward = 0

        if action > 0:
            self.ep_segments.append(self.t - 1)
            self.c = action - 1

            reward += self.base_agent_cum_reward - self.alpha
            self.base_agent_reward = 0

        # update base agent reward
        self.base_policy = self.base_agent.act(self.get_obs(), self.c, self.base_agent_last_reward, self.base_env.get_true_action())
        self.raw_state, self.base_agent_last_reward, done, info = self.base_env.step(self.base_policy)
        self.base_agent_reward += self.base_agent_last_reward
        self.ep_states.append(self.raw_state)
        self.cs.append(self.c)

        if done:
            next_obs = None
        else:
            next_obs = self.get_obs()

        return next_obs, reward, done, info

    def tensor_of_trajectory(self):
        vid = []
        for c, frame in zip(self.cs, self.ep_states):
            im = F.to_pil_image(frame)
            d = ImageDraw.Draw(im)
            d.text((5, 5), f'Regime: {c}', fill=(255, 0, 0))
            vid.append(F.pil_to_tensor(im))
        vid = torch.tensor(vid)  # t, h, w, c
        return vid.permute(0, 3, 1, 2).unsqueeze(0)
