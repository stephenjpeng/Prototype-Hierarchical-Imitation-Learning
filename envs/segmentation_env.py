import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import torch
import torchvision.transforms.functional as F


from copy import deepcopy
from gym.spaces import Box
from PIL import Image
from PIL import ImageDraw
from skimage.transform import resize


class OfflineEnv(gym.Env):
    def __init__(
            self,
            D,            # dataset of trajectories (N, 2, ep_len)
            env_params,   # parameters for our environment
            ):
        super(OfflineEnv, self).__init__()

        self.device = env_params['device']
        self.D = D
        self.max_ep_len = env_params['max_ep_len']
        self.shuffle = env_params['shuffle']
        # self.crop_info = env_params['crop_info']
        self.rng = np.random.default_rng(env_params['seed'])
        if self.shuffle:
            self.rng.shuffle(self.D)
        self.N = len(D)

        self.t = 0
        self.last_reward_step = 0
        self.total_reward = 0
        self.n_episodes = 0


    def get_observation(self):
        state = self.D[self.n_episodes % self.N][0][self.t]
        # if self.crop_info:
        #     orig_shape = state.shape
        #     state = resize(state[:-15, :, :], orig_shape)
        return state

    def get_true_action(self):
        labeled_action = self.D[self.n_episodes % self.N][1][self.t]

        # map [steer, accel, brake] back into [steer, speed] of [-1, 1] range
        steering = (labeled_action[0])
        speed = (labeled_action[1] - labeled_action[2])
        return [steering, speed]

    def reset(self):
        self.t = 0
        self.last_reward_step = 0
        self.total_reward = 0
        self.n_episodes += 1

        if (self.n_episodes % self.N == 0) and self.shuffle:
            self.rng.shuffle(self.D)
        return self.get_observation()

    def step(self, action):
        reward = -torch.nn.functional.mse_loss(
            action, torch.tensor(self.get_true_action()).to(self.device).unsqueeze(0)
        )

        self.t += 1
        done = (self.t > self.max_ep_len) or (self.t == len(self.D[self.n_episodes % self.N][0]))
        next_obs = None if done else self.get_observation()

        return next_obs, reward, done, {}


class OnlineEnv(gym.Wrapper):
    def __init__(self, frame_skip=0, seed=2023):
        self.seed = seed
        self.env = gym.make("CarRacing-v1")
        super().__init__(self.env)
        self.env.seed(seed)

        self.action_space = Box(low=0, high=1, shape=(2,))
        self.observation_space = Box(low=0, high=1, shape=(96, 96))

        self.t = 0
        self.last_reward_step = 0
        self.total_reward = 0
        self.n_episodes = 0
        self.processed_frame = None

        self.frame_skip = frame_skip


    def preprocess(self, original_action):
        action = np.zeros(3)
        original_action = original_action.squeeze(0).detach().cpu().numpy()

        action[0] = original_action[0]

        # Separate acceleration and braking
        action[1] = max(0, original_action[1])
        action[2] = max(0, -original_action[1])

        return action

    def postprocess(self, original_observation):
        # crop and resize
        observation = (255 * resize(original_observation[:-15, :, :], (96, 96))).astype('uint8')
        return observation

    def shape_reward(self, reward):
        return np.clip(reward, -1, 1)

    def get_observation(self):
        if self.processed_frame is None:
            self.processed_frame = self.postprocess(self.frame)
        return self.processed_frame

    def reset(self):
        self.t = 0
        self.last_reward_step = 0
        self.n_episodes += 1
        self.total_reward = 0

        self.frame = self.env.reset()
        self.processed_frame = None
        self.env.seed(self.seed + self.n_episodes)

        return self.get_observation()

    def step(self, action, real_action=False):
        self.t += 1

        if not real_action:
            action = self.preprocess(action)

        total_reward = 0
        for _ in range(self.frame_skip + 1):
            new_frame, reward, done, info = self.env.step(action)
            self.total_reward += reward
            reward = self.shape_reward(reward)
            total_reward += reward

            if reward > 0:
                self.last_reward_step = self.t

        if self.t - self.last_reward_step > 30:
            done = True

        reward = total_reward / (self.frame_skip + 1)


        self.frame = new_frame

        return self.get_observation(), reward, done, info


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

        self.base_agent.reset()
        self.vision_core.reset()

        self.max_regimes = env_params['max_regimes']
        self.max_seg_len = env_params['max_seg_len']
        self.device = env_params['device']
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
        self.state = None
        self.ep_states = [self.raw_state]
        self.ep_rewards = []
        self.ep_attns = []

    def get_obs(self):
        if self.state is None:
            self.state = self.vision_core(torch.tensor(self.raw_state).float().to(self.device).unsqueeze(0))
        return self.state

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
        self.vision_core.reset()
        self.raw_state = self.base_env.reset()
        self.state = None
        self.ep_states = [self.raw_state]
        self.ep_rewards = []
        self.ep_attns = []

        return self.get_obs()

    def step(self, action):
        self.t += 1
        reward = 0

        if action > 0:
            self.ep_segments.append(self.t - 1)
            self.c = action - 1

            reward += self.base_agent_cum_reward - self.alpha
            self.base_agent_cum_reward = 0

        # update base agent reward
        # if self.online:
        last_action = self.base_agent.action.clone().detach() if self.base_agent.action is not None else None
        self.base_policy = self.base_agent.act(self.get_obs(), self.c, self.base_agent_last_reward, last_action)
        # else:
        #     self.base_policy = self.base_agent.act(self.get_obs(), self.c, self.base_agent_last_reward, self.base_env.get_true_action())
        self.raw_state, self.base_agent_last_reward, done, info = self.base_env.step(self.base_policy)
        self.state = None

        self.base_agent_cum_reward += self.base_agent_last_reward
        self.ep_attns.append(self.base_agent.A)
        self.ep_states.append(self.raw_state)
        self.cs.append(self.c.item())

        if done:
            next_obs = None
            if action == 0: # force reward update even if no action
                self.ep_segments.append(self.t - 1)
                self.c = self.max_regimes

                reward += self.base_agent_cum_reward - self.alpha
                self.base_agent_cum_reward = 0
        else:
            next_obs = self.get_obs()

        self.ep_rewards.append(reward)

        return next_obs, reward, done, info

    def tensor_of_trajectory(self):
        vid = []
        for c, frame, attn in zip(self.cs, self.ep_states, self.ep_attns):
            # one frame for each head
            frame = np.tile(frame, [1, self.base_agent.num_queries_per_agent, 1])
            frame = F.to_pil_image(frame)

            # adjust attn to RGB
            attn = attn.squeeze(0).permute(2, 0, 1)
            for i in range(self.base_agent.num_queries_per_agent):
                attn[i] = (attn[i] - attn[i].min()) / (attn[i].max() - attn[i].min())
            attn = torch.hstack([*attn]).unsqueeze(0)
            attn = Image.fromarray(
                    (plt.get_cmap('jet')(
                        attn.cpu().numpy().squeeze())[:, :, :3] * 255
                    ).astype(np.uint8))
            attn = attn.resize(frame.size)

            im = Image.blend(frame, attn, 0.6)
            d = ImageDraw.Draw(im)
            d.text((5, 5), f'Regime: {c}', fill=(255, 0, 0))
            vid.append(F.pil_to_tensor(im))
        vid = torch.stack(vid)  # t, h, w, c
        return vid.unsqueeze(0)
