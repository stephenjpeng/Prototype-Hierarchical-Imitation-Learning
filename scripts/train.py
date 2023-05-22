import numpy as np
import pickle
import time
import torch

from argparse import ArgumentParser
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from envs.segmentation_env import OfflineEnv, SegmentationEnv
from models.attention import VisionNetwork
from models.car_base_agent import CarBaseAgents
from models.car_detector_agent import CarDetectorAgent
from models.conv_vision_core import ConvVisionCore


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--max_regimes', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--env', type=str, default="car")

    parser.add_argument('--train_X_file', type=str, default="data/obs_train.pkl")
    parser.add_argument('--train_y_file', type=str, default="data/real_actions.pkl")

    # vision params
    parser.add_argument('--vision_lstm', action='store_true', help='use a vision lstm')
    parser.add_argument('--no_vision_lstm', dest='vision_lstm', action='store_false', help='don\'t use a vision lstm')
    parser.add_argument('--vision_core', type=str, default="complex", help='vision core type')

    # base agent params
    parser.add_argument('--spatial_basis_size', type=int, default=8, help="u / v for the spatial basis (sqrt of basis # channels)")
    parser.add_argument('--base_mlp_size', type=int, default=32, help="base agent mlp (a and q) size")

    # train params
    parser.add_argument('--alpha', type=float, default=0.5, help="penalty for higher segments")
    parser.add_argument('--lr', type=float, default=0.01, help="base agent learning rate")
    parser.add_argument('--update_every', type=int, default=5, help="episodes before update")
    parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")
    parser.add_argument('--max_ep_len', type=int, default=1000, help="Max frames in a segment")

    # detector params
    parser.add_argument('--max_seg_len', type=int, default=1000, help="Max frames in a segment")
    parser.add_argument('--n_layers', type=int, default=2, help="# MLP layers in detector")
    parser.add_argument('--hidden_size', type=int, default=128, help="LSTM hidden size detector")
    parser.add_argument('--v_activation', type=str, default='identity')
    parser.add_argument('--pi_activation', type=str, default='identity')
    parser.add_argument('--vision_summ', type=str, default='max', help="Pooling type for output of vision core")

    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--device', type=str)

    parser.add_argument('--tensorboard', action='store_true', help='Start a tensorboard session and write the results of training.  Only applies to training.')
    parser.add_argument('--tensorboard_suffix', type=str, default="")
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--val_every', type=int, default=50)

    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false', help="Don't shuffle offline data")
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args(args=args)

    if args.cpu:
        args.cuda = False

    args.device = 'cuda' if args.cuda else 'cpu'

    args = vars(args)
    return args


def main(args=None):
    args = parse_args(args=args)
    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)


def val_iteration(detector, base_agent, vision_core, env, args):
    detector.eval()
    base_agent.eval()
    vision_core.eval()
    print("*** VALIDATING... ***")
    with torch.no_grad():

        total_base_loss = 0
        total_detector_rewards = 0
        total_critic_loss = 0
        total_actor_loss = 0
        for episode in tqdm(range(env.base_env.N)):
            state = env.reset()
            detector.reset()
            done = False

            actions = []
            values = []
            log_probs = []
            rewards = []
            base_loss = 0
            T = 0

            # run a trajectory
            while not done:
                T += 1
                policy = detector.act(state, [[env.c]], env.get_valid_actions())
                action = policy.mode
                state, reward, done, info = env.step(action)

                actions.append(action.detach().cpu().numpy())
                log_probs.append(policy.log_prob(action).detach().cpu().numpy())
                values.append(detector.value.item())
                rewards.append(reward.item() if torch.is_tensor(reward) else reward)
                base_loss -= env.base_agent_last_reward.item()

            # calculate loss
            actions = np.array(actions)
            rewards = np.array(rewards)
            log_probs = np.array(log_probs)
            values = np.array(values)

            ## update totals
            y = rewards + args['gamma'] * values
            adv = rewards[:-1] + args['gamma'] * values[1:] - values[:-1]

            total_base_loss += base_loss / T
            total_detector_rewards += np.sum(rewards)
            total_critic_loss += np.sum(np.power(values - y, 2))
            total_actor_loss += np.dot(adv, log_probs[:-1]) / detector.num_actions

        base_loss = total_base_loss / env.base_env.N
        detector_reward = total_detector_rewards / env.base_env.N
        critic_loss = total_critic_loss / env.base_env.N
        actor_loss = total_actor_loss / env.base_env.N

    detector.train()
    base_agent.train()
    vision_core.train()
    return base_loss, detector_reward, critic_loss, actor_loss, env.tensor_of_trajectory()


def train(args):
    # Load in black box data to imitate
    with open(args['train_X_file'], 'rb') as f:
        X_train = pickle.load(f)
    with open(args['train_y_file'], 'rb') as f:
        real_actions = pickle.load(f)

    # X_train = np.array([item for sublist in X_train for item in sublist])
    # real_actions = np.array([item for sublist in real_actions for item in sublist])
    # tensor_x = torch.Tensor(X_train)
    # tensor_y = torch.tensor(real_actions, dtype=torch.float32)

    # dataset = TensorDataset(tensor_x, tensor_y)
    # generator = torch.Generator().manual_seed(args['seed'])
    # train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator)

    # split data into test and train
    dataset = list(zip(X_train, real_actions))
    N = len(dataset)
    rng = np.random.default_rng(args['seed'])
    rng.shuffle(dataset)
    split_pt = 9 * N // 10
    train_dataset = dataset[:split_pt]
    val_dataset   = dataset[split_pt:]

    #### Training
    model_name = (f'{args["env"]}_' +
                 f'{args["max_regimes"]}regimes_' +
                 f'{args["n_layers"],args["hidden_size"]}detector_' +
                 f'{args["vision_core"]}_core_' +
                 f'alpha{args["alpha"]}_' +
                 f'spatial{args["spatial_basis_size"]}_' +
                 f'{args["base_mlp_size"]}basemlpsize_' +
                 f'{args["tensorboard_suffix"]}_')
    # create models
    base_agent = CarBaseAgents(args['max_regimes'], args['base_mlp_size'], args={'device': args['device']})
    detector   = CarDetectorAgent(args)
    if args['vision_core'] == "basic":
        vision_core = ConvVisionCore(args)
    else:
        vision_core = VisionNetwork(args)

    base_agent.to(args['device'])
    detector.to(args['device'])
    vision_core.to(args['device'])

    offline_env = OfflineEnv(train_dataset, args)
    env = SegmentationEnv(offline_env, base_agent, vision_core, False, args)

    with torch.no_grad():
        val_offline_env = OfflineEnv(val_dataset, args)
        val_env = SegmentationEnv(val_offline_env, base_agent, vision_core, False, args)

    # optimizers
    base_opt = torch.optim.Adam(base_agent.parameters(), lr=args['lr'])
    detector_opt = torch.optim.Adam(detector.parameters(), lr=args['lr'])
    vision_opt = torch.optim.Adam(vision_core.parameters(), lr=args['lr'])
    b_scheduler = torch.optim.lr_scheduler.ExponentialLR(base_opt, gamma=0.97)
    d_scheduler = torch.optim.lr_scheduler.ExponentialLR(detector_opt, gamma=0.97)
    v_scheduler = torch.optim.lr_scheduler.ExponentialLR(vision_opt, gamma=0.97)

    best_reward = -float('inf')

    if args['tensorboard']:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(f'runs/{model_name}{time.strftime("%Y%m%d-%H%M%S")}')

    print("*** Starting Training... ***")
    global_step = 0
    for epoch in tqdm(range(args['num_epochs'])):
        for episode in tqdm(range(offline_env.N)):
            global_step += 1
            state = env.reset().to(args['device'])
            detector.reset()
            done = False

            ## update base agent and vision
            if (global_step - 1) % args['update_every'] == 0:
                base_agent.zero_grad()
                vision_core.zero_grad()
                base_loss = 0

                detector.zero_grad()

                actions = []
                values = []
                log_probs = []
                rewards = []
                T = 0

            # run a trajectory
            while not done:
                T += 1

                policy = detector.act(state.clone().detach(), [[env.c]], env.get_valid_actions())
                action = policy.sample()
                state, reward, done, info = env.step(action)

                actions.append(action)
                log_probs.append(policy.log_prob(action))
                values.append(detector.value)
                rewards.append(reward)
                base_loss -= env.base_agent_last_reward


            if global_step % args['update_every'] == 0:
                # update parameters
                ## update base agent and vision
                rewards = torch.tensor(rewards, requires_grad=True).to(args['device'])
                actions = torch.tensor(actions)
                log_probs = torch.tensor(log_probs, requires_grad=True).float().to(args['device'])
                values = torch.tensor(values, requires_grad=True).to(args['device'])

                base_loss /= T
                base_loss.backward()
                base_opt.step()


                ## update detector and vision core with AC
                # with torch.no_grad():
                y = rewards + args['gamma'] * values
                adv = rewards[:-1] + args['gamma'] * values[1:] - values[:-1]

                critic_loss = torch.pow(values - y, 2).sum()
                actor_loss = torch.dot(adv, log_probs[:-1].float()) / detector.num_actions
                detector_loss = critic_loss + actor_loss

                detector_loss.backward()
                detector_opt.step()
                vision_opt.step()

            # log training
            if global_step % args['log_every'] == 0 and args['tensorboard']:
                sample_trajectory = env.tensor_of_trajectory()
                writer.add_scalar('Train/BaseLoss', base_loss.item(), global_step)
                writer.add_scalar('Train/DetectorReward', rewards.sum().item(), global_step)
                writer.add_scalar('Train/CriticLoss', critic_loss.item(), global_step)
                writer.add_scalar('Train/ActorLoss', actor_loss.item(), global_step)
                writer.add_scalar('Train/LR', d_scheduler.get_last_lr(), global_step)
                writer.add_video('Train/SampleTrajectory', sample_trajectory, global_step)

            # Validation
            if global_step % args['val_every'] == 0:
                base_loss, detector_reward, val_critic_loss, val_actor_loss, sample_trajectory = val_iteration(
                    detector, base_agent, vision_core, val_env, args
                )
                if args['tensorboard']:
                    writer.add_scalar('Val/BaseLoss', base_loss, global_step)
                    writer.add_scalar('Val/DetectorReward', detector_reward, global_step)
                    writer.add_scalar('Val/CriticLoss', val_critic_loss, global_step)
                    writer.add_scalar('Val/ActorLoss', val_actor_loss, global_step)
                    writer.add_video('Val/SampleTrajectory', sample_trajectory, global_step)

                # save model if best so far
                if detector_reward > best_reward:
                    torch.save(detector.state_dict(), f'saved_models/{model_name}detector.pth')
                    torch.save(vision_core.state_dict(), f'saved_models/{model_name}vision.pth')
                    torch.save(base_agent.state_dict(), f'saved_models/{model_name}base.pth')
                    best_reward = detector_reward

        b_scheduler.step()
        d_scheduler.step()
        v_scheduler.step()


if __name__ == "__main__":
    main()
