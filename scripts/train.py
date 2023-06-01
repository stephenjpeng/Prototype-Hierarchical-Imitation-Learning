import numpy as np
import pickle
import time
import torch
import torchvision
import wandb

from argparse import ArgumentParser
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

from envs.segmentation_env import OfflineEnv, OnlineEnv, SegmentationEnv
from games.carracing import RacingNet
from models.attention import VisionNetwork
from models.car_base_agent import CarBaseAgents, BasicCarBaseAgents
from models.car_detector_agent import CarDetectorAgent
from models.conv_vision_core import ConvVisionCore


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--max_regimes', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=100)

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--env', type=str, default="car")

    # train params
    parser.add_argument('--train_X_file', type=str, default="data/obs_train_trimmed.pkl")
    parser.add_argument('--train_y_file', type=str, default="data/real_actions_trimmed.pkl")
    parser.add_argument('--mean_image_file', type=str, default="data/mean_image.png")

    # val params
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--num_iterations', type=int, default=5)

    # vision params
    parser.add_argument('--reward_boost', type=float, default=0.0, help="Boost reward for completed segments")
    parser.add_argument('--allow_same_regime', action='store_true', help='max_seg_len forces a regime switch')
    parser.add_argument('--shift_rewards', action='store_true', help='Shift rewards back to the action that caused it')
    parser.add_argument('--no_sparse_rewards', dest='sparse_rewards', action='store_false', help='Sparse rewards not only when switching regimes')

    # vision params
    parser.add_argument('--vision_lstm', action='store_true', help='use a vision lstm')
    parser.add_argument('--no_vision_lstm', dest='vision_lstm', action='store_false', help='don\'t use a vision lstm')
    parser.add_argument('--vision_core', type=str, default="complex", help='vision core type')

    # base agent params
    parser.add_argument('--spatial_basis_size', type=int, default=8, help="u / v for the spatial basis (sqrt of basis # channels)")
    parser.add_argument('--base_mlp_size', type=int, default=32, help="base agent mlp (a and q) size")
    parser.add_argument('--base_mlp_depth', type=int, default=2, help="base agent mlp hidden layers")
    parser.add_argument('--c_k', type=int, default=8, help="size of keys")
    parser.add_argument('--num_queries_per_agent', type=int, default=2, help="# of queries per agent")
    parser.add_argument('--base_agent', type=str, default="complex", help="complex or basic base agent")
    parser.add_argument('--base_agent_c', type=str, default="probs", help="probs or single")
    parser.add_argument('--limit_attention', action='store_true', help='Limit attention mechanism')
    parser.add_argument('--rbf_limit', action='store_true', help='Limit attention mechanism with an RBF-like mask')
    parser.add_argument('--base_weight', type=float, default=0.7, help="Weight of time-invariant attention")

    # train params
    parser.add_argument('--alpha', type=float, default=0.5, help="penalty for higher segments")
    parser.add_argument('--lr', type=float, default=0.01, help="base agent learning rate")
    parser.add_argument('--update_every', type=int, default=5, help="episodes before update")
    parser.add_argument('--gamma', type=float, default=0.99, help="discount factor")
    parser.add_argument('--max_ep_len', type=int, default=1000, help="Max frames in a segment")
    parser.add_argument('--dagger', action="store_true", help="Use DAgger for training")
    parser.add_argument('--dagger_sweep', type=str, default=None, help="True if not none")
    parser.add_argument('--expert_file', type=str, default="weights/agent_weights.pt")
    parser.add_argument('--dagger_num_trajectories', type=int, default=180, help="Number of trajectories to sample with DAgger")
    parser.add_argument('--initial_train_len', type=int, default=1000, help="Number of trajectories to start training with")

    # detector params
    parser.add_argument('--max_seg_len', type=int, default=1000, help="Max frames in a segment")
    parser.add_argument('--n_layers', type=int, default=2, help="# MLP layers in detector")
    parser.add_argument('--hidden_size', type=int, default=128, help="LSTM hidden size detector")
    parser.add_argument('--v_activation', type=str, default='identity')
    parser.add_argument('--pi_activation', type=str, default='identity')
    parser.add_argument('--vision_summ', type=str, default='max', help="Pooling type for output of vision core")
    parser.add_argument('--regime_encoding', type=str, default='none', help="Encoding of regime")

    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--device', type=str)

    parser.add_argument('--tensorboard', action='store_true', help='Start a tensorboard session and write the results of training.  Only applies to training.')
    parser.add_argument('--tensorboard_suffix', type=str, default="")
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--val_every', type=int, default=50)

    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false', help="Don't shuffle offline data")
    # parser.add_argument('--no_crop', dest='crop_info', action='store_false', help="crop the state")
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args(args=args)

    if args.cpu:
        args.cuda = False

    args.device = 'cuda' if args.cuda else 'cpu'

    args = vars(args)

    try:
        args['mean_image'] = torch.tensor(np.asarray(Image.open(args['mean_image_file']))).to(args['device'])
    except:
        args['mean_image'] = None

    args['dagger'] = args['dagger_sweep'] is not None

    return args


def main(args=None):
    args = parse_args(args=args)
    if args['mode'] == 'train':
        try:
            train(args)
        except KeyboardInterrupt:
            if args['tensorboard']:
                wandb.finish()
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
                policy = detector.act(state, env.get_regime().detach(), env.get_valid_actions())
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
    return base_loss, detector_reward, critic_loss, actor_loss, env.tensor_of_trajectory(), env.cs


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
    rng = np.random.default_rng(args['seed'])
    rng.shuffle(dataset)
    dataset = dataset[:args['initial_train_len']]  # only use a subset of the offline data for DAgger
    N = len(dataset)
    split_pt = 9 * N // 10
    train_dataset = dataset[:split_pt]
    val_dataset   = dataset[split_pt:]

    ## Calculate mean image
    if args['mean_image'] is None:
        images = np.array([
            frame for ep in train_dataset for frame in ep[0]
            ])
        args['mean_image'] = np.mean(images, axis=0).astype('uint8')
        Image.fromarray(args['mean_image']).save(args['mean_image_file'])
        args['mean_image'] = torch.tensor(args['mean_image']).to(args['device'])

    #### Training
    model_name = (f'{args["env"]}/' +
                 f'{args["max_regimes"],args["base_agent"]}agents/' +
                 f'{args["vision_core"]}{"_lstm" if args["vision_lstm"] else ""}_core/' +
                 f'{args["n_layers"],args["hidden_size"]}detector/' +
                 f'({args["base_mlp_depth"]}x{args["base_mlp_size"]})basemlp/' +
                 f'spatial{args["spatial_basis_size"]}/' +
                 f'{args["c_k"]}ck/' +
                 f'{args["lr"]}lr/' +
                 f'alpha{args["alpha"]}/' +
                 f'dagger:{args["dagger"]}/' +
                 f'{args["tensorboard_suffix"]}_')
    # create models
    if args["base_agent"] == "basic":
        base_agent = BasicCarBaseAgents(args['max_regimes'], args['base_mlp_size'], args=args)
    else:
        base_agent = CarBaseAgents(args['max_regimes'], args['base_mlp_size'], args=args)
    detector   = CarDetectorAgent(args)
    if args['vision_core'] == "basic":
        vision_core = ConvVisionCore(args)
    else:
        vision_core = VisionNetwork(args)

    if args['dagger']:
        expert = RacingNet((4, 96, 96), [2])
        expert.to(args['device'])
        expert.load_state_dict(torch.load(args['expert_file'], map_location=args['device']))
        expert.eval()
    else:
        expert = None

    base_agent.to(args['device'])
    detector.to(args['device'])
    vision_core.to(args['device'])

    offline_env = OfflineEnv(train_dataset, args)
    env = SegmentationEnv(offline_env, base_agent, vision_core, False, args)

    # for validation
    with torch.no_grad():
        val_offline_env = OfflineEnv(val_dataset, args)
        val_args = args.copy()
        val_args.update({'alpha': 0, 'base_agent_c': 'single'}) # no penalty for switching in the real world
        val_env = SegmentationEnv(val_offline_env, base_agent, vision_core, False, val_args)

        online_env = OnlineEnv()
        val_online_env = SegmentationEnv(online_env, base_agent, vision_core, True,
                val_args, args['dagger'], expert)

    # optimizers
    base_opt = torch.optim.Adam(base_agent.parameters(), lr=args['lr'])
    detector_opt = torch.optim.Adam(detector.parameters(), lr=args['lr'])
    vision_opt = torch.optim.Adam(vision_core.parameters(), lr=args['lr'])
    b_scheduler = torch.optim.lr_scheduler.ExponentialLR(base_opt, gamma=0.97)
    d_scheduler = torch.optim.lr_scheduler.ExponentialLR(detector_opt, gamma=0.97)
    v_scheduler = torch.optim.lr_scheduler.ExponentialLR(vision_opt, gamma=0.97)

    best_reward = -float('inf')

    if args['tensorboard']:
        wandb.init(
                project='cs224r',
                config=args,
                sync_tensorboard=True
        )
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
                log_probs = []
                rewards = []
                targets = []
                advs    = []
                T = 0

            ep_values = []
            # run a trajectory
            while not done:
                T += 1

                policy = detector.act(state.clone().detach(), env.get_regime().detach(), env.get_valid_actions())
                action = policy.sample()
                state, reward, done, info = env.step(action, policy.probs[0])

                actions.append(action)
                log_probs.append(policy.log_prob(action))
                ep_values.append(detector.value)
                rewards.append(reward)
                base_loss -= env.base_agent_last_reward

            # compute targets
            # TODO: CONSIDER N-STEP RETURNS HERE
            with torch.no_grad():
                ep_values = torch.tensor(ep_values + [0])
                y = torch.tensor(env.ep_rewards) + args['gamma'] * ep_values[1:]
                adv  = y - ep_values[:-1]

                targets.append(y)
                advs.append(adv)

            # last_value = values[-1]
            # values = values[:-1]  # remove last value
            # log_probs = log_probs[:-1]  # remove last log_prob

            # compute targets
            # q = reward.detach().item()
            # with torch.no_grad():
            #     y = torch.zeros(len(env.ep_rewards) - 1)
            #     for i in reversed(range(len(env.ep_rewards) - 1)):
            #         q = env.ep_rewards[i] + args['gamma'] * q
            #         y[i] = q
            #     ys.append(y)

            if global_step % args['update_every'] == 0:
                # update parameters
                ## update base agent and vision
                rewards = torch.tensor(rewards, requires_grad=True).to(args['device'])
                targets = torch.cat(targets).to(args['device'])
                advs = torch.cat(advs).to(args['device'])
                # ys = torch.cat(ys)
                actions = torch.tensor(actions)
                log_probs = torch.tensor(log_probs, requires_grad=True).float().to(args['device'])
                # values = torch.tensor(values, requires_grad=True).to(args['device'])

                base_loss /= T
                base_loss.backward()
                base_opt.step()


                ## update detector and vision core with AC
                # with torch.no_grad():
                # y = rewards[:-1] + args['gamma'] * values[1:]
                # adv = rewards[:-1] + args['gamma'] * values[1:] - values[:-1]
                # adv = ys - values
                critic_loss = torch.pow(advs, 2).mean()
                # check negative here?
                actor_loss = torch.dot(advs, log_probs.float()) / detector.num_actions
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
                base_loss, detector_reward, val_critic_loss, val_actor_loss, sample_trajectory, histogram = val_iteration(
                    detector, base_agent, vision_core, val_env, args
                )
                mean_online, std_online, trajectories = run_online_trajectory(
                        val_online_env, detector, model_name, args['num_iterations'], args['device']
                )
                if args['tensorboard']:
                    writer.add_scalar('Val/BaseLoss', base_loss, global_step)
                    writer.add_scalar('Val/DetectorReward', detector_reward, global_step)
                    writer.add_scalar('Val/CriticLoss', val_critic_loss, global_step)
                    writer.add_scalar('Val/ActorLoss', val_actor_loss, global_step)
                    writer.add_scalar('Val/MeanOnlineReward', mean_online, global_step)
                    writer.add_scalar('Val/StdOnlineReward', std_online, global_step)
                    writer.add_video('Val/SampleOfflineTrajectory', sample_trajectory, global_step)
                    writer.add_video('Val/SampleOnlineTrajectory',
                            torch.cat(trajectories).unsqueeze(0).permute(0, 1, 4, 2, 3), global_step)
                    writer.add_histogram('Val/RegimeSample', histogram, global_step)

                # save model if best so far
                if detector_reward > best_reward:
                    torch.save(detector.state_dict(),
                            f'saved_models/{model_name.replace("/", "_")}detector.pth')
                    torch.save(vision_core.state_dict(),
                            f'saved_models/{model_name.replace("/", "_")}vision.pth')
                    torch.save(base_agent.state_dict(),
                            f'saved_models/{model_name.replace("/", "_")}base.pth')
                    best_reward = detector_reward

        b_scheduler.step()
        d_scheduler.step()
        v_scheduler.step()

        # Collect DAgger trajectories
        if args['dagger']:
            new_data = run_online_trajectory(
                        val_online_env,
                        detector,
                        args['model_name'],
                        args['dagger_num_trajectories'],
                        args['device'],
                        dagger=True
                    )
            offline_env.add_data(new_data)


def run_online_trajectory(env, detector, model_name, num_iter, device, dagger=False):
    detector.eval()
    env.vision_core.eval()
    env.base_agent.eval()

    if dagger:
        print("*** Starting DAgger collection... ***")
        states = []
        expert_actions = []
    else:
        print("*** Starting Online Evaluation... ***")
        ep_rewards = []
        trajectories = []

    global_step = 0
    for iteration in tqdm(range(num_iter)):
        global_step += 1
        state = env.reset().to(device)
        detector.reset()
        done = False

        ep_reward = 0
        T = 0

        # run a trajectory
        while not done:
            T += 1

            policy = detector.act(state.clone().detach(), env.get_regime(), env.get_valid_actions())
            action = policy.mode
            state, reward, done, info = env.step(action)

            ep_reward += reward.item() if torch.is_tensor(reward) else reward

        if dagger:
            if len(env.expert_actions) > 128:
                start_pt = np.random.randint(0, len(env.expert_actions) - 128)
                states.append(env.ep_states[:-1][start_pt:start_pt+128])
                expert_actions.append(env.expert_actions[start_pt:start_pt+128])
            else:
                states.append(env.ep_states[:-1])
                expert_actions.append(env.expert_actions)
        else:
            ep_rewards.append(ep_reward)
            trajectories.append(env.tensor_of_trajectory().squeeze(0).permute(0, 2, 3, 1))

    if dagger:
        dagger_data = list(zip(states, expert_actions))
    else:
        ep_rewards = np.array(ep_rewards)
        print(f"Average reward (N={num_iter}): {np.mean(ep_rewards)}")
        print(f"Std Dev reward (N={num_iter}): {np.std(ep_rewards)}")

    detector.train()
    env.vision_core.train()
    env.base_agent.train()

    if dagger:
        return dagger_data
    else:
        return np.mean(ep_rewards), np.std(ep_rewards), trajectories


def evaluate(args):
    # create models
    if args["base_agent"] == "basic":
        base_agent = BasicCarBaseAgents(args['max_regimes'], args['base_mlp_size'], args=args)
    else:
        base_agent = CarBaseAgents(args['max_regimes'], args['base_mlp_size'], args=args)
    detector   = CarDetectorAgent(args)
    if args['vision_core'] == "basic":
        vision_core = ConvVisionCore(args)
    else:
        vision_core = VisionNetwork(args)

    base_agent.to(args['device'])
    detector.to(args['device'])
    vision_core.to(args['device'])

    # Load in models
    model_name = args['model_name']
    detector.load_state_dict(torch.load(f'saved_models/{model_name}detector.pth'))
    vision_core.load_state_dict(torch.load(f'saved_models/{model_name}vision.pth'))
    base_agent.load_state_dict(torch.load(f'saved_models/{model_name}base.pth'))

    online_env = OnlineEnv()
    args.update({'alpha': 0, 'base_agent_c': 'single'}) # no penalty for switching in the real world
    env = SegmentationEnv(online_env, base_agent, vision_core, True, args)

    _, _, trajectories = run_online_trajectory(env, detector, model_name, args['num_iterations'], args['device'])
    for i, trajectory in enumerate(trajectories):
        torchvision.io.write_video(f'videos/{model_name}_ep{i+1}.mp4',
                trajectory,
                fps=10)


if __name__ == "__main__":
    main()
