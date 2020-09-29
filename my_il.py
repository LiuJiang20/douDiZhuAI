from os import path
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import SACPolicy, ImitationPolicy, MultiAgentPolicyManager
from tianshou.utils.net.continuous import Actor, ActorProb, Critic
from doudizhuc.scripts.agents import make_agent
from cdqn import CDQN
import cdqn_all
from neural_nets import get_optim, SmallFullNetWithDropout, FullNet, LargeNet
from my_imitation_policy import MyImitationPolicy
from doudizhu_env import DetailEnv, LandlordEnv
import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--il-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=2400)
    # parser.add_argument('--step-per-epoch', type=int, default=1200)
    parser.add_argument('--collect-per-step', type=int, default=10)
    # parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    # parser.add_argument('--training-num', type=int, default=1)
    # parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--ignore-done', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=4)
    parser.add_argument('--network', type=str)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


args = get_args()


def get_agents():
    peasant_upper_model = make_agent('CDQN', 1)
    peasant_lower_model = make_agent('CDQN', 3)
    landlord_model = make_agent('CDQN', 2)
    # peasant_upper_model = None
    # peasant_lower_model = None
    # TODO Policy need agent ID
    landlord_policy = cdqn_all.CDQNAll(landlord_model)
    peasant_upper_policy = CDQN(peasant_upper_model)
    peasant_lower_policy = CDQN(peasant_lower_model)
    # policy = MultiAgentPolicyManager([landlord_policy, peasant_lower_policy, peasant_upper_policy])
    return landlord_policy, peasant_upper_policy, peasant_lower_policy


def reward_metrix(x):
    return x[0]


def train(net, save_path):
    landlord_path = save_path
    landlord_policy, upper_policy, lower_policy = get_agents()
    train_envs = DummyVectorEnv(
        [lambda: LandlordEnv(upper_policy, lower_policy) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv(
        [lambda: LandlordEnv(upper_policy, lower_policy) for _ in range(args.test_num)])
    # train_envs = SubprocVectorEnv(
    #     [lambda: LandlordEnv(upper_policy, lower_policy) for _ in range(args.training_num)])
    # test_envs = SubprocVectorEnv(
    #     [lambda: LandlordEnv(upper_policy, lower_policy) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # collect expert experiences
    train_collector = Collector(
        landlord_policy, train_envs, ReplayBuffer(args.buffer_size))
    # either create or load network
    optim = get_optim(net, args.il_lr)

    # build test policy
    il_policy = MyImitationPolicy(net, optim)
    il_test_collector = Collector(il_policy, test_envs)
    train_collector.reset()
    # il_test_collector.collect(n_episode=1)

    # writer
    log_path = path.join('models', 'landlord_il')
    writer = SummaryWriter(log_path)

    def stop_fn(x):
        return x > -0.2

    def save_fn(policy):
        torch.save(policy.model.state_dict(), landlord_path)

    result = offpolicy_trainer(
        il_policy, train_collector, il_test_collector, args.epoch,
        args.step_per_epoch // 5, args.collect_per_step, args.test_num,
        args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer)
    print(f'Final reward: {result["rew"]}, length: {result["len"]}')


if __name__ == '__main__':
    if args.network is None:
        sys.stderr.write('No network choice is specified, choose default full')
        args.network = 'full'
    if args.network == 'small':
        landlord_path = 'il_landlord.pt'
        net = SmallFullNetWithDropout().cuda()
        if path.exists(landlord_path):
            net.load_state_dict(torch.load(landlord_path))
        train(net, landlord_path)
    elif args.network == 'full':
        landlord_path = 'il_landlord_full.pt'
        net = FullNet().cuda()
        if path.exists(landlord_path):
            net.load_state_dict(torch.load(landlord_path))
        train(net, landlord_path)
    elif args.network == 'large':
        landlord_path = 'il_landlord_large.pt'
        net = LargeNet().cuda()
        if path.exists(landlord_path):
            net.load_state_dict(torch.load(landlord_path))
        train(net, landlord_path)
