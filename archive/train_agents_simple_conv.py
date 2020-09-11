from models.neural_nets import SimpleCov, get_optim
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from archive.adapted_dqn import AdaptedDQN
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from doudizhu_env import DouDiZhuEnv
from tianshou.policy import MultiAgentPolicyManager

NEURAL_NET_CLASS = SimpleCov.__name__
EXTENSION = '.pt'
LANDLORD_MODEL_PATH = 'models/landlord_model' + NEURAL_NET_CLASS + EXTENSION
PEASANT_UPPER_MODEL_PATH = 'models/peasant_upper_model' + NEURAL_NET_CLASS + EXTENSION
PEASANT_LOWER_MODEL_PATH = 'models/peasant_lower_model' + NEURAL_NET_CLASS + EXTENSION
LOG_PATH = 'models/' + NEURAL_NET_CLASS
lr = 1e-3
SEED = 42
BATCH_SIZE = 128
EPS_START = 1
EPS_TRAIN = 0.1  # epsilon for epsilon-greedy
EPS_TEST = 0.05
TRAIN_ENV_NUM = 8
TEST_ENV_NUM = 8
REPLAY_BUFFER_SIZE = 20000
COLLECT_PER_STEP = 50
STEP_PER_EPOCH = 100


def get_agents():
    landlord_model = SimpleCov().cuda()
    peasant_upper_model = SimpleCov().cuda()
    peasant_lower_model = SimpleCov().cuda()
    if os.path.isfile(LANDLORD_MODEL_PATH):
        landlord_model.load_state_dict(torch.load(LANDLORD_MODEL_PATH))
    if os.path.isfile(PEASANT_UPPER_MODEL_PATH):
        peasant_upper_model.load_state_dict(torch.load(PEASANT_UPPER_MODEL_PATH))
    if os.path.isfile(PEASANT_LOWER_MODEL_PATH):
        peasant_lower_model.load_state_dict(torch.load(PEASANT_LOWER_MODEL_PATH))

    optim = get_optim(landlord_model, lr)
    # TODO Policy need agent ID
    landlord_policy = AdaptedDQN(landlord_model, optim)
    peasant_upper_policy = AdaptedDQN(peasant_upper_model, optim)
    peasant_lower_policy = AdaptedDQN(peasant_lower_model, optim)
    landlord_policy.set_eps(EPS_TRAIN)
    peasant_upper_policy.set_eps(EPS_TRAIN)
    peasant_lower_policy.set_eps(EPS_TRAIN)
    policy = MultiAgentPolicyManager([landlord_policy, peasant_upper_policy, peasant_lower_policy])
    return policy, optim


def reward_metric(x):
    return -np.sum(np.abs(x)).astype(float)


def train_agents():
    def env_func():
        return DouDiZhuEnv()

    train_envs = SubprocVectorEnv([env_func for _ in range(TRAIN_ENV_NUM)])
    test_envs = SubprocVectorEnv([env_func for _ in range(TEST_ENV_NUM)])
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_envs.seed(SEED)
    test_envs.seed(SEED)

    policy, optim = get_agents()

    train_collector = Collector(policy, train_envs, ReplayBuffer(REPLAY_BUFFER_SIZE), reward_metric=reward_metric)
    test_collector = Collector(policy, test_envs, reward_metric=reward_metric)

    train_collector.collect(n_episode=10)

    # log
    writer = SummaryWriter(LOG_PATH)

    def save_fn(policy):
        torch.save(policy.policies[0].model.state_dict(), LANDLORD_MODEL_PATH)
        torch.save(policy.policies[1].model.state_dict(), PEASANT_UPPER_MODEL_PATH)
        torch.save(policy.policies[1].model.state_dict(), PEASANT_LOWER_MODEL_PATH)

    def train_fn(x):
        # nature DQN setting, linear decay in the first 1M steps
        now = x * COLLECT_PER_STEP * STEP_PER_EPOCH
        if now <= 1e6:
            eps = EPS_START - now / 1e6 * \
                  (EPS_START - EPS_TRAIN)
        else:
            eps = EPS_TRAIN
        for p in policy.policies:
            p.set_eps(eps)
        # print("set eps =", eps)

    def test_fn(x):
        pass

    def stop_fn(x):
        return x > -0.1

    result = offpolicy_trainer(policy, train_collector, test_collector, 10, 50, 100, TEST_ENV_NUM, BATCH_SIZE,
                               train_fn=train_fn, test_fn=test_fn,
                               save_fn=save_fn, writer=writer, test_in_train=False)
    return result


def watch():
    env = DouDiZhuEnv()
    policy, optim = get_agents()
    collector = Collector(policy, env, reward_metric=reward_metric)
    result = collector.collect(n_episode=1, render=0.1)



if __name__ == '__main__':
    train_agents()
    # watch()
