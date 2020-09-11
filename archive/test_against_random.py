from models.neural_nets import SimpleCov, get_optim
import os
import numpy as np
import torch
from archive.adapted_dqn import AdaptedDQN
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector
from doudizhu_env import DouDiZhuEnv, ResultCollector, DetailEnv
from adapted_random_policy import AdaptedRandomPolicy
from tianshou.policy import MultiAgentPolicyManager

NEURAL_NET_CLASS = SimpleCov.__name__
EXTENSION = '.pt'
LANDLORD_MODEL_PATH = 'models/landlord_model' + NEURAL_NET_CLASS + EXTENSION
PEASANT_UPPER_MODEL_PATH = 'models/peasant_upper_model' + NEURAL_NET_CLASS + EXTENSION
PEASANT_LOWER_MODEL_PATH = 'models/peasant_lower_model' + NEURAL_NET_CLASS + EXTENSION
LOG_PATH = 'models/' + NEURAL_NET_CLASS
lr = 1e-3
SEED = 42

TRAIN_ENV_NUM = 1
TEST_ENV_NUM = 8
REPLAY_BUFFER_SIZE = 20000


def get_agents(random_landlord=False, random_peasant=False):
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
    landlord_policy = AdaptedDQN(landlord_model, optim) if not random_landlord else AdaptedRandomPolicy(None, None)
    peasant_upper_policy = AdaptedDQN(peasant_upper_model, optim) if not random_peasant else AdaptedRandomPolicy(None,
                                                                                                                 None)
    peasant_lower_policy = AdaptedDQN(peasant_lower_model, optim) if not random_peasant else AdaptedRandomPolicy(None,
                                                                                                                 None)
    landlord_policy.set_eps(0.01)
    peasant_upper_policy.set_eps(0.01)
    peasant_lower_policy.set_eps(0.01)
    policy = MultiAgentPolicyManager([landlord_policy, peasant_upper_policy, peasant_lower_policy])
    return policy, optim


def reward_metric(x):
    return np.var(x).astype(float)


def watch():
    env = DouDiZhuEnv()
    policy, optim = get_agents()
    collector = Collector(policy, env, reward_metric=reward_metric)
    result = collector.collect(n_episode=1, render=0.1)


def test_against_random_agent():
    # result_collector = ResultCollector()
    # env = DummyVectorEnv([lambda: DouDiZhuEnv(result_collector) for _ in range(1)])
    # policy, optim = get_agents(random_peasant=True)
    # collector = Collector(policy, env, reward_metric=reward_metric)
    # result = collector.collect(n_episode=50)
    # env_result = result_collector.get_result()
    # land_lord_win = len([i for i in env_result if i == 0])
    # print('landlord win rate:', land_lord_win / len(env_result))
    #
    # policy, optim = get_agents(random_landlord=True)
    # collector = Collector(policy, env, reward_metric=reward_metric)
    # result = collector.collect(n_episode=50)
    # env_result = result_collector.get_result()
    # peasant_win_rate = len([i for i in env_result if i != 0])
    # print('peasant win rate:', peasant_win_rate / len(env_result))
    #
    # policy, optim = get_agents(random_landlord=True, random_peasant=True)
    # collector = Collector(policy, env, reward_metric=reward_metric)
    # result = collector.collect(n_episode=50)
    # env_result = result_collector.get_result()
    # peasant_win_rate = len([i for i in env_result if i != 0])
    # print('landlord win rate:', 1 - peasant_win_rate / len(env_result))
    # print('peasant win rate:', peasant_win_rate / len(env_result))

    result_collector = ResultCollector()
    env = DummyVectorEnv([lambda: DetailEnv(result_collector) for _ in range(1)])
    policy, optim = get_agents(random_peasant=True)
    collector = Collector(policy, env, reward_metric=reward_metric)
    # collector.collect(n_episode=1, render=0.1)
    collector.collect(n_episode=1)


if __name__ == '__main__':
    test_against_random_agent()
