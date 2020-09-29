from adapted_random_policy import AdaptedRandomPolicy
from doudizhu_env import ResultCollector, DetailEnv
from tianshou.env import DummyVectorEnv
import numpy as np
from tianshou.data import Collector
from neural_nets import FullNet, get_optim, SmallFullNet
from tianshou.policy import MultiAgentPolicyManager
from adpted_dqn_full import AdaptedDQN
import argparse
from cdqn import CDQN
from doudizhuc.scripts.agents import make_agent
import torch


def reward_metric(x):
    return x[0]


def test_ten_times(net_name: str, net_path):
    if net_name == 'full':
        net = FullNet().cuda()
    elif net_name == 'small':
        net = SmallFullNet().cuda()
    elif net_name == 'cdqn':
        pass
    else:
        assert False, 'Network name doesn\'t match, please specify a correct network'

    if net_name == 'cdqn':
        landlord_policy = CDQN(make_agent('CDQN', 2))
    else:
        net.load_state_dict(torch.load(net_path))
        optim = get_optim(net, lr=10e-3)
        landlord_policy = AdaptedDQN(net, optim)
    upper = AdaptedRandomPolicy(None, None)
    lower = AdaptedRandomPolicy(None, None)
    test_results = []
    for i in range(30):
        result_collector = ResultCollector()
        env = DummyVectorEnv([lambda: DetailEnv(result_collector) for _ in range(1)])
        policy = MultiAgentPolicyManager([landlord_policy, upper, lower])
        for p in policy.policies:
            p.set_eps(0)
        collector = Collector(policy, env, reward_metric=reward_metric)
        result = collector.collect(n_episode=100)
        env_result = result_collector.get_result()
        land_lord_win = len([i for i in env_result if i == 0])
        test_results.append(land_lord_win)
    print(test_results)
    print('mean', np.mean(test_results))
    print('std', np.std(test_results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network')
    parser.add_argument('--netpath', default=None)
    args = parser.parse_args()
    test_ten_times(args.network, args.netpath)
