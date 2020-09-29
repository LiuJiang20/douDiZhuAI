import torch
import numpy as np
from copy import deepcopy
from typing import Dict, Union, Optional

import time
from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from utility import get_available_moves, CardType, decode_hand
from doudizhu_env import DetailEnv


class AdaptedDQN(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602

    Implementation of Double Q-Learning. arXiv:1509.06461

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here)

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.
    :param int target_update_freq: the target network update frequency (``0``
        if you do not use the target network).
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 discount_factor: float = 0.99,
                 target_update_freq: int = 0,
                 reward_normalization: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0.1  # epsilon for epsilon-greedy exploration
        assert 0 <= discount_factor <= 1, 'discount_factor should in [0, 1]'
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._cnt = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode=True) -> torch.nn.Module:
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(self, buffer: ReplayBuffer,
                  indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            a = self(batch, input='obs_next', eps=0).act
            with torch.no_grad():
                target_q = self(
                    batch, model='model_old', input='obs_next').logits
            target_q = target_q[np.arange(len(a)), a]
        else:
            with torch.no_grad():
                target_q = self(batch, input='obs_next').logits.max(dim=1)[0]
        return target_q

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        """Compute the n-step return for Q-learning targets. More details can
        be found at :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_episodic_return(batch, None, self._gamma)
        return batch

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                model: str = 'model',
                input: str = 'obs',
                eps: Optional[float] = None,
                **kwargs) -> Batch:
        """Compute action over the given batch data. If you need to mask the
        action, please add a "mask" into batch.obs, for example, if we have an
        environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :param float eps: in [0, 1], for epsilon-greedy exploration method.

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = getattr(batch, input)
        # obs_ is the actual observation of the state
        obss_ = obs.obs if hasattr(obs, 'obs') else obs
        infos = obs.mask
        q_acts = [self.process_single_obs(obss_[i], infos[i], eps) for i in range(len(obss_))]
        qs = [item[0] for item in q_acts]
        acts = [item[1] for item in q_acts]
        h = None
        return Batch(logits=qs, act=acts, state=h, policy=qs)

    def process_single_obs(self, obs, info, eps=None):
        last_play = tuple(info['last_play'])  # see collector.py line 124, 273, 285 for more detail
        agent_hand = decode_hand(info['agent_hand'])
        available_moves = get_available_moves(agent_hand, last_play[0], last_play[1])
        # TODO do all calcs first
        all_obs = torch.tensor(
            np.vstack([self.gen_obs(obs, available_move) for available_move in available_moves])).float().cuda()
        # start = time.time()
        all_q = self.model(all_obs)
        # end = time.time()
        # print('time per forward: ', (end - start) * 1000)
        available_moves = [available_moves[i] + (all_q[i],) for i in range(len(available_moves))]
        available_moves.sort(key=lambda x: x[-1], reverse=True)
        if len(available_moves) == 1:
            return available_moves[0][-1], available_moves[0][:-1]
        # add eps to act
        if eps is None:
            eps = self.eps
        if not np.isclose(eps, 0):
            roll = np.random.random()
            if roll > eps:
                choose = 0
            else:
                choose = np.random.randint(1, len(available_moves))
        else:
            choose = 0
        q = all_q[choose]
        act = available_moves[choose][:-1]
        return q, act

    def gen_obs(self, obs, move):
        last_row = DetailEnv.encode_action(move)
        one_obs = np.concatenate((obs, last_row)).reshape((1, -1))
        return one_obs

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        if self._target and self._cnt % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop('weight', 1.)
        q = self(batch, eps=0.).logits
        # q = torch.tensor(q, requires_grad=True)
        # q = q[np.arange(len(q)), batch.act]
        r = to_torch_as(batch.returns, q).flatten()
        td = r - q
        loss = (td.pow(2) * weight).mean()
        batch.weight = td  # prio-buffer
        loss.backward()
        self.optim.step()
        self._cnt += 1
        return {'loss': loss.item()}


class FixedPolicy(AdaptedDQN):
    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        return {}
