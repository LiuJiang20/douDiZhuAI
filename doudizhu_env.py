import gym
import numpy as np
from functools import partial
from typing import Tuple, Optional
import random
from tianshou.env import MultiAgentEnv
from utility import inplaceRemoveFromHand

'''
Representation(number->card):
(3->"3")
(4->"4")
(5->"5")
(6->"6")
(7->"7")
(8->"8")
(9->"9")
(10->"10")
(11->"Jack")
(12->"Queen")
(13->"King")
(14->"Ace")
(15->"2")
(16->"Black Joker")
(17->"Red Joker")
(->)
(->)
(->)
'''
cardMap = {3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10", 11: "J", 12: "Q",
           13: "K", 14: "A", 15: "2", 16: "B", 17: "R"}
fullDeck = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * 4 + [16, 17]


class DouDiZhuEnv(MultiAgentEnv):


'''
    self.hands: hands for each player in sequence (landlord, upper peasent, lower peasant)
'''


def __init__(self):
    super().__init__()
    self.hands = None
    self.current_agent = -1
    self.reset()


def reset(self) -> dict:
    self.hands, landlord_cards = self.distribute_cards()
    scored_hands = [(i, sum(i)) for i in self.hands]
    scored_hands.sort(key=lambda x: x[1])
    self.hands = [i[0] for i in scored_hands]
    self.hands[0] += landlord_cards
    self.hands = [sorted[i] for i in self.hands]
    self.current_agent = 0
    return {
        'agent_id': self.current_agent,
        'obs': None,
        'mask': None
    }


def step(self, action) -> Tuple[dict, np.ndarray, np.ndarray, dict]:
    '''
    :param self:
    :param action: a tuple of cards played by the current agent
    :return:
    '''
    inplaceRemoveFromHand(self.hands)
    obs = {
        'agent_id': self.current_agent,
        'obs': None,
        'mask': None
    }

    return obs, vec_rew, np.array(done), {}


def distribute_cards(self):
    '''
    Randomly assign 17 cards for each player, hold the last three for the landlord
    :return: None
    '''
    shuffled = fullDeck.copy()
    random.shuffle(shuffled)
    return [shuffled[:17], shuffled[17:34], shuffled[34:-3]], shuffled[-3:]


def get_winner(self):
    for i in range(len(self.hands)):
        if not self.hands[i]:
            return i
    return -1


def seed(self, seed: Optional[int] = None) -> int:
    pass


def render(self, **kwargs) -> None:
    pass


def close(self) -> None:
    pass
