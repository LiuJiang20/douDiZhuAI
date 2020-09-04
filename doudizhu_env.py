import gym
import numpy as np
from functools import partial
from typing import Tuple, Optional, List
import random
from tianshou.env import MultiAgentEnv
from utility import inplaceRemoveFromHand, CardType, hand_to_nparray

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
        self.current_agent = 0
        self.record = None
        self.last_agent = -1
        self.last_play_info = None
        self.log: List = []
        self.reset()

    def reset(self) -> dict:
        self.hands, landlord_cards = self.distribute_cards()
        scored_hands = [(i, sum(i)) for i in self.hands]
        scored_hands.sort(key=lambda x: x[1])
        self.hands = [i[0] for i in scored_hands]
        self.hands[0] += landlord_cards
        self.hands = [sorted(i) for i in self.hands]
        self.current_agent = 0
        self.last_agent = -1
        self.last_play_info = (CardType.UNRESTRICTED, ())
        self.record = np.zeros(shape=([3, 15]))  # records card played by each player
        self.log = []  # record proceeding of the game
        return {
            'agent_id': self.current_agent + 1,
            'obs': np.concatenate((self.record, hand_to_nparray(self.hands[self.current_agent]))),
            'mask': None,
            'info': {'last_play': self.last_play_info,
                     'agent_hand': self.hands[self.current_agent]}
        }

    def step(self, action):
        '''
        :param self:
        :param action: a tuple of cards played by the current agent
        :return:
        '''

        # Remove cards played by current agent
        cards = action[1]
        inplaceRemoveFromHand(self.hands[self.current_agent], cards)
        # update records
        for card in cards:
            self.record[self.current_agent][card - 3] += 1
        # update log
        self.log.append(action)
        if cards:
            self.last_agent = self.current_agent
            self.last_play_info = action

        new_round = DouDiZhuEnv.distance(self.last_agent, self.current_agent) == 1
        if new_round:
            self.last_play_info = (CardType.UNRESTRICTED, ())
        # Change the playing agent to next one
        self.current_agent = (self.current_agent + 1) % 3
        winner = self.get_winner()
        if winner == 0:  # landlord wins
            done = 1
            vec_rew = [1, -1, -1]
        elif winner > 0:  # peasant wins
            done = 1
            vec_rew = [-1, 1, 1]
        else:  # no winners yet
            done = 0
            vec_rew = [0, 0, 0]

        obs = {
            'agent_id': self.current_agent + 1,
            'obs': np.concatenate((self.record, hand_to_nparray(self.hands[self.current_agent]))),
            'mask': None
        }
        return obs, vec_rew, np.array(done), {'last_play': self.last_play_info,
                                              'agent_hand': self.hands[self.current_agent]}

    # Distance from b to a
    @staticmethod
    def distance(a, b):
        dis = 0
        while a != b:
            dis += 1
            b = (b + 1) % 3
        return dis

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
