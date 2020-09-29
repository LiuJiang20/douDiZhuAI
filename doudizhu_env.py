import gym
import numpy as np
from functools import partial
from typing import Tuple, Optional, List
import random
from tianshou.env import MultiAgentEnv
from utility import inplaceRemoveFromHand, CardType, hand_to_nparray, encode_hand
from threading import Lock
from collections import Counter
from doudizhuc.scripts.agents import make_agent

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
fullDeck.sort()


class ResultCollector:
    def __init__(self):
        self.lock = Lock()
        self.results = []

    def get_result(self):
        return self.results

    def add(self, result):
        self.lock.acquire()
        self.results.append(result)
        self.lock.release()


class DouDiZhuEnv(MultiAgentEnv):
    '''
        self.hands: hands for each player in sequence (landlord, upper peasent, lower peasant)
    '''

    def __init__(self, result_collector: ResultCollector = None):
        super().__init__()
        self.hands = None
        self.current_agent = 0
        self.record = None
        self.last_agent = -1
        self.last_play_info = None
        self.log: List = []
        self.winner = -1
        self.result_collector = result_collector
        self.cards_remain = fullDeck.copy()
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
        self.cards_remain = fullDeck.copy()
        self.record = np.zeros(shape=([3, 15]))  # records card played by each player
        self.log = []  # record proceeding of the game (self.current_agent,action)
        self.winner = -1
        return self.get_obs()

    def step(self, action):
        '''
        :param self:
        :param action: a tuple of cards played by the current agent [cardType,tuple(cards)]
        :return:
        '''
        # Remove cards played by current agent
        cards = action[1]
        inplaceRemoveFromHand(self.hands[self.current_agent], cards)
        # update cards remain i.e remove cards played from remain_cards
        inplaceRemoveFromHand(self.cards_remain, cards)
        # update records
        for card in cards:
            self.record[self.current_agent][card - 3] += 1
        # update log
        self.log.append((self.current_agent, action))
        if cards:
            self.last_agent = self.current_agent
            self.last_play_info = action
        self.current_agent = (self.current_agent + 1) % 3
        new_round = False if self.last_agent == -1 else self.current_agent == self.last_agent
        self.winner = self.get_winner()
        if new_round:
            self.last_play_info = (CardType.UNRESTRICTED, ())
        # Change the playing agent to next one
        if self.winner == 0:  # landlord wins
            done = 1
            vec_rew = [1, -1, -1]
        elif self.winner > 0:
            done = 1
            vec_rew = [-1, 1, 1]
        else:  # no winners yet
            done = 0
            vec_rew = [0, 0, 0]
        if done and self.result_collector:
            self.result_collector.add(self.winner)
        # print('*******************************')
        # print(action)
        # print(self.last_play_info)
        # print(self.hands)
        # print('*******************************')
        # if done:
        #     print('One match done')
        obs = self.get_obs()
        return obs, np.array(vec_rew), np.array(done), {}

    def get_obs(self):
        return {
            'agent_id': self.current_agent + 1,
            'obs': np.concatenate((self.record, hand_to_nparray(self.hands[self.current_agent]))),
            'action': None,
            'mask': {'last_play': self.last_play_info,
                     'agent_hand': encode_hand(self.hands[self.current_agent])}
        }

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
        np.random.seed(seed)

    def render(self, **kwargs) -> None:
        landlord_played = ",".join(
            ["".join([cardMap[j] for j in i[1][1]]) for i in self.log if i[0] == 0 and len(i[1][1]) > 0])
        peasant_upper_played = ",".join(
            ["".join([cardMap[j] for j in i[1][1]]) for i in self.log if i[0] == 1 and len(i[1][1]) > 0])
        peasant_lower_played = ",".join(
            ["".join([cardMap[j] for j in i[1][1]]) for i in self.log if i[0] == 2 and len(i[1][1]) > 0])
        landlord_hand = " ".join(map(lambda x: cardMap[x], self.hands[0]))
        peasant_upper_hand = " ".join(map(lambda x: cardMap[x], self.hands[1]))
        peasant_lower_hand = " ".join(map(lambda x: cardMap[x], self.hands[2]))
        cards_line = " ".join([cardMap[i] for i in range(3, 18)])
        front_padding = " " * 30
        line1 = front_padding + "    Landlord"
        line2 = front_padding + "Hand Info: " + landlord_hand
        line3 = front_padding + landlord_hand
        line4 = front_padding + "Card Played: " + landlord_played
        line5_front = "Upper"
        line5_end = "Lower"
        line6_front = "Hand Info: " + peasant_upper_hand
        line6_end = "Hand Info: " + peasant_lower_hand
        front_padding = (" " * 11)
        line7_front = front_padding + peasant_upper_hand
        line7_end = front_padding + peasant_lower_hand
        line8_front = "Card Played: " + peasant_upper_played
        line8_end = "Card Played: " + peasant_lower_played
        hand_info_padding = " " * 30
        line6 = line6_front + hand_info_padding + line6_end
        line7 = line7_front + hand_info_padding + line7_end
        card_played_padding = " " * (len(line6_front) - len(line8_front) + len(hand_info_padding))
        line8 = line8_front + card_played_padding + line8_end
        line5_padding = " " * (len(line6_front) - len(line5_front) + 20)
        line5 = line5_front + line5_padding + line5_end
        print(line1)
        print(line2)
        # print(line3)
        print(line4)
        print(line5)
        print(line6)
        # print(line7)
        print(line8)
        last_log = self.log[-1]
        player_map = {0: "landlord", 1: "lower", 2: "upper"}
        print("Last player: ", player_map[last_log[0]])
        last_play_str = ",".join(map(lambda x: cardMap[x], last_log[1][1]))
        print("Cards played", last_play_str if len(last_play_str) > 0 else "pass")

    def close(self) -> None:
        pass


class DetailEnv(DouDiZhuEnv):
    row_len = 71

    def __init__(self, result_collector: ResultCollector = None):
        self.round = -1
        # one line |player|card type| {0,1,2,3} * 13 + 1 + 1|
        #          |3     |14       | 54                    |
        self.np_log = np.zeros((54 * 3, DetailEnv.row_len))
        super().__init__(result_collector)

    def reset(self):
        to_return = super().reset()
        self.round = 0
        self.np_log = np.zeros((54 * 3, DetailEnv.row_len))
        return to_return

    def step(self, action):
        self.np_log[self.round] = self.encode_row((self.current_agent, action))
        to_return = super().step(action)
        self.round += 1
        return to_return

    def get_obs(self):
        last = self.log[-1][1][1] if len(self.log) > 0 else []
        last = encode_hand(last)
        second_last = self.log[-2][1][1] if len(self.log) > 1 else []
        second_last = encode_hand(second_last)
        obs = {'agent_id': self.current_agent + 1,
               'obs': np.concatenate((self.np_log, DetailEnv.encode_hand(self.hands[self.current_agent]))),
               'mask': {'last_play': self.last_play_info,
                        'agent_hand': encode_hand(self.hands[self.current_agent]),
                        'last_two_play': np.vstack((last, second_last)),
                        'cards_remain': encode_hand(self.cards_remain),
                        'next_count': len(self.hands[(self.current_agent + 1) % 3]),
                        'next_next_count': len(self.hands[(self.current_agent + 2) % 3])
                        }
               }
        return obs

    @staticmethod
    def encode_row(row):
        agent, action = row
        card_type = action[0]
        cards = action[1]
        np_row = np.zeros(DetailEnv.row_len)
        np_row[agent] = 1
        np_row[3 + card_type] = 1
        counter = Counter(cards)
        for card, num in counter.items():
            if card not in (16, 17):
                np_row[17 + (card - 3) * 4 + num - 1] = 1
            else:
                pos = -1 if card == 17 else -2
                np_row[pos] = 1
        return np_row

    @staticmethod
    def encode_hand(hand):
        np_row = np.zeros(DetailEnv.row_len)
        counter = Counter(hand)
        for card, num in counter.items():
            if card not in (16, 17):
                for i in range(0, num):
                    np_row[17 + (card - 3) * 4 + i] = 1
            else:
                pos = -1 if card == 17 else -2
                np_row[pos] = 1
        return np_row.reshape((1, -1))

    @staticmethod
    def encode_action(action):
        row = DetailEnv.encode_row((0, action))
        row[0] = 0
        return row.reshape((1, -1))


class LandlordEnv(gym.Env):
    def __init__(self, upper_policy, lower_policy):
        self.env = DetailEnv()
        self.upper_policy = upper_policy
        self.lower_policy = lower_policy

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if action.ndim > 1:
            action = action[0]
        obs, reward, done, info = self.env.step(action)
        if not done:
            # game is not over after landlord played
            action = self.lower_policy.process_single_obs(obs['obs'], obs['mask'])[1]
            obs, reward, done, info = self.env.step(action)
        if not done:
            action = self.upper_policy.process_single_obs(obs['obs'], obs['mask'])[1]
            obs, reward, done, info = self.env.step(action)

        reward = reward[0]
        return obs, reward, done, info


if __name__ == '__main__':
    # env = DouDiZhuEnv()
    # env.step((CardType.UNRESTRICTED, ()))
    # print('done')
    env = DetailEnv()
    result = env.encode_row((1, (CardType.SOLO, ())))
    print('Done')
