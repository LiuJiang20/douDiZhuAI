import random
from utility import *
from typing import Tuple
import numpy as np

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


class Game:
    LANDLORD = 0
    PEASANT_UPPER = 1
    PEASANT_LOWER = 2

    def __init__(self):
        self.players: List[Player] = [Player(), Player(), Player()]
        self.landLord: Player = None
        self.peasantUpper: Player = None
        self.peasantLower: Player = None
        self.landLordCard = []
        self.landLordPos = -1
        self.playLog: List[Tuple[int, Tuple[int]]] = []
        self.winner: int = -1

    def start(self):
        # Keep restarting the game while no one claims the landlord
        # while not self.landLord:
        #     self.bidPhase()
        self.greedyBid()
        self.inspectPlayers()

        # for player in self.players:
        #     player.showHand()
        # After there is a landlord start the game
        self.playPhase()
        print(self.playLog)

    def distribute_cards(self):
        '''
        Randomly assign 17 cards for each player, hold the last three for the landlord
        :return: None
        '''
        shuffled = fullDeck.copy()
        random.shuffle(shuffled)
        self.players[0].setHand(shuffled[:17])
        self.players[1].setHand(shuffled[17:34])
        self.players[2].setHand(shuffled[34:-3])
        self.landLordCard = shuffled[-3:]

    def bidPhase(self):
        '''
        First assign cards to each player by calling distribute_cards()
        Then ask each player to claim landlord as specified by the rule
        When this method finishes, the object can end up in 2 states:
        1. No landlord is decided, game needs to restart
        2. Landlord is assigned, and all players has their hands ready
        :return: None
        '''
        self.distribute_cards()
        bidCount = 0
        passCount = 0
        currentPos = random.randint(0, 2)
        while bidCount < 3 and passCount < 2:
            if self.players[currentPos].bid(bidCount):
                self.landLord = self.players[currentPos]
                self.landLordPos = currentPos
                bidCount += 1
                passCount = 0
            else:
                passCount += 1
        if self.landLord:
            self.landLord.hand += self.landLordCard
            self.landLord.hand.sort()
            self.peasantUpper = self.players[(currentPos + 1) % 3]
            self.peasantUpper = self.players[(currentPos + 2) % 3]

    def greedyBid(self):
        self.distribute_cards()
        pos = np.argmax([sum(player.hand) for player in self.players])
        self.landLordPos = pos
        self.landLord = self.players[pos]
        self.landLord.hand += self.landLordCard
        self.landLord.hand.sort()
        self.peasantUpper = self.players[(pos + 1) % 3]
        self.peasantLower = self.players[(pos + 2) % 3]

    def playPhase(self):
        currentPos = self.landLordPos
        round = 0
        while True:
            start = len(self.playLog)
            currentPos = self.oneRound(currentPos)
            print(self.playLog[start:])
            round += 1
            if self.hasWinner():
                break
        self.winner = self.getDistance(self.getWinner())

    def oneRound(self, currentPos):
        cardPlayed = []
        playType = CardType.UNRESTRICTED
        lastPos = -1
        while (not self.hasWinner()) and lastPos != currentPos:
            currCardPlayed, playType = self.players[currentPos].playCards(cardPlayed, playType)
            # Log card played
            # print('-------------')
            # print(currentPos)
            # print(cardPlayed, playType)
            # print('----------------')
            self.playLog.append((self.getDistance(currentPos), currCardPlayed))
            if currCardPlayed:
                lastPos = currentPos
                cardPlayed = currCardPlayed
            if not self.checkHandIntegrity():
                self.inspectPlayers()
            currentPos = (currentPos + 1) % 3
        return currentPos

    def getDistance(self, pos: int):
        currPos = self.landLordPos
        distance = 0
        while currPos != pos:
            currPos = (currPos + 1) % 3
            distance += 1
        return distance

    def hasWinner(self):
        return any([i.handEmpty() for i in self.players])

    def getWinner(self):
        for i in range(len(self.players)):
            if self.players[i].handEmpty():
                return i
        return -1

    def isLandLord(self, player):
        return player is self.landLord

    def isPeasantUpper(self, player):
        return player is self.peasantUpper

    def isPeasantLower(self, player):
        return player is self.peasantLower

    def inspectPlayers(self):
        print('***********************')
        for player in self.players:
            player.showHand()
        print('***********************')

    def checkHandIntegrity(self):
        for player in self.players:
            for i in range(1, len(player.hand)):
                if player.hand[i] < player.hand[i - 1]:
                    return False
        return True


class Player:
    def __init__(self):
        self.hand = []
        self.exploreRate = 0.5

    def playCards(self, lastPlay: list, cardType: CardType):
        '''
        :param lastPlay:
        :param cardType:
        :return: (cards to play, cards' type)
        '''
        scoredMoves = self.getAvailableMoves(lastPlay, cardType)
        scoredMoves.sort(key=lambda x: x[2], reverse=True)
        randNum = random.random()
        # with 90% chance play the best move so far
        # and 10% chance explore other moves
        if len(scoredMoves) == 0:
            return [], cardType
        if len(scoredMoves) == 1 or randNum > self.exploreRate:
            selectMove = scoredMoves[0]
        else:
            selectMove = scoredMoves[random.randint(1, len(scoredMoves) - 1)]
        inplaceRemoveFromHand(self.hand, selectMove[0])
        return selectMove[:2]

    def setHand(self, cards):
        self.hand = cards
        self.hand.sort()

    def evaluateMove(self, cards):
        return 1

    def getAvailableMoves(self, lastPlay: list, cardType: CardType):
        if cardType == CardType.UNRESTRICTED:
            allMoves = getAllMoves(self.hand)
            moves = [[move, moveType, self.evaluateMove(move)] for moveType, oneTypeOfMoves in allMoves.items() for move
                     in oneTypeOfMoves]
        else:
            allMoves = getMovesWithSameType(self.hand, cardType, lastPlay)
            moves = [[move, cardType, self.evaluateMove(move)] for move in allMoves]
        if cardType != CardType.UNRESTRICTED:
            moves.append([(), cardType, self.evaluateMove(())])
        return moves

    def bid(self, lastBid):
        if random.random() > 0.5:
            return True
        return False

    def handEmpty(self):
        return len(self.hand) == 0

    def showHand(self):
        print(" ".join([cardMap[i] for i in self.hand]))


if __name__ == '__main__':
    g = Game()
    g.start()
