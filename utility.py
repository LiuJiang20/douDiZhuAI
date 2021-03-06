from enum import auto, Enum, IntEnum
from itertools import combinations
from typing import List, Tuple
import numpy as np
import itertools


class CardType(IntEnum):
    NUKE = 0
    SOLO = 1
    PAIR = 2
    TRIO = 3
    TRIO_SOLO = 4
    TRIO_PAIR = 5
    QUADPLEX = 6
    QUADPLEX_SOLOS = 7
    QUADPLEX_PAIRS = 8
    STRIP = 9
    PAIR_STRIP = 10
    TRIO_STRIP = 11
    TRIO_SOLO_STRIP = 12
    TRIO_PAIR_STRIP = 13
    UNRESTRICTED = 14


def encode_hand(hand: List[int]):
    array_hand = np.zeros(15, dtype=np.intc)
    for card in hand:
        array_hand[card - 3] += 1
    return array_hand


def decode_hand(array_hand):
    hand = []
    for i in range(len(array_hand)):
        if array_hand[i] != 0:
            number = i + 3
            hand += [number] * int(array_hand[i])
    # print('hand after decoding:',hand)
    return hand


def hand_to_nparray(hand):
    nparray = np.zeros((1, 15))
    for card in hand:
        nparray[0][card - 3] += 1
    return nparray


def get_available_moves(hand: List[int], card_type: CardType, last_play):
    if card_type == CardType.UNRESTRICTED:
        all_moves = getAllMoves(hand)
        moves = [(moveType, move) for moveType, oneTypeOfMoves in all_moves.items() for move
                 in oneTypeOfMoves]
    else:
        all_moves = getMovesWithSameType(hand, card_type, last_play) + [()]
        moves = [(card_type, move) for move in all_moves]

    return moves


def getMovesWithSameType(hand: List[int], cardType: CardType, lastPlay):
    moves = []
    # Bombs and Nuke are always available
    if cardType != CardType.QUADPLEX or cardType != CardType.NUKE:
        moves += getQuadplexes(hand)
    else:
        moves += getQuadplexes(hand, lastPlay)
    moves += getNuke(hand)
    if cardType == CardType.SOLO:
        moves += getSolos(hand, lastPlay)
    elif cardType == CardType.PAIR:
        moves += getPairs(hand, lastPlay)
    elif cardType == CardType.TRIO:
        moves += getTrios(hand, lastPlay)
    elif cardType == CardType.TRIO_SOLO:
        moves += getTrioSolos(hand, lastPlay)
    elif cardType == CardType.TRIO_PAIR:
        moves += getTrioPairs(hand, lastPlay)
    elif cardType == CardType.QUADPLEX_SOLOS:
        moves += getQuadplexSolos(hand, lastPlay)
    elif cardType == CardType.QUADPLEX_PAIRS:
        moves += getQuadplexPairs(hand, lastPlay)
    elif cardType == CardType.STRIP:
        moves += getStrips(hand, lastPlay)
    elif cardType == CardType.PAIR_STRIP:
        moves += getPairStrips(hand, lastPlay)
    elif cardType == CardType.TRIO_STRIP:
        moves += getTrioStrips(hand, lastPlay)
    elif cardType == CardType.TRIO_SOLO_STRIP:
        moves += getTrioSoloStrips(hand, lastPlay)
    elif cardType == CardType.TRIO_PAIR_STRIP:
        moves += getTrioPairStrips(hand, lastPlay)
    return moves


def getAllMoves(hand: List[int]):
    moves = {CardType.QUADPLEX: getQuadplexes(hand),
             CardType.NUKE: getNuke(hand),
             CardType.SOLO: getSolos(hand),
             CardType.PAIR: getPairs(hand),
             CardType.TRIO: getTrios(hand),
             CardType.TRIO_SOLO: getTrioSolos(hand),
             CardType.TRIO_PAIR: getTrioPairs(hand),
             CardType.QUADPLEX_SOLOS: getQuadplexSolos(hand),
             CardType.QUADPLEX_PAIRS: getQuadplexPairs(hand),
             CardType.STRIP: getStrips(hand),
             CardType.PAIR_STRIP: getPairStrips(hand),
             CardType.TRIO_STRIP: getTrioStrips(hand),
             CardType.TRIO_SOLO_STRIP: getTrioSoloStrips(hand),
             CardType.TRIO_PAIR_STRIP: getTrioPairStrips(hand)}
    return moves


def getSolos(hand: List[int], lastPlay=None):
    if lastPlay:
        return [(i,) for i in hand if i > lastPlay[0]]
    return [(i,) for i in hand]


def getAllPairs(hand: List[int], lastPlay=None):
    pairs = []
    handSize = len(hand)
    pos = 1
    lastIndex = -1
    while pos < handSize:
        if hand[pos] == hand[pos - 1] and pos - 1 != lastIndex:
            if not lastPlay or hand[pos] > lastPlay[0]:
                pairs.append((hand[pos],) * 2)
            lastIndex = pos
        pos += 1
    return pairs


def getPairs(hand: List[int], lastPlay=None):
    pairs = set()
    handSize = len(hand)
    pos = 1
    while pos < handSize:
        if hand[pos] == hand[pos - 1] and (not lastPlay or hand[pos] > lastPlay[0]):
            pairs.add((hand[pos],) * 2)
        pos += 1
    return sorted(list(pairs))


def getTrios(hand: List[int], lastPlay=None):
    trios = set()
    handSize = len(hand)
    pos = 2
    while pos < handSize:
        if hand[pos] == hand[pos - 1] and hand[pos] == hand[pos - 2] and (not lastPlay or hand[pos] > lastPlay[0]):
            trios.add((hand[pos],) * 3)
        pos += 1
    return sorted(list(trios))


def getTrioSolos(hand: List[int], lastPlay=None):
    trioSolo = []
    if lastPlay:
        trios = getTrios(hand, lastPlay[:3])
    else:
        trios = getTrios(hand)
    for trio in trios:
        for solo in sorted(set(getSolos(removeFromHand(hand, trio)))):
            if solo[0] != trio[0]:
                trioSolo.append(trio + solo)
    return trioSolo


def getTrioPairs(hand: List[int], lastPlay=None):
    trioSolo = []
    if lastPlay:
        trios = getTrios(hand, lastPlay[:3])
    else:
        trios = getTrios(hand)
    for trio in trios:
        for pair in getPairs(removeFromHand(hand, trio)):
            trioSolo.append(trio + pair)
    return trioSolo


def getQuadplexes(hand: List[int], lastPlay=None):
    quadplexes = []
    handSize = len(hand)
    pos = 3
    while pos < handSize:
        if hand[pos] == hand[pos - 1] and hand[pos] == hand[pos - 2] and hand[pos] == hand[pos - 3] \
                and (not lastPlay or hand[pos] > lastPlay[0]):
            quadplexes.append((hand[pos],) * 4)
        pos += 1
    return quadplexes


def getQuadplexSolos(hand: List[int], lastPlay=None):
    quadplexSolos = []
    if lastPlay:
        quadplexes = getQuadplexes(hand, lastPlay[:4])
    else:
        quadplexes = getQuadplexes(hand)
    for quadplex in quadplexes:
        unique_solos = [i for i in set(getSolos(removeFromHand(hand, list(quadplex))))]
        unique_solos.sort()
        for solos in set(combinations(unique_solos, 2)):
            quadplexSolos.append(quadplex + solos[0] + solos[1])
    return quadplexSolos


def getQuadplexPairs(hand: List[int], lastPlay=None):
    quadplexPairs = []
    if lastPlay:
        quadplexes = getQuadplexes(hand, lastPlay[:4])
    else:
        quadplexes = getQuadplexes(hand)
    for quadplex in quadplexes:
        for pairs in set(combinations([i for i in getPairs(removeFromHand(hand, list(quadplex)))], 2)):
            if pairs[0] == pairs[1]:
                continue
            quadplexPairs.append(quadplex + pairs[0] + pairs[1])
    return quadplexPairs


def getStrips(hand: List[int], lastPlay=None):
    strips = []
    uniqueHand = list(np.unique(np.array(hand)))
    n = len(uniqueHand)
    i = 0
    while i < n:
        j = i + 1
        while j < n and uniqueHand[j] == uniqueHand[j - 1] + 1 and uniqueHand[j] not in (15, 16, 17):
            if (not lastPlay and j - i + 1 >= 5) or (lastPlay and hand[i] > lastPlay[0] and j - i + 1 == len(lastPlay)):
                strips.append(tuple(uniqueHand[i:j + 1]))
            j += 1
        i += 1
    return strips


def getPairStrips(hand: List[int], lastPlay=None):
    pairStrips = []
    pairs = getPairs(hand)
    n = len(pairs)
    i = 0
    # if strip length is not specified all pair strips with length greater than 3 are available
    # otherwise strip length is restricted to specified value
    while i < n:
        strip = pairs[i]
        j = i + 1
        while j < n and pairs[j][0] == pairs[j - 1][0] + 1 and pairs[j][0] not in (15, 16, 17):
            strip += pairs[j]
            if (not lastPlay and j - i + 1 >= 3) or (
                    lastPlay and strip[0] > lastPlay[0] and len(strip) == len(lastPlay)):
                pairStrips.append(strip)
            j += 1
        i += 1
    return pairStrips


def getTrioStrips(hand: List[int], lastPlay=None):
    trioStrips = []
    trios = getTrios(hand)
    n = len(trios)
    i = 0
    while i < n:
        strip = trios[i]
        j = i + 1
        while j < n and trios[j][0] == trios[j - 1][0] + 1 and trios[j][0] not in (15, 16, 17):
            strip += trios[j]
            if (not lastPlay and j - i + 1 >= 2) or (
                    lastPlay and strip[0] > lastPlay[i] and len(strip) == len(lastPlay)):
                trioStrips.append(strip)
            j += 1
        i += 1
    return trioStrips


def getTrioSoloStrips(hand: List[int], lastPlay=None):
    trioSoloStrips = []
    if lastPlay:
        trioStrips = getTrioStrips(hand, lastPlay[:len(lastPlay) // 4 * 3])
    else:
        trioStrips = getTrioStrips(hand)
    for trioStrip in trioStrips:
        trioNum = len(trioStrip) // 3
        cardsLeft = removeFromHand(hand, list(trioStrip))
        if len(cardsLeft) < trioNum:
            continue
        unique_solos = list(set(getSolos(cardsLeft)))
        unique_solos.sort()
        for solos in set(combinations(unique_solos, trioNum)):
            trioSolos = trioStrip
            cards = set(trioStrip)
            if cards.intersection(set([i[0] for i in solos])):
                continue
            for solo in solos:
                trioSolos += solo
            trioSoloStrips.append(trioSolos)
    return trioSoloStrips


def getTrioPairStrips(hand: List[int], lastPlay=None):
    trioPairStrips = []
    if lastPlay:
        trioStrips = getTrioStrips(hand, lastPlay[:len(lastPlay) // 5 * 3])
    else:
        trioStrips = getTrioStrips(hand)
    for trioStrip in trioStrips:
        trioNum = len(trioStrip) // 3
        allPairs = getPairs(removeFromHand(hand, list(trioStrip)))
        if len(allPairs) < trioNum:
            continue
        for pairs in set(combinations(allPairs, trioNum)):
            trioPairs = trioStrip
            for pair in pairs:
                trioPairs += pair
            trioPairStrips.append(trioPairs)
    return trioPairStrips


def getNuke(hand: List[int]):
    if 16 in hand and 17 in hand:
        return [(16, 17)]
    else:
        return []


def removeFromHand(hand: List[int], cards):
    left = hand.copy()
    for i in cards:
        left.remove(i)
    return left


def inplaceRemoveFromHand(hand: List[int], cards):
    for i in cards:
        hand.remove(i)


if __name__ == '__main__':
    a1 = [3, 3, 3, 4, 5, 6, 6]
    a2 = [3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7]
    # print(getPairs(a1))
    # print(getPairs(a2))
    # print(getQuadplexPairs(a2))
    # a3 = [3, 4, 6, 7, 8, 9, 10, 11, 12]
    # print(getStrips(a3))
    # print(getPairStrips(a2))
    # encoded = encode_hand(a2)
    # decoded = decode_hand(encoded)
    # if decoded == a2:
    #     print("encoding works")
    a3 = [13, 13, 14, 14, 15, 15]
    print(getPairStrips(a3))
