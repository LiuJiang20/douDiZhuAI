from utility import *
from unittest import TestCase
import unittest


# TODO: TRIO_PAIR_STRIP


class TestAllCardSelectionMethod(TestCase):
    def testGetSolos(self):
        h1 = []
        self.assertEqual(getSolos(h1), [])
        h2 = [3]
        self.assertEqual(getSolos(h2), [(3,)])
        h3 = [3, 3, 3]
        self.assertEqual(getSolos(h3), [(3,), (3,), (3,)])
        h4 = [3, 3, 3, 3, 4]
        self.assertEqual(getSolos(h4), [(3,), (3,), (3,), (3,), (4,)])
        h5 = [3, 3, 3, 4, 5, 5, 5, 6, 6, 7, 8, 10, 11, 12, 13, 13, 14]
        self.assertEqual(getSolos(h5),
                         [(3,), (3,), (3,), (4,), (5,), (5,), (5,), (6,), (6,), (7,), (8,), (10,), (11,),
                          (12,), (13,), (13,), (14,)])
        self.assertEqual(getSolos(h5, (3,)), [(4,), (5,), (5,), (5,), (6,), (6,), (7,), (8,), (10,), (11,),
                                              (12,), (13,), (13,), (14,)])
        self.assertEqual(getSolos(h5, (13,)), [(14,)])

    def testGetPairs(self):
        h1 = []
        self.assertEqual(getPairs(h1), [])
        h2 = [6]
        self.assertEqual(getPairs(h2), [])
        h3 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17]
        self.assertEqual(getPairs(h3), [])
        h4 = [4, 4]
        self.assertEqual(getPairs(h4), [(4, 4)])
        h5 = [4, 4, 4]
        self.assertEqual(getPairs(h5), [(4, 4)])
        h6 = [4, 4, 4, 4]
        self.assertEqual(getPairs(h6), [(4, 4)])
        h7 = [3, 3, 5, 6, 7, 8, 8, 16, 17]
        self.assertEqual(getPairs(h7), [(3, 3), (8, 8)])
        self.assertEqual(getPairs(h7, (3, 3)), [(8, 8)])
        self.assertEqual(getPairs(h7, (7, 7)), [(8, 8)])
        self.assertEqual(getPairs(h7, (8, 8)), [])
        h8 = [3, 4, 5, 6, 6, 7, 7, 9, 9, 10, 10, 11, 12, 12, 13, 15, 16]
        self.assertEqual(getPairs(h8), [(6, 6), (7, 7), (9, 9), (10, 10), (12, 12)])
        self.assertEqual(getPairs(h8, (12, 12)), [])
        self.assertEqual(getPairs(h8, (3, 3)), [(6, 6), (7, 7), (9, 9), (10, 10), (12, 12)])
        self.assertEqual(getPairs(h8, (9, 9)), [(10, 10), (12, 12)])

    def testGetAllPairs(self):
        h1 = []
        self.assertEqual(getAllPairs(h1), [])
        h2 = [6]
        self.assertEqual(getAllPairs(h2), [])
        h3 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17]
        self.assertEqual(getAllPairs(h3), [])
        h4 = [4, 4]
        self.assertEqual(getAllPairs(h4), [(4, 4)])
        h5 = [4, 4, 4]
        self.assertEqual(getAllPairs(h5), [(4, 4)])
        h6 = [4, 4, 4, 4]
        self.assertEqual(getAllPairs(h6), [(4, 4), (4, 4)])
        h7 = [3, 3, 5, 6, 7, 8, 8, 16, 17]
        self.assertEqual(getAllPairs(h7), [(3, 3), (8, 8)])
        h8 = [3, 4, 5, 6, 6, 7, 7, 9, 9, 10, 10, 11, 12, 12, 13, 15, 16]
        self.assertEqual(getAllPairs(h8), [(6, 6), (7, 7), (9, 9), (10, 10), (12, 12)])
        self.assertEqual(getAllPairs(h8, (12, 12)), [])
        self.assertEqual(getAllPairs(h8, (3, 3)), [(6, 6), (7, 7), (9, 9), (10, 10), (12, 12)])
        self.assertEqual(getAllPairs(h8, (9, 9)), [(10, 10), (12, 12)])
        h9 = [3, 4, 5, 6, 6, 6, 6, 9, 9, 10, 10, 11, 12, 12, 13, 15, 16]
        self.assertEqual(getAllPairs(h9), [(6, 6), (6, 6), (9, 9), (10, 10), (12, 12)])
        self.assertEqual(getAllPairs(h9, (12, 12)), [])
        self.assertEqual(getAllPairs(h9, (3, 3)), [(6, 6), (6, 6), (9, 9), (10, 10), (12, 12)])
        self.assertEqual(getAllPairs(h9, (9, 9)), [(10, 10), (12, 12)])

    def testGetTrios(self):
        h1 = []
        self.assertEqual(getTrios(h1), [])
        h2 = [8]
        self.assertEqual(getTrios(h2), [])
        h3 = [8, 8]
        self.assertEqual(getTrios(h3), [])
        h4 = [8, 8, 8]
        self.assertEqual(getTrios(h4), [(8, 8, 8)])
        self.assertEqual(getTrios(h4, (4, 4, 4)), [(8, 8, 8)])
        self.assertEqual(getTrios(h4, (8, 8, 8)), [])
        h5 = [8, 8, 8, 8]
        self.assertEqual(getTrios(h5), [(8, 8, 8)])
        h6 = [5, 9, 9, 9, 9, 2]
        self.assertEqual(getTrios(h6), [(9, 9, 9)])
        h7 = [6, 7, 10, 15, 16, 17]
        self.assertEqual(getTrios(h7), [])
        h8 = [3, 3, 4, 6, 6, 6, 7, 8, 9, 10, 11, 11, 12, 12, 15, 15, 16]
        self.assertEqual(getTrios(h8), [(6, 6, 6)])
        h8 = [3, 3, 4, 6, 6, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12, 15, 16]
        self.assertEqual(getTrios(h8), [(6, 6, 6), (12, 12, 12)])
        self.assertEqual(getTrios(h8, (5, 5, 5)), [(6, 6, 6), (12, 12, 12)])
        self.assertEqual(getTrios(h8, (7, 7, 7)), [(12, 12, 12)])
        self.assertEqual(getTrios(h8, (11, 11, 11)), [(12, 12, 12)])
        self.assertEqual(getTrios(h8, (12, 12, 12)), [])

    def testGetQuadplex(self):
        h1 = []
        self.assertEqual(getQuadplexes(h1), [])
        h2 = [11]
        self.assertEqual(getQuadplexes(h2), [])
        h3 = [11, 11]
        self.assertEqual(getQuadplexes(h3), [])
        h4 = [11, 11, 11]
        self.assertEqual(getQuadplexes(h4), [])
        h5 = [11, 11, 11, 11]
        self.assertEqual(getQuadplexes(h5), [(11, 11, 11, 11)])
        self.assertEqual(getQuadplexes(h5, (10, 10, 10, 10)), [(11, 11, 11, 11)])
        self.assertEqual(getQuadplexes(h5, (12, 12, 12, 12)), [])
        h6 = [6, 11, 11, 11]
        self.assertEqual(getQuadplexes(h6), [])
        h7 = [4, 4, 4, 4, 5, 5, 5, 6, 7, 7, 7, 7]
        self.assertEqual(getQuadplexes(h7), [(4, 4, 4, 4), (7, 7, 7, 7)])
        self.assertEqual(getQuadplexes(h7, (3, 3, 3, 3)), [(4, 4, 4, 4), (7, 7, 7, 7)])
        self.assertEqual(getQuadplexes(h7, (5, 5, 5, 5)), [(7, 7, 7, 7)])
        self.assertEqual(getQuadplexes(h7, (8, 8, 8, 8)), [])
        h8 = [3, 3, 4, 5, 6, 6, 6, 6, 8, 10, 10, 11, 13, 13, 15, 15, 17]
        self.assertEqual(getQuadplexes(h8), [(6, 6, 6, 6)])
        self.assertEqual(getQuadplexes(h8, (5, 5, 5, 5)), [(6, 6, 6, 6)])
        self.assertEqual(getQuadplexes(h8, (7, 7, 7, 7)), [])

    def testGetTrioSolos(self):
        h1 = []
        self.assertEqual(getTrioSolos(h1), [])
        self.assertEqual(getTrioSolos(h1, (3, 3, 3, 1)), [])
        h2 = [5]
        self.assertEqual(getTrioSolos(h2), [])
        h3 = [6, 6, 6]
        self.assertEqual(getTrioSolos(h3), [])
        h4 = [6, 6, 6, 9]
        self.assertEqual(getTrioSolos(h4), [(6, 6, 6, 9)])
        h5 = [6, 6, 6, 6]
        self.assertEqual(getTrioSolos(h5), [])
        h6 = [4, 6, 7, 8, 8, 11, 11, 11, 13, 15]
        self.assertEqual(getTrioSolos(h6),
                         [(11, 11, 11, 4), (11, 11, 11, 6), (11, 11, 11, 7), (11, 11, 11, 8), (11, 11, 11, 13),
                          (11, 11, 11, 15)])
        h7 = [3, 5, 5, 5, 5, 6, 6, 6, 7]
        self.assertEqual(getTrioSolos(h7),
                         [(5, 5, 5, 3), (5, 5, 5, 6), (5, 5, 5, 7), (6, 6, 6, 3), (6, 6, 6, 5), (6, 6, 6, 7)])
        self.assertEqual(getTrioSolos(h7, (4, 4, 4, 3)),
                         [(5, 5, 5, 3), (5, 5, 5, 6), (5, 5, 5, 7), (6, 6, 6, 3), (6, 6, 6, 5), (6, 6, 6, 7)])
        self.assertEqual(getTrioSolos(h7, (5, 5, 5, 3)), [(6, 6, 6, 3), (6, 6, 6, 5), (6, 6, 6, 7)])
        h8 = [3, 3, 3, 5, 7, 7, 8, 9, 9, 10, 10, 11, 11, 11, 12, 14, 15]
        self.assertEqual(getTrioSolos(h8),
                         [(3, 3, 3, 5), (3, 3, 3, 7), (3, 3, 3, 8), (3, 3, 3, 9), (3, 3, 3, 10), (3, 3, 3, 11),
                          (3, 3, 3, 12), (3, 3, 3, 14), (3, 3, 3, 15),
                          (11, 11, 11, 3), (11, 11, 11, 5), (11, 11, 11, 7), (11, 11, 11, 8), (11, 11, 11, 9),
                          (11, 11, 11, 10), (11, 11, 11, 12), (11, 11, 11, 14), (11, 11, 11, 15)])
        self.assertEqual(getTrioSolos(h8, (3, 3, 3, 7)), [
            (11, 11, 11, 3), (11, 11, 11, 5), (11, 11, 11, 7), (11, 11, 11, 8), (11, 11, 11, 9),
            (11, 11, 11, 10), (11, 11, 11, 12), (11, 11, 11, 14), (11, 11, 11, 15)])
        self.assertEqual(getTrioSolos(h8, (10, 10, 10, 15)), [
            (11, 11, 11, 3), (11, 11, 11, 5), (11, 11, 11, 7), (11, 11, 11, 8), (11, 11, 11, 9),
            (11, 11, 11, 10), (11, 11, 11, 12), (11, 11, 11, 14), (11, 11, 11, 15)])

        self.assertEqual(getTrioSolos(h8, (11, 11, 11, 15)), [])

    def testGetTrioPair(self):
        h1 = []
        self.assertEqual(getTrioPairs(h1), [])
        h2 = [8]
        self.assertEqual(getTrioPairs(h2), [])
        h3 = [8, 8, 8, 10]
        self.assertEqual(getTrioPairs(h3), [])
        h4 = [8, 8, 8, 10, 10]
        self.assertEqual(getTrioPairs(h4), [(8, 8, 8, 10, 10)])
        h5 = [8, 8, 8, 10, 11]
        self.assertEqual(getTrioPairs(h5), [])
        h6 = [4, 4, 4, 4, 6, 6, 8, 8, 8, 15, 15, 15]
        self.assertEqual(getTrioPairs(h6), [(4, 4, 4, 6, 6), (4, 4, 4, 8, 8), (4, 4, 4, 15, 15),
                                            (8, 8, 8, 4, 4), (8, 8, 8, 6, 6), (8, 8, 8, 15, 15),
                                            (15, 15, 15, 4, 4), (15, 15, 15, 6, 6), (15, 15, 15, 8, 8)])
        self.assertEqual(getTrioPairs(h6, (3, 3, 3, 10, 10)), [(4, 4, 4, 6, 6), (4, 4, 4, 8, 8), (4, 4, 4, 15, 15),
                                                               (8, 8, 8, 4, 4), (8, 8, 8, 6, 6), (8, 8, 8, 15, 15),
                                                               (15, 15, 15, 4, 4), (15, 15, 15, 6, 6),
                                                               (15, 15, 15, 8, 8)])
        self.assertEqual(getTrioPairs(h6, (4, 4, 4, 10, 10)), [(8, 8, 8, 4, 4), (8, 8, 8, 6, 6), (8, 8, 8, 15, 15),
                                                               (15, 15, 15, 4, 4), (15, 15, 15, 6, 6),
                                                               (15, 15, 15, 8, 8)])
        self.assertEqual(getTrioPairs(h6, (12, 12, 12, 10, 10)), [(15, 15, 15, 4, 4), (15, 15, 15, 6, 6),
                                                                  (15, 15, 15, 8, 8)])

    def testQuadPlexSolos(self):
        h1 = []
        self.assertEqual(getQuadplexSolos(h1), [])
        h2 = [4]
        self.assertEqual(getQuadplexSolos(h2), [])
        h3 = [8, 8]
        self.assertEqual(getQuadplexSolos(h3), [])
        h4 = [7, 7, 7, 7, 10]
        self.assertEqual(getQuadplexSolos(h4), [])
        h5 = [7, 7, 7, 7, 10, 10]
        self.assertEqual(getQuadplexSolos(h5), [(7, 7, 7, 7, 10, 10)])
        h6 = [7, 7, 7, 7, 10, 11]
        self.assertEqual(getQuadplexSolos(h6), [(7, 7, 7, 7, 10, 11)])
        h7 = [4, 4, 4, 4, 7, 8, 9, 9, 9, 9]
        self.assertSetEqual(set(getQuadplexSolos(h7)),
                            {(4, 4, 4, 4, 7, 8), (4, 4, 4, 4, 7, 9), (4, 4, 4, 4, 8, 9), (4, 4, 4, 4, 9, 9),
                             (9, 9, 9, 9, 4, 4), (9, 9, 9, 9, 4, 7), (9, 9, 9, 9, 4, 8), (9, 9, 9, 9, 7, 8)})
        self.assertEqual(len(getQuadplexSolos(h7)), 8)
        h8 = [4, 4, 4, 4, 7, 7, 7, 8, 9, 9, 9, 9]
        self.assertSetEqual(set(getQuadplexSolos(h8)),
                            {(4, 4, 4, 4, 7, 8), (4, 4, 4, 4, 7, 9), (4, 4, 4, 4, 8, 9), (4, 4, 4, 4, 9, 9),
                             (4, 4, 4, 4, 7, 7), (9, 9, 9, 9, 4, 4), (9, 9, 9, 9, 4, 7), (9, 9, 9, 9, 4, 8),
                             (9, 9, 9, 9, 7, 8), (9, 9, 9, 9, 7, 7)})
        self.assertEqual(len(getQuadplexSolos(h8)), 10)
        self.assertSetEqual(set(getQuadplexSolos(h8, (3, 3, 3, 3, 12, 12))),
                            {(4, 4, 4, 4, 7, 8), (4, 4, 4, 4, 7, 9), (4, 4, 4, 4, 8, 9), (4, 4, 4, 4, 9, 9),
                             (4, 4, 4, 4, 7, 7), (9, 9, 9, 9, 4, 4), (9, 9, 9, 9, 4, 7), (9, 9, 9, 9, 4, 8),
                             (9, 9, 9, 9, 7, 8), (9, 9, 9, 9, 7, 7)})
        self.assertEqual(len(getQuadplexSolos(h8, (3, 3, 3, 3, 12, 12))), 10)

    def testGetQuadplexPair(self):
        h1 = []
        self.assertEqual(getQuadplexPairs(h1), [])
        h2 = [4]
        self.assertEqual(getQuadplexPairs(h2), [])
        h3 = [4, 4, 4, 4]
        self.assertEqual(getQuadplexPairs(h3), [])
        h4 = [4, 4, 4, 4, 5, 5]
        self.assertEqual(getQuadplexPairs(h4), [])
        h5 = [4, 4, 4, 4, 6, 7]
        self.assertEqual(getQuadplexPairs(h5), [])
        h6 = [4, 4, 4, 4, 5, 5, 10]
        self.assertEqual(getQuadplexPairs(h6), [])
        h7 = [5, 5, 5, 5, 6, 6, 9, 9]
        self.assertEqual(getQuadplexPairs(h7), [(5, 5, 5, 5, 6, 6, 9, 9)])
        h8 = [5, 5, 5, 5, 6, 6, 9, 9, 13, 13]
        self.assertSetEqual(set(getQuadplexPairs(h8)),
                            {(5, 5, 5, 5, 6, 6, 9, 9), (5, 5, 5, 5, 6, 6, 13, 13), (5, 5, 5, 5, 9, 9, 13, 13)})
        self.assertEqual(len(getQuadplexPairs(h8)), 3)
        h9 = [5, 5, 5, 5, 6, 6, 6, 9, 9, 9, 13, 13]
        self.assertSetEqual(set(getQuadplexPairs(h9)),
                            {(5, 5, 5, 5, 6, 6, 9, 9), (5, 5, 5, 5, 6, 6, 13, 13), (5, 5, 5, 5, 9, 9, 13, 13)})
        self.assertEqual(len(getQuadplexPairs(h9)), 3)
        h10 = [6, 6, 6, 6, 7, 7, 12, 12, 13, 13, 13, 13]
        self.assertEqual(set(getQuadplexPairs(h10)),
                         {(6, 6, 6, 6, 7, 7, 12, 12), (6, 6, 6, 6, 7, 7, 13, 13), (6, 6, 6, 6, 12, 12, 13, 13),
                          (13, 13, 13, 13, 6, 6, 7, 7), (13, 13, 13, 13, 6, 6, 12, 12), (13, 13, 13, 13, 7, 7, 12, 12)})
        self.assertEqual(len(getQuadplexPairs(h10)), 6)
        self.assertEqual(set(getQuadplexPairs(h10, (5, 5, 5, 5, 10, 10, 11, 11))),
                         {(6, 6, 6, 6, 7, 7, 12, 12), (6, 6, 6, 6, 7, 7, 13, 13), (6, 6, 6, 6, 12, 12, 13, 13),
                          (13, 13, 13, 13, 6, 6, 7, 7), (13, 13, 13, 13, 6, 6, 12, 12), (13, 13, 13, 13, 7, 7, 12, 12)})
        self.assertEqual(len(getQuadplexPairs(h10)), 6)
        self.assertEqual(set(getQuadplexPairs(h10, (7, 7, 7, 7, 3, 3, 4, 4))),
                         {(13, 13, 13, 13, 6, 6, 7, 7), (13, 13, 13, 13, 6, 6, 12, 12), (13, 13, 13, 13, 7, 7, 12, 12)})
        self.assertEqual(len(getQuadplexPairs(h10, (7, 7, 7, 7, 3, 3, 4, 4))), 3)

    def testGetStrips(self):
        h1 = []
        self.assertEqual(getStrips(h1), [])
        h2 = [4]
        self.assertEqual(getStrips(h2), [])
        h3 = [4, 4, 5, 6]
        self.assertEqual(getStrips(h3), [])
        h4 = [4, 5, 6, 7]
        self.assertEqual(getStrips(h4), [])
        h5 = [4, 5, 6, 7, 8]
        self.assertEqual(getStrips(h5), [(4, 5, 6, 7, 8)])
        h6 = [4, 5, 6, 8, 9]
        self.assertEqual(getStrips(h6), [])
        h7 = [4, 5, 6, 7, 8, 8, 9, 10]
        self.assertEqual(set(getStrips(h7)), {(4, 5, 6, 7, 8), (5, 6, 7, 8, 9), (6, 7, 8, 9, 10), (4, 5, 6, 7, 8, 9),
                                              (5, 6, 7, 8, 9, 10), (4, 5, 6, 7, 8, 9, 10)})
        self.assertEqual(len(getStrips(h7)), 6)
        self.assertEqual(set(getStrips(h7, (3, 4, 5, 6, 7))), {(4, 5, 6, 7, 8), (5, 6, 7, 8, 9), (6, 7, 8, 9, 10)})
        self.assertEqual(set(getStrips(h7, (4, 5, 6, 7, 8))), {(5, 6, 7, 8, 9), (6, 7, 8, 9, 10)})
        self.assertEqual(set(getStrips(h7, (7, 8, 9, 10, 11))), set())
        h8 = [4, 5, 6, 7, 8, 10, 11, 12, 13, 14]
        self.assertEqual(set(getStrips(h8)), {(4, 5, 6, 7, 8), (10, 11, 12, 13, 14)})
        h9 = [4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
        self.assertEqual(set(getStrips(h9)), {(4, 5, 6, 7, 8), (10, 11, 12, 13, 14)})

    def testGetNuke(self):
        h1 = []
        self.assertEqual(getNuke(h1), [])
        h2 = [16]
        self.assertEqual(getNuke(h2), [])
        h3 = [16, 17]
        self.assertEqual(getNuke(h3), [(16, 17)])
        h4 = [15, 16, 17]
        self.assertEqual(getNuke(h4), [(16, 17)])
        h5 = [5, 5, 6, 7, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 16]
        self.assertEqual(getNuke(h5), [])

    def testGetPairStrip(self):
        h1 = []
        self.assertEqual(getPairStrips(h1), [])
        h2 = [5, 5]
        self.assertEqual(getPairStrips(h2), [])
        h3 = [4, 5, 6, 7, 8]
        self.assertEqual(getPairStrips(h3), [])
        h4 = [3, 3, 4, 4]
        self.assertEqual(getPairStrips(h4), [])
        h5 = [3, 3, 4, 4, 5, 6, 7]
        self.assertEqual(getPairStrips(h5), [])
        h6 = [3, 3, 4, 4, 5, 5]
        self.assertEqual(getPairStrips(h6), [(3, 3, 4, 4, 5, 5)])
        h7 = [3, 3, 4, 4, 7, 7, 8, 8]
        self.assertEqual(getPairStrips(h7), [])
        h8 = [13, 13, 14, 14, 15, 15]
        self.assertEqual(getPairStrips(h8), [])
        h9 = [6, 6, 7, 7, 7, 8, 8, 9, 9]
        self.assertSetEqual(set(getPairStrips(h9)), {(6, 6, 7, 7, 8, 8), (7, 7, 8, 8, 9, 9), (6, 6, 7, 7, 8, 8, 9, 9)})
        self.assertEqual(len(getPairStrips(h9)), 3)
        self.assertSetEqual(set(getPairStrips(h9, (5, 5, 6, 6, 7, 7))), {(6, 6, 7, 7, 8, 8), (7, 7, 8, 8, 9, 9)})
        self.assertSetEqual(set(getPairStrips(h9, (6, 6, 7, 7, 8, 8))), {(7, 7, 8, 8, 9, 9)})

    def testGetTrioStrip(self):
        h1 = []
        self.assertEqual(getTrioStrips(h1), [])
        h2 = [3, 3, 3, 4, 4]
        self.assertEqual(getTrioStrips(h2), [])
        h3 = [3, 3, 3, 4, 5, 6]
        self.assertEqual(getTrioStrips(h3), [])
        h4 = [3, 3, 3]
        self.assertEqual(getTrioStrips(h4), [])
        h5 = [3, 3, 3, 4, 4, 4]
        self.assertEqual(getTrioStrips(h5), [(3, 3, 3, 4, 4, 4)])
        h6 = [3, 3, 3, 5, 5, 5]
        self.assertEqual(getTrioStrips(h6), [])
        h7 = [14, 14, 14, 15, 15, 15]
        self.assertEqual(getTrioStrips(h7), [])
        h8 = [6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 15]
        self.assertSetEqual(set(getTrioStrips(h8)),
                            {(6, 6, 6, 7, 7, 7), (7, 7, 7, 8, 8, 8), (6, 6, 6, 7, 7, 7, 8, 8, 8)})
        self.assertEqual(len(getTrioStrips(h8)), 3)
        self.assertSetEqual(set(getTrioStrips(h8, (3, 3, 3, 4, 4, 4))),
                            {(6, 6, 6, 7, 7, 7), (7, 7, 7, 8, 8, 8)})
        self.assertEqual(len(getTrioStrips(h8, (3, 3, 3, 4, 4, 4))), 2)
        h9 = [10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13]
        self.assertSetEqual(set(getTrioStrips(h9, (3, 3, 3, 4, 4, 4))),
                            {(10, 10, 10, 11, 11, 11), (11, 11, 11, 12, 12, 12), (12, 12, 12, 13, 13, 13)})
        self.assertSetEqual(set(getTrioStrips(h9, (3, 3, 3, 4, 4, 4, 5, 5, 5))),
                            {(10, 10, 10, 11, 11, 11, 12, 12, 12), (11, 11, 11, 12, 12, 12, 13, 13, 13)})

    def testGetTrioSoloStrip(self):
        h1 = []
        self.assertEqual(getTrioSoloStrips(h1), [])
        h2 = [3, 3, 3, 1]
        self.assertEqual(getTrioSoloStrips(h2), [])
        h3 = [3, 3, 3, 4, 4, 5, 5, 5]
        self.assertEqual(getTrioSoloStrips(h3), [])
        h4 = [5, 6, 8, 8, 8, 9, 9, 9]
        self.assertEqual(getTrioSoloStrips(h4), [(8, 8, 8, 9, 9, 9, 5, 6)])
        h5 = [8, 8, 8, 9, 9, 9, 10, 11]
        self.assertEqual(getTrioSoloStrips(h5), [(8, 8, 8, 9, 9, 9, 10, 11)])
        h6 = [5, 5, 7, 7, 7, 8, 8, 8]
        self.assertEqual(getTrioSoloStrips(h6), [(7, 7, 7, 8, 8, 8, 5, 5)])
        h7 = [5, 7, 7, 7, 8, 8, 8, 10]
        self.assertEqual(getTrioSoloStrips(h7), [(7, 7, 7, 8, 8, 8, 5, 10)])
        h8 = [7, 7, 7, 8, 8, 8, 10]
        self.assertEqual(getTrioSoloStrips(h8), [])
        h9 = [5, 5, 7, 7, 7, 8, 8, 8, 10]
        self.assertEqual(set(getTrioSoloStrips(h9)), {(7, 7, 7, 8, 8, 8, 5, 5), (7, 7, 7, 8, 8, 8, 5, 10)})
        self.assertEqual(len(getTrioSoloStrips(h9)), 2)
        h10 = [5, 5, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 14]
        self.assertSetEqual(set(getTrioSoloStrips(h10)),
                            {(7, 7, 7, 8, 8, 8, 5, 5), (7, 7, 7, 8, 8, 8, 5, 9), (7, 7, 7, 8, 8, 8, 5, 10),
                             (7, 7, 7, 8, 8, 8, 10, 14),
                             (7, 7, 7, 8, 8, 8, 5, 14), (7, 7, 7, 8, 8, 8, 9, 9), (7, 7, 7, 8, 8, 8, 9, 10),
                             (7, 7, 7, 8, 8, 8, 9, 14), (8, 8, 8, 9, 9, 9, 5, 5), (8, 8, 8, 9, 9, 9, 5, 7),
                             (8, 8, 8, 9, 9, 9, 5, 10), (8, 8, 8, 9, 9, 9, 5, 14), (8, 8, 8, 9, 9, 9, 7, 7),
                             (8, 8, 8, 9, 9, 9, 10, 14),
                             (8, 8, 8, 9, 9, 9, 7, 10), (8, 8, 8, 9, 9, 9, 7, 14),
                             (7, 7, 7, 8, 8, 8, 9, 9, 9, 5, 10, 14),
                             (7, 7, 7, 8, 8, 8, 9, 9, 9, 5, 5, 10), (7, 7, 7, 8, 8, 8, 9, 9, 9, 5, 5, 14)})
        self.assertSetEqual(set(getTrioSoloStrips(h10, (3, 3, 3, 4, 4, 4, 15, 15))),
                            {(7, 7, 7, 8, 8, 8, 5, 5), (7, 7, 7, 8, 8, 8, 5, 9), (7, 7, 7, 8, 8, 8, 5, 10),
                             (7, 7, 7, 8, 8, 8, 10, 14), (7, 7, 7, 8, 8, 8, 5, 14), (7, 7, 7, 8, 8, 8, 9, 9),
                             (7, 7, 7, 8, 8, 8, 9, 10), (7, 7, 7, 8, 8, 8, 9, 14), (8, 8, 8, 9, 9, 9, 5, 5),
                             (8, 8, 8, 9, 9, 9, 5, 7), (8, 8, 8, 9, 9, 9, 5, 10), (8, 8, 8, 9, 9, 9, 5, 14),
                             (8, 8, 8, 9, 9, 9, 7, 7), (8, 8, 8, 9, 9, 9, 10, 14), (8, 8, 8, 9, 9, 9, 7, 10),
                             (8, 8, 8, 9, 9, 9, 7, 14)})
        self.assertEqual(getTrioSoloStrips(h10, (10, 10, 10, 11, 11, 11, 12, 12, 12, 3, 4, 5)), [])
        self.assertEqual(getTrioSoloStrips(h10, (10, 10, 10, 11, 11, 11, 3, 4, 5)), [])

    def testGetTrioPairStrip(self):
        h1 = []
        self.assertEqual(getTrioPairStrips(h1), [])
        h2 = [3, 3, 3]
        self.assertEqual(getTrioPairStrips(h2), [])
        h3 = [3, 3, 3, 4, 4, 4]
        self.assertEqual(getTrioPairStrips(h3), [])
        h4 = [3, 3, 3, 4, 4, 4, 5, 5]
        self.assertEqual(getTrioPairStrips(h4), [])
        h5 = [3, 3, 3, 4, 4, 4, 5, 6]
        self.assertEqual(getTrioPairStrips(h5), [])
        h6 = [3, 3, 3, 4, 4, 4, 5, 5, 6]
        self.assertEqual(getTrioPairStrips(h6), [])
        h65 = [3,3,3,4,4,4,5,5,5,5]
        self.assertEqual(getTrioPairStrips(h65),[(3,3,3,4,4,4,5,5,5,5)])
        h66 = [3,3,4,4,14,14,15,15,15]
        self.assertEqual(getTrioPairStrips(h66),[])
        h67 = [3,3,3,4,4,4,5,5,15,15]
        self.assertEqual(getTrioPairStrips(h67),[(3,3,3,4,4,4,5,5,15,15)])
        h7 = [3, 3, 3, 4, 4, 4, 5, 5, 6, 6]
        self.assertEqual(getTrioPairStrips(h7), [(3, 3, 3, 4, 4, 4, 5, 5, 6, 6)])
        h8 = [3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6]
        self.assertSetEqual(set(getTrioPairStrips(h8)),
                            {(3, 3, 3, 4, 4, 4, 5, 5, 6, 6), (4, 4, 4, 5, 5, 5, 3, 3, 6, 6)})
        h9 = [3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 10, 10, 10, 11, 11, 11]
        self.assertSetEqual(set(getTrioPairStrips(h9)),
                            {(3, 3, 3, 4, 4, 4, 5, 5, 6, 6), (3, 3, 3, 4, 4, 4, 5, 5, 10, 10),
                             (3, 3, 3, 4, 4, 4, 5, 5, 11, 11), (3, 3, 3, 4, 4, 4, 6, 6, 10, 10),
                             (3, 3, 3, 4, 4, 4, 6, 6, 11, 11), (3, 3, 3, 4, 4, 4, 10, 10, 11, 11),
                             (10, 10, 10, 11, 11, 11, 5, 5, 6, 6), (10, 10, 10, 11, 11, 11, 3, 3, 4, 4),
                             (10, 10, 10, 11, 11, 11, 3, 3, 5, 5), (10, 10, 10, 11, 11, 11, 3, 3, 6, 6),
                             (10, 10, 10, 11, 11, 11, 4, 4, 5, 5), (10, 10, 10, 11, 11, 11, 4, 4, 6, 6)})
        self.assertEqual(len(getTrioPairStrips(h9)), 12)
        self.assertSetEqual(set(getTrioPairStrips(h9, (7, 7, 7, 8, 8, 8, 12, 12, 13, 13))),
                            {(10, 10, 10, 11, 11, 11, 5, 5, 6, 6), (10, 10, 10, 11, 11, 11, 3, 3, 4, 4),
                             (10, 10, 10, 11, 11, 11, 3, 3, 5, 5), (10, 10, 10, 11, 11, 11, 3, 3, 6, 6),
                             (10, 10, 10, 11, 11, 11, 4, 4, 5, 5), (10, 10, 10, 11, 11, 11, 4, 4, 6, 6)})
        h10 = [3, 3, 3, 4, 4, 4, 10, 10, 10, 10]
        self.assertEqual(getTrioPairStrips(h10), [(3, 3, 3, 4, 4, 4, 10, 10, 10, 10)])
        h11 = [7, 7, 7, 8, 8, 8, 9, 9, 9, 11, 11, 12, 12, 13, 13]
        self.assertEqual(getTrioPairStrips(h11, (3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 10, 10, 13, 13)),
                         [(7, 7, 7, 8, 8, 8, 9, 9, 9, 11, 11, 12, 12, 13, 13)])
        self.assertSetEqual(set(getTrioPairStrips(h11, (3, 3, 3, 4, 4, 4, 6, 6, 10, 10))),
                            {(7, 7, 7, 8, 8, 8, 9, 9, 11, 11), (7, 7, 7, 8, 8, 8, 9, 9, 12, 12),
                             (7, 7, 7, 8, 8, 8, 9, 9, 13, 13), (7, 7, 7, 8, 8, 8, 11, 11, 12, 12),
                             (7, 7, 7, 8, 8, 8, 11, 11, 13, 13), (7, 7, 7, 8, 8, 8, 12, 12, 13, 13),
                             (8, 8, 8, 9, 9, 9, 7, 7, 11, 11), (8, 8, 8, 9, 9, 9, 7, 7, 12, 12),
                             (8, 8, 8, 9, 9, 9, 7, 7, 13, 13), (8, 8, 8, 9, 9, 9, 11, 11, 12, 12),
                             (8, 8, 8, 9, 9, 9, 11, 11, 13, 13), (8, 8, 8, 9, 9, 9, 12, 12, 13, 13)})


if __name__ == '__main__':
    unittest.main()
