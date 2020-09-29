import unittest
from cdqn import CDQN
from utility import CardType


class CardTypeTest(unittest.TestCase):
    def test_card_type(self):
        self.assertEqual(CardType.UNRESTRICTED, CDQN.get_card_type(()))
        self.assertEqual(CardType.SOLO, CDQN.get_card_type((10,)))
        self.assertEqual(CardType.PAIR, CDQN.get_card_type((3, 3)))
        self.assertEqual(CardType.TRIO, CDQN.get_card_type((3, 3, 3)))
        self.assertEqual(CardType.TRIO_SOLO, CDQN.get_card_type((3, 3, 3, 10)))
        self.assertEqual(CardType.TRIO_PAIR, CDQN.get_card_type((3, 3, 3, 10, 10)))
        self.assertEqual(CardType.QUADPLEX, CDQN.get_card_type((5, 5, 5, 5)))
        self.assertEqual(CardType.QUADPLEX_SOLOS, CDQN.get_card_type((5, 5, 5, 5, 7, 9)))
        self.assertEqual(CardType.QUADPLEX_SOLOS, CDQN.get_card_type((5, 5, 5, 5, 7, 7)))
        self.assertEqual(CardType.QUADPLEX_PAIRS, CDQN.get_card_type((7, 7, 7, 7, 6, 6, 3, 3)))
        # self.assertEqual(CardType.QUADPLEX_PAIRS, CDQN.get_card_type((7, 7, 7, 7, 6, 6, 6, 6)))
        self.assertEqual(CardType.STRIP, CDQN.get_card_type((3, 4, 5, 6, 7)))
        self.assertEqual(CardType.STRIP, CDQN.get_card_type((3, 4, 5, 6, 7, 8, 9, 10)))
        self.assertEqual(CardType.PAIR_STRIP, CDQN.get_card_type((4, 4, 5, 5, 6, 6)))
        self.assertEqual(CardType.PAIR_STRIP, CDQN.get_card_type((4, 4, 5, 5, 6, 6, 7, 7)))
        self.assertEqual(CardType.TRIO_STRIP, CDQN.get_card_type((3, 3, 3, 4, 4, 4)))
        self.assertEqual(CardType.TRIO_STRIP, CDQN.get_card_type((3, 3, 3, 4, 4, 4, 5, 5, 5)))
        self.assertEqual(CardType.TRIO_SOLO_STRIP, CDQN.get_card_type((3, 3, 3, 7, 10, 12, 4, 4, 4, 5, 5, 5)))
        self.assertEqual(CardType.TRIO_PAIR_STRIP,
                         CDQN.get_card_type((3, 3, 3, 7, 7, 10, 10, 12, 12, 4, 4, 4, 5, 5, 5)))
        self.assertEqual(CardType.QUADPLEX_PAIRS,
                         CDQN.get_card_type((11, 11, 11, 11, 4, 9, 4, 9)))

if __name__ == '__main__':
    unittest.main()
