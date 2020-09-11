import unittest
from doudizhu_env import DetailEnv
from utility import CardType
import numpy as np


class TestConversion(unittest.TestCase):
    def test_conversion(self):
        log_row = (0, (CardType.TRIO, (3, 3, 3)))
        ans = np.zeros((71,))
        ans[0] = 1
        ans[6] = 1
        ans[19] = 1
        self.assertEqual((ans == DetailEnv.encode_row(log_row)).all(), True)

        log_row = (2, (CardType.SOLO, (17,)))
        ans = np.zeros((71,))
        ans[2] = 1
        ans[4] = 1
        ans[70] = 1
        self.assertEqual((ans == DetailEnv.encode_row(log_row)).all(), True)

        log_row = (1, (CardType.PAIR_STRIP, (12, 12, 13, 13, 14, 14)))
        ans = np.zeros((71,))
        ans[1] = 1
        ans[13] = 1
        ans[-17] = 1
        ans[-13] = 1
        ans[-9] = 1
        # print(ans)
        # print(DetailEnv.to_np_log_row(log_row))
        # print(ans == DetailEnv.to_np_log_row(log_row))
        self.assertEqual((ans == DetailEnv.encode_row(log_row)).all(), True)



if __name__ == '__main__':
    unittest.main()
