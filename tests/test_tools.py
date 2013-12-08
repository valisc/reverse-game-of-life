import unittest

import numpy as np

from reverse_game_of_life.tools import *

class ToolsTestCase(unittest.TestCase):
    ''' Test for tools module. '''

    def test_bootstrap_confidence_interval(self):
        # non-deterministic method, so just do sanity checks
        a = np.random.randn(1000)
        (lb,ub) = bootstrap_confidence_interval(a)
        # lower bound is less than or equal to upper bound
        self.assertLessEqual(lb,ub)

        # obtain interval for the median
        (lb,ub) = bootstrap_confidence_interval(a,lambda x:np.median(x))
        self.assertLessEqual(lb,ub)

    def test_board_to_int(self):
        # 010
        # 110
        # 100
        b = np.array([[0,1,0],[1,1,0],[1,0,0]])
        self.assertEqual(board_to_int(b),2+8+16+64)
        
        self.assertEqual(board_to_int(b[0:2,0:2]),2+4+8)
        
        self.assertEqual(board_to_int(b[1:3,1:3]),1)

    def test_int_to_board(self):
        # 100
        # 100
        # 000
        self.assertTrue(np.array_equal(int_to_board(9,3,3),[[1,0,0],[1,0,0],[0,0,0]]))

    def test_board_and_int_conversion(self):
        # confirm v=int_to_board(board_to_int(v))
        # for small 3x3 boards
        for i in range(0,2**9):
            self.assertEqual(board_to_int(int_to_board(i,3,3)),i,'3x3 board with value '+str(i)+' fails int_to_board(board_to_int(...))')

        # for a large 20x20 board (mostly to confirm aren't issues
        # with 64 bit ints, this requires arbitrary precision ints)
        a = np.zeros([20,20],np.int)
        a[18,15] = 1
        a[18,19] = 1
        a[10,2] = 1
        a[13,18] = 1
        a[2,3] = 1
        self.assertTrue(np.array_equal(int_to_board(board_to_int(a),20,20),a))

        
