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

    
    def test_transform_board(self):
        # 000
        # 011
        # 100
        board = np.array([[0,0,0],[0,1,1],[1,0,0]])
        
        # order of transforms is not important, just that we get all 8
        all_transforms = [
            [0,0,0,0,1,1,1,0,0], # original
            [1,0,0,0,1,0,0,1,0], # clockwise 90
            [0,0,1,1,1,0,0,0,0], # clockwise 180
            [0,1,0,0,1,0,0,0,1], # clockwise 270
            [0,0,0,1,1,0,0,0,1], # flip vertical
            [0,1,0,0,1,0,1,0,0], # flip and 90
            [1,0,0,0,1,1,0,0,0], # flip and 180
            [0,0,1,0,1,0,0,1,0]] # flip and 270
        
        received = [list(transform_board(board,i).flatten()) for i in range(8)]
        self.assertEqual(sorted(received),sorted(all_transforms))


    def test_inverse_transform(self):
        # 000
        # 011
        # 100
        board = np.array([[0,0,0],[0,1,1],[1,0,0]])

        # applying transform and its inverse should bring back to original board
        for t in range(8):
            self.assertTrue(np.array_equal(transform_board(transform_board(board,t),inverse_transform(t)),board), repr(t) + ' and inverse do not combine to identity')
            
    def test_inverse_transform2(self):
        # 0100
        # 1010
        # 1101
        # 1111
        board = np.array([[0,1,0,0],
                          [1,0,1,0],
                          [1,1,0,1],
                          [1,1,1,1]])

        # applying transform and its inverse should bring back to original board
        for t in range(8):
            self.assertTrue(np.array_equal(transform_board(transform_board(board,t),inverse_transform(t)),board), repr(t) + ' and inverse do not combine to identity')
