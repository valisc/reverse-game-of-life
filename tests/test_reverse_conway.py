#!/usr/bin/python3
# Test code for reverse_conway.py
# Released under GPL2 - see LICENCE for details

# usage:
#  $ nosetests

import numpy as np

import unittest

from reverse_game_of_life import *

class ConwayBoardTestCase(unittest.TestCase):
    def assert_array_equal(self, a1, a2):        
        self.assertTrue(np.array_equal(a1,a2),repr(a1) + " != " + repr(a2))

    def test_constructor_blank(self):
        ''' Dead board constructor. '''
        b = ConwayBoard(rows=3,cols=3)
        self.assert_array_equal(b.board,[[0,0,0],[0,0,0],[0,0,0]])

    def test_construct_board(self):
        ''' Initialize constructor. '''
        board = [[1,1,1],[0,1,0],[1,0,0]]
        b = ConwayBoard(board=board)
        self.assert_array_equal(b.board,board)
        
        
    def test_advance_steady(self):
        ''' All dead stays dead. '''
        start = [[0,0,0],[0,0,0],[0,0,0]]
        end = [[0,0,0],[0,0,0],[0,0,0]]
        b = ConwayBoard(board=start)
        b.advance()
        self.assert_array_equal(b.board,end)

    def test_advance_death(self):
        ''' 1 neighbor dies. '''
        start = [[0,0,0],[1,1,0],[0,0,0]]
        end = [[0,0,0],[0,0,0],[0,0,0]]
        b = ConwayBoard(board=start)
        b.advance()
        self.assert_array_equal(b.board,end)
        

    def test_advance_survival(self):
        ''' 2 or 3 neighbors lives. '''
        start = [[0,0,0],[1,1,0],[1,1,0]]
        end = [[0,0,0],[1,1,0],[1,1,0]]
        b = ConwayBoard(board=start)
        b.advance()
        self.assert_array_equal(b.board,end)

    def test_advance_overcrowding(self):
        ''' More than 3 neighbors dies. '''
        start = [[1,1,1],[1,1,1],[1,1,1]]
        end = [[1,0,1],[0,0,0],[1,0,1]]
        b = ConwayBoard(board=start)
        b.advance()
        self.assert_array_equal(b.board,end)
        
    def test_advance_birth(self):
        ''' Exactly 3 neighbors spawns live cell. '''
        start = [[0,1,0],[1,0,0],[0,0,1]]
        end = [[0,0,0],[0,1,0],[0,0,0]]
        b = ConwayBoard(board=start)
        b.advance()
        self.assert_array_equal(b.board,end)

    def test_example_constructor(self):
        ''' start and end board specified '''
        start_board = [[0,0],[1,0]]
        end_board = [[0,0],[0,0]]
        example = Example(delta=1,start_board=start_board, end_board=end_board)
        self.assertEqual(example.delta,1)
        self.assert_array_equal(example.start_board.board,start_board)
        self.assert_array_equal(example.end_board.board,end_board)
        
    def test_example_constructor_error(self):
        ''' Constructor fails when no board specified. '''
        self.assertRaises(ValueError, Example,(1))
        
    def test_example_evaluate(self):
        ''' Example.evaluate() 3x3 board. '''
        start_board = np.array([[0,1,0],[1,1,0],[0,1,1]])
        end_board = np.array([[1,1,0],[1,0,0],[1,1,1]])
        e = Example(delta=1,start_board=start_board,end_board=end_board)
        self.assertAlmostEqual(e.evaluate([[0,0,0],[0,0,0],[0,0,0]]),5.0/9)
        self.assertAlmostEqual(e.evaluate(start_board),0)
        self.assertAlmostEqual(e.evaluate(end_board),3.0/9)



    def test_localclassifier_make_features(self):
        ''' LocalClassifier._make_features() '''
        # 000
        # 011
        # 011
        lc = LocalClassifier(window_size=1,off_board_value=0)
        board = np.array([[0,0,0],[0,1,1],[0,1,1]])
        self.assert_array_equal(lc._make_features(board,0,0),
                                [0,0,0,0,0,0,0,0,1])
        self.assert_array_equal(lc._make_features(board,1,1),
                                [0,0,0,0,1,1,0,1,1])
        
    def test_localclassifier_make_features2(self):
        ''' LocalClassifier._make_features() '''
        # 000
        # 011
        # 100
        lc = LocalClassifier(window_size=1,off_board_value=-1)
        board = np.array([[0,0,0],[0,1,1],[1,0,0]])
        self.assert_array_equal(lc._make_features(board,0,0,transform=0),
                                [-1,-1,-1,-1,0,0,-1,0,1])
        all_transforms = [
            [0,0,0,0,1,1,1,0,0], # original
            [1,0,0,0,1,0,0,1,0], # clockwise 90
            [0,0,1,1,1,0,0,0,0], # clockwise 180
            [0,1,0,0,1,0,0,0,1], # clockwise 270
            [0,0,0,1,1,0,0,0,1], # flip vertical
            [0,1,0,0,1,0,1,0,0], # flip and 90
            [1,0,0,0,1,1,0,0,0], # flip and 180
            [0,0,1,0,1,0,0,1,0]] # flip and 270

        received = set([tuple(lc._make_features(board,1,1,transform=i)) for i in range(8)])
        expected = set([tuple([float(i) for i in x]) for x in all_transforms])
        self.assertEqual(received,expected)

    def test_localclassifier_make_training_data(self):
        '''' LocalClassifier.make_training_data() '''
        # 10
        # 01
        lc = LocalClassifier(window_size=1,off_board_value=0)
        # NOTE: end board is not one conway step of start board
        examples = [Example(delta=1,start_board=[[1,0],[0,1]],end_board=[[1,0],[0,1]])]
        expected_x = np.array([ [0,0,0,0,1,0,0,0,1],
                                [0,0,0,1,0,0,0,1,0],
                                [0,1,0,0,0,1,0,0,0],
                                [1,0,0,0,1,0,0,0,0]])
        expected_y = np.array([1,0,0,1])
        
        (x,y) = lc.make_training_data(examples)
        self.assert_array_equal(x,expected_x)
        self.assert_array_equal(y,expected_y)
                                    
        
