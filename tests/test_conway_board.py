import numpy as np
import unittest

from reverse_game_of_life import *

class ConwayBoardTestCase(unittest.TestCase):
    ''' Test ConwayBoard class. '''

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

    def test_unique_arrays(self):
        # ensure each conway board (even if created from the same
        # nparray) has it's own array copy and so advancing one
        # doesn't advance the other
        start = np.array([[0,1,0],[1,0,0],[0,0,1]])
        b1 = ConwayBoard(board=start)
        b2 = ConwayBoard(board=start)
        b1.advance()
        self.assertFalse(np.array_equal(b1.board,b2.board),"ConwayBoards share common array.")
