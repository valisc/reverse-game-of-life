#!/usr/bin/python3
# Test code for reverse_conway.py
# Released under GPL2 - see LICENCE for details

# usage (auto discovery):
#    python3 -m unittest discover


import unittest

from reverse_conway import *

class ConwayBoardTestCase(unittest.TestCase):
    def test_constructor_blank(self):
        b = ConwayBoard(rows=3,cols=3)
        self.assertEqual(b.board,[[0,0,0],[0,0,0],[0,0,0]])

    def test_construct_board(self):
        board = [[1,1,1],[0,1,0],[1,0,0]]
        b = ConwayBoard(board=board)
        self.assertEqual(b.num_rows,3)
        self.assertEqual(b.num_cols,3)
        self.assertEqual(b.board,board)
        
        
    def test_advance_steady(self):
        ''' All dead stays dead. '''
        start = [[0,0,0],[0,0,0],[0,0,0]]
        end = [[0,0,0],[0,0,0],[0,0,0]]
        b = ConwayBoard(board=start)
        b.advance()
        self.assertEqual(b.board,end)

    def test_advance_death(self):
        ''' 1 neighbor dies. '''
        start = [[0,0,0],[1,1,0],[0,0,0]]
        end = [[0,0,0],[0,0,0],[0,0,0]]
        b = ConwayBoard(board=start)
        b.advance()
        self.assertEqual(b.board,end)
        

    def test_advance_survival(self):
        ''' 2 or 3 neighbors lives. '''
        start = [[0,0,0],[1,1,0],[1,1,0]]
        end = [[0,0,0],[1,1,0],[1,1,0]]
        b = ConwayBoard(board=start)
        b.advance()
        self.assertEqual(b.board,end)

    def test_advance_overcrowding(self):
        ''' More than 3 neighbors dies. '''
        start = [[1,1,1],[1,1,1],[1,1,1]]
        end = [[1,0,1],[0,0,0],[1,0,1]]
        b = ConwayBoard(board=start)
        b.advance()
        self.assertEqual(b.board,end)
        
    def test_advance_birth(self):
        ''' Exactly 3 neighbors spawns live cell. '''
        start = [[0,1,0],[1,0,0],[0,0,1]]
        end = [[0,0,0],[0,1,0],[0,0,0]]
        b = ConwayBoard(board=start)
        b.advance()
        self.assertEqual(b.board,end)

                                
                                    
        
