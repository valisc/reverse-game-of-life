#!/usr/bin/python3
# Code to solve Conway's Reverse Game of Life
# Released under GPL2 - see LICENCE for details

import numpy as np


DEAD = 0
ALIVE = 1


class ConwayBoard:
    def __init__(self,rows=0,cols=0,board=None):
        '''Conway Board constructor. Construct all dead board by specifiying rows and cols or initialize by specifying board.'''
        # TODO - um, is this the best way to "overload" a constructor? -- KFB
        if board is not None:
            self.num_rows = len(board)
            self.num_cols = len(board[0])
            self.board = np.asarray(board)
        else:            
            self.num_rows = rows
            self.num_cols = cols
            self.board = np.empty([self.num_rows,self.num_cols],dtype=int)
            self.board.fill(DEAD)

    def __conway_rule(self,cur_state, live_neighbors):
        ''' Returns next state in Conway Game of Life for given current state and number of living neighbors. '''
        if cur_state==DEAD and live_neighbors==3:
            return ALIVE
        elif cur_state==ALIVE and (live_neighbors==3 or live_neighbors==2):
            return ALIVE
        else:
            return DEAD

    def __count_live_neighbors(self,row,col):
        ''' Count number of living neighbors of cell (row,col) in current board. '''
        count = 0
        for r in range(row-1,row+2):
            for c in range(col-1,col+2):
                if (r!=row or c!=col) and r>=0 and r<self.num_rows and c>=0 and c<self.num_cols and self.board[r][c]==ALIVE:
                    count += 1
        return count
                    
    
    # advance board by 1 step of conway's game of life
    def advance(self):
        ''' Advance board by 1 step using Conway's Game of Life rules.'''
        new_board = np.array([ [self.__conway_rule(self.board[row][col],self.__count_live_neighbors(row,col)) for col in range(self.num_cols)] for row in range(self.num_rows)])
        self.board = new_board

    def pretty_string(self):
        ''' Make ASCII representation of board. Use print(b.pretty_string()) for useful display. '''
        return '\n'.join([''.join([str(x) for x in row]) for row in self.board])



class Example:
    ''' A reverse conway example, stores the starting board (if known), the ending board, and the delta steps between start and end. '''

    def __init__(self,delta, start_board=None,end_board=None):
        ''' Construct an example, delta is required, start and end, or just one board can be supplied. If only start is supplied, the end board is computed, if only end board is supplied then no evaluation is possible. '''
        self.delta = delta
        if start_board is None and end_board is None:
            raise ValueError('start_board and end_board cannot both be None')
        elif end_board is None:
            self.start_board = ConwayBoard(board=start_board)
            # compute the ending board
            self.end_board = ConwayBoard(board=start_board)
            for i in range(self.delta):
                self.end_board.advance()
        elif start_board is None:
            self.start_board = None
            self.end_board = ConwayBoard(end_board)
        else:
            self.start_board = ConwayBoard(start_board)
            self.end_board = ConwayBoard(end_board)

    def evaluate(self, predicted_board):
        ''' Computer error rate of predicted start board. '''
        if start_board is None:
            raise RuntimeError('Cannot evaluate an example with no start board.')
        # the magic of broadcasting ... maybe
        errors = sum([label==predict for label,predict in np.nditer([start_board,predicted_board])])
        return 1.0*errors/(self.num_rows*self.num_cols)

            

    

        
