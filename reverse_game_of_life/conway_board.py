import numpy as np


DEAD = 0
ALIVE = 1

class ConwayBoard:
    def __init__(self,rows=0,cols=0,board=None):
        '''Conway Board constructor. Construct all dead board by specifiying rows and cols or initialize by specifying board.'''
        if board is None:
            self.num_rows = rows
            self.num_cols = cols
            self.board = np.empty([self.num_rows,self.num_cols],dtype=int)
            self.board.fill(DEAD)
        else:            
            self.num_rows = len(board)
            self.num_cols = len(board[0])
            # needs it's own copy so advance doesn't change other conwayboards
            self.board = np.array(board,copy=True)
           

    def __conway_rule(self,cur_state, live_neighbors):
        ''' Returns next state in Conway Game of Life for given current state and number of living neighbors. '''
        if cur_state==DEAD and live_neighbors==3:
            return ALIVE
        elif cur_state==ALIVE and (live_neighbors==3 or live_neighbors==2):
            return ALIVE
        else:
            return DEAD

    def __count_live_neighbors(self):
        '''
        Count number of living neighbors for all cells in current board.
        Returns array of all neighboring cells
        http://www.loria.fr/~rougier/teaching/numpy/scripts/game-of-life-numpy.py
        '''
        Z = np.pad(self.board, 1, 'constant')
        # summation of the 8 translations of the GoL board
        census = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
                  Z[1:-1,0:-2] +                Z[1:-1,2:] +
                  Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
        return census
    
    # advance board by 1 step of conway's game of life
    def advance(self):
        '''
        Advance board by 1 step using Conway's Game of Life rules.
        http://www.loria.fr/~rougier/teaching/numpy/scripts/game-of-life-numpy.py
        '''
        census = self.__count_live_neighbors()
        # Create boolean matrix based on GoL rules
        birth = (census == 3) & (self.board == 0)
        survive = ((census == 2) | (census == 3)) & (self.board == 1)

        self.board[...] = 0 # clear board to zero Ie no reallocate memory
        self.board[birth|survive] = 1 # Apply boolean mask to board

    def pretty_string(self):
        ''' Make ASCII representation of board. Use print(b.pretty_string()) for useful display. '''
        return '\n'.join([''.join([str(x) for x in row]) for row in self.board])

    def __str__(self):
        ''' Automatic string printing using str(conway_board). '''
        return self.pretty_string()
