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
            self.board[...] = DEAD
        else:            
            self.num_rows = len(board)
            self.num_cols = len(board[0])
            # needs it's own copy so advance doesn't change other conwayboards
            self.board = np.array(board,copy=True)


    def __count_live_neighbors(self):
        '''
        Count number of living neighbors for all cells in current board.
        Returns array of all neighboring cells
        http://www.loria.fr/~rougier/teaching/numpy/scripts/game-of-life-numpy.py
        '''
        
        Z = np.pad(self.board, 1, 'constant',constant_values=DEAD)
        # summation of the 8 translations of the GoL board
        census = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
                  Z[1:-1,0:-2] +                Z[1:-1,2:] +
                  Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
        return census
    

    def advance(self):
        '''
        Advance board by 1 step using Conway's Game of Life rules.
        http://www.loria.fr/~rougier/teaching/numpy/scripts/game-of-life-numpy.py
        '''
        # only need to change board if something is alive, provides
        # slight performance boost for smallish boards that tend to
        # die out
        if (self.board==ALIVE).any():
            census = self.__count_live_neighbors()
            # Create boolean matrix based on GoL rules
            birth = (census == 3) & (self.board == DEAD)
            survive = ((census == 2) | (census == 3)) & (self.board == ALIVE)

            self.board[...] = DEAD # clear board to zero Ie no reallocate memory
            self.board[birth|survive] = ALIVE # Apply boolean mask to board


    def pretty_string(self):
        ''' Make ASCII representation of board. Use print(b.pretty_string()) for useful display. '''
        return '\n'.join([''.join([str(x) for x in row]) for row in self.board])


    def __str__(self):
        ''' Automatic string printing using str(conway_board). '''
        return self.pretty_string()
