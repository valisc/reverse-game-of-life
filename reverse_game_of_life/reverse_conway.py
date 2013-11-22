#!/usr/bin/python3
# Code to solve Conway's Reverse Game of Life
# Released under GPL2 - see LICENCE for details

import numpy as np
from random import random
from random import choice



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
            self.end_board = ConwayBoard(board=end_board)
        else:
            self.start_board = ConwayBoard(board=start_board)
            self.end_board = ConwayBoard(board=end_board)

    def evaluate(self, predicted_board):
        ''' Computer error rate of predicted start board. '''
        if self.start_board is None:
            raise RuntimeError('Cannot evaluate an example with no start board.')
        # the magic of broadcasting ... maybe
        errors = sum([int(label)!=int(predict) for label,predict in np.nditer([self.start_board.board,predicted_board])])
        return 1.0*errors/(self.start_board.num_rows*self.start_board.num_cols)


# example creation routines
def create_random_board(num_rows,num_cols,fill_percent):
    ''' Returns a randomly initialzed ConwayBoard. '''
    return ConwayBoard(board=[[ALIVE if random()<fill_percent else DEAD for j in range(num_cols)] for i in range(num_rows)])

def create_example(num_rows, num_cols, delta, fill_percent,burn_in):
    ''' Create an example for reverse game of life. '''
    board = create_random_board(num_rows,num_cols,fill_percent)
    # do burn in
    for i in range(burn_in):
        board.advance()
    return Example(delta=delta, start_board=board.board)

def create_examples(num_examples=1, deltas=list(range(1,6)),num_rows=20,num_cols=20,burn_in=5,min_fill=0.01,max_fill=0.99):
    ''' Create examples for reverse game of life task. '''
    example_list = list()
    
    while len(example_list) < num_examples:
        e = create_example(num_rows,num_cols,choice(deltas), random()*(max_fill-min_fill)+min_fill,burn_in)
        # keep only if end_board is non-empty
        if (e.end_board.board==ALIVE).any():
            example_list.append(e)
            
    return example_list
        

    

class Classifier:
    ''' Default all-dead classifier and super-class for other reverse game of life solutions. 
    usage:
    >>> c = Classifier()
    >>> c.train(train_examples)
    >>> c.test(test_examples)
    '''

    
    def __init__(self):
        pass # all dead predictor needs to state

    def train(self,examples):
        pass # no training needed

    def predict(self,end_board, delta):
        ''' Returns classifiers prediction for ConwayBoard and given delta. '''
        # this dummy classifier predicts all dead
        prediction = np.empty([end_board.num_rows,end_board.num_cols],dtype=int)
        prediction.fill(DEAD)
        return prediction

    def test(self,examples):
        ''' Evaluate performance on test examples. Returns mean error rate. '''
        total_error_rate = 0
        for example in examples:
            total_error_rate += example.evaluate(self.predict(example.end_board,example.delta))
        return total_error_rate/len(examples)

            
# TODO - not working at all ... need to create some unit tests for __make_features and a new __make_train_data
class LocalClassifier:
    ''' Predict each cell 1 at a time. '''

    def __init__(self,window_size=1,off_board_value=0):
        ''' LocalClassifier constructor. window_size is number of cells (in all 4 directions) to include in the features for predicting the center cell. So window_size=1 uses 3x3 chunks, =2 uses 5x5 chunks, and so on. off_board_value is the value to represent features that are outside the board, defaults to 0 (ie DEAD).'''
        self.window_size = window_size
        self.off_board_value = off_board_value

    def __num_features(self):
        return (2*self.window_size+1)**2

    def __make_features(self,board,i,j):
        ''' Creates features describing the (i,j) position. Returns numpy array of features.'''
        features = np.empty(self.__num_features())
        index = 0
        (num_rows,num_cols) = board.shape
        for delta_row in range(-self.window_size,self.window_size+1):
            for delta_col in range(-self.window_size,self.window_size+1):
                if i+delta_row>=0 and i+delta_row<num_rows and j+delta_col>=0 and j+delta_col<num_cols:
                    features[index] = board[i+delta_row][j+delta_col]
                else:
                    features[index] = self.off_board_value
                index += 1
                
        return features

    def train(self, examples):
        # create training data (x - features, y - labels

        # assume all examples have same size
        num_rows = examples[0].start_board.num_rows
        num_cols = examples[0].end_board.num_cols
        
        x = np.empty([num_rows*num_cols*len(examples), self.__num_features()])
        y = np.empty([nuM_rows*num_cols*len(examples),1])
        
        index = 0
        for example in examples:
            for i in range(num_rows):
                for j in range(num_cols):
                    # training features (the window around i,j) for
                    # the i,j cell in this example
                    x[index] = self.__make_features(example.end_board.board,i,j)
                    y[index] = example.start_board.board[i][j]
                    index += 1
        return (x,y) # just for testing    
        
    
    def predict(self,end_board, delta):
        pass
