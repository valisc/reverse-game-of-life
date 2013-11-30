#!/usr/bin/python3
# Code to solve Conway's Reverse Game of Life
# Released under GPL2 - see LICENCE for details

import csv
import time
from copy import copy
import numpy as np
from random import random
from random import choice

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


class Example:
    ''' A reverse conway example, stores the starting board (if known), the ending board, and the delta steps between start and end. '''

    def __init__(self,delta, start_board=None,end_board=None,kaggle_id=None):
        ''' Construct an example, delta is required, start and end, or just one board can be supplied. If only start is supplied, the end board is computed, if only end board is supplied then no evaluation is possible. '''
        self.kaggle_id = kaggle_id
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

    def __str__(self):
        ''' A human readable string representation. '''
        return 'delta='+str(self.delta)+'\nstart board:\n'+str(self.start_board)+'\nend_board:\n'+str(self.end_board)


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
        

# example loading routines
def load_examples(file_name,num_rows=20,num_cols=20):
    ''' Load examples from .csv files as formatted by Kaggle. '''
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        if not (len(header)==2+num_rows*num_cols or len(header)==2+2*num_rows*num_cols):
            raise RuntimeError('Unexpected number of columns ('+str(len(header))+ ') in example file.')
        
        examples = list()
        for ex_data in reader:
            ex_id = int(ex_data[0])
            delta = int(ex_data[1])
            start_board = None
            end_board = None
            index = 2
            while index<len(ex_data):
                # read board
                a = np.array(ex_data[index:(index+num_rows*num_cols)],dtype='int')
                # reshape
                board = a.reshape((num_rows,num_cols))
                # store in appropriate variable
                if header[index]=='start.1':
                    start_board = board
                elif header[index]=='stop.1':
                    end_board = board
                else:
                    raise RuntimeError('Unknown column header = '+header[index])

                index += num_rows*num_cols
                
            examples.append(Example(kaggle_id=ex_id, delta=delta, start_board=start_board, end_board=end_board))

        
        return examples

# kaggle format example writing, ie predictions
# get predictions and save from a classifer by
# >>> predictions = [classifier.predict(e.end_board,e.delta) for e in examples]
# >>> save_examples('file.csv',examples,predictions)
def save_examples(file_name,examples,predictions):
    ''' Save predictions in kaggle format. '''
    with open(file_name,'w') as csvfile:
        writer = csv.writer(csvfile)
        
        (num_rows,num_cols) = examples[0].end_board.board.shape
        # header row
        header = ['id'] + ['start.'+str(i+1) for i in range(num_rows*num_cols)]
        writer.writerow(header)
        for (example,prediction) in zip(examples,predictions):
            row = [str(example.kaggle_id)] + [str(int(v)) for v in prediction.flatten()]
            writer.writerow(row)

        
