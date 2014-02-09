import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn import clone

from ..conway_board import DEAD,ALIVE
from ..tools import transform_board,inverse_transform
from .classifier import Classifier

class GlobalClassifier(Classifier):
    ''' Predict each cell 1 at a time but use entire board as features. '''

    def __init__(self,clf=RandomForestClassifier()):
        '''
        GlobalClassifier constructor.
        '''
        self.base_classifier = clf
        self.classifiers = dict()
    
    def train(self, examples,verbosity=0):
        time_start = time.time()

        self.classifiers = dict()

        deltas = set([e.delta for e in examples])
        
        for delta in deltas:
            self.classifiers[delta] = dict()
            
            # create classifiers for delta
            cur_examples = [e for e in examples if e.delta==delta]
            num_rows,num_cols = cur_examples[0].start_board.board.shape

            # main feature vectors and ys (with all transforms)
            x = np.empty((len(cur_examples)*8,num_rows*num_cols),dtype=int)
            start_boards = []
            for i in range(len(cur_examples)):
                for transform in range(8):
                    x[i*8 + transform] = transform_board(cur_examples[i].end_board.board,transform).reshape((1,num_rows*num_cols))
                    start_boards.append(transform_board(cur_examples[i].start_board.board,transform))
            print(x.shape)
            print(len(start_boards))

            assert(num_rows==num_cols)
            # train the required (55) classifiers
            for row in range((num_rows+1)//2):
                for col in range(row,(num_rows+1)//2):
                    print('training to predict '+str((row,col)))
                    y = np.array([b[row][col] for b in start_boards])
                    # train
                    clf = clone(self.base_classifier)
                    clf.fit(x,y)
                    # store
                    self.classifiers[delta][(row,col)] = clf
        time_end = time.time()
        if verbosity>0:
            print('training completed in {0:.1f} seconds'.format(time_end-time_start))
            
                    
    def predict(self,end_board,delta):
        ''' Predict starting board for a single end board and delta. '''
        num_rows,num_cols = end_board.board.shape
        
        y_hat = np.zeros((num_rows,num_cols))
        counts = np.zeros((num_rows,num_cols))

        for transform in range(8):
            x = transform_board(end_board.board,transform).reshape((1,num_rows*num_cols))
            y_cur = np.zeros((num_rows,num_cols))
            c = np.zeros((num_rows,num_cols))
            for row in range((num_rows+1)//2):
                for col in range(row,(num_rows+1)//2):
                    y_cur[row,col] = self.classifiers[delta][(row,col)].predict_proba(x)[0,1]
                    c[row,col] += 1
            y_hat += transform_board(y_cur,inverse_transform(transform))
            counts += transform_board(c,inverse_transform(transform))
            #print('transform='+str(transform))
            #print('y_hat='+str(y_hat))
            #print('counts='+str(counts))

        
        y_hat /= counts
        predictions = np.empty((num_rows,num_cols),dtype=int)
        predictions[...] = DEAD
        predictions[y_hat>0.5] = ALIVE

        return predictions
        
        
