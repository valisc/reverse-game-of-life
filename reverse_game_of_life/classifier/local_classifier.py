import numpy as np
from numpy.lib.stride_tricks import as_strided
import  time


from collections import Counter
from ..conway_board import DEAD,ALIVE
from .classifier import Classifier
from ..kaggle_example import create_examples
from ..tools import transform_board,inverse_transform
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import ParameterGrid
from random import shuffle

class LocalClassifier(Classifier):
    ''' Predict each cell 1 at a time. '''

    def __init__(self,window_size=1,off_board_value=0,clf=RandomForestClassifier(), test_use_transforms = False, test_threshold = 0.5):
        '''
        LocalClassifier constructor.
        window_size is number of cells (in all 4 directions) to include in the features for predicting the center cell.
        So window_size=1 uses 3x3 chunks, =2 uses 5x5 chunks, and so on. off_board_value is the value to represent
        features that are outside the board, defaults to 0 (ie DEAD). Copies of clf will be made and separate
        classifiers used for each unique delta.
        test_use_transforms - if true, uses all 8 board transforms and averages results to predict, uses test_treshold as proba threshold for ALIVE predictions
        '''
        self.window_size = window_size
        self.off_board_value = off_board_value
        self.base_classifier = clf
        self.classifiers = dict()
        self.test_use_transforms = test_use_transforms
        self.test_threshold = test_threshold
        
    def _num_features(self):
        ''' Number of features used by this instance to classify each cell. '''
        return (2*self.window_size+1)**2

    def _make_neighbor_features_board(self,board):
        '''
        Create features of neighborhood around all cells in board. Returns numpy 2d array of features in row-major
        order, first row is for (0,0) cell, next row for (0,1) cell, and so on.
        '''
        # put board in larger 2d array with off board values around the edges so slice notation will grab the neighborhoods
        square_size = (self.window_size*2+1)
        embed = np.pad(board, self.window_size, 'constant', constant_values=self.off_board_value) 
        
        size = square_size
        step = 1
        
        num_rows,num_cols = embed.shape
        row_stride, column_stride = embed.strides
        # populate features
        features = as_strided(
            embed,
            (int((num_rows - size + 1) / step), int((num_cols - size + 1) / step), size, size),
            (row_stride * step, column_stride * step, row_stride, column_stride)
        )
        
        return features.reshape(int(features.size/(size*size)), size*size)
        
    def _make_features_board(self,board):
        '''
        Create features describing all cells on board. Returns 2d numpy array of features in row-major order, so
        first row is for (0,0), second row for (0,1), etc.
        '''
        # just use neighborhood features for now
        return self._make_neighbor_features_board(board)

    def make_weighted_training_data(self, examples, use_transformations=False):
        ''' Make training data (x,y,w) from these examples using current settings. Returns weighted data set with no duplicates in (x,y). '''
        time_start = time.time()
        if len(examples)==0:
            raise ValueError('examples must be non-empty')
        if examples[0].end_board is None:
            raise ValueError('all examples must have non-None end_board')
        
        num_rows = examples[0].end_board.num_rows
        num_cols = examples[0].end_board.num_cols
        
        copies_per = 1
        if use_transformations:
            copies_per = 8
        
        data_counter = Counter()
        
        for example in examples:
            if example.start_board is None:
                raise ValueError('all examples must have non-None start_board')
            if example.end_board is None:
                raise ValueError('all examples must have non-None end_board')
            for t in range(copies_per):
                # do transform here to make sure features and labels line up
                local_board = transform_board(example.end_board.board,t)
                start_board = transform_board(example.start_board.board,t)
                
                local_x = self._make_features_board(local_board)
                local_y = start_board.flatten()
                # make tuples for each example and dump into counter
                data_counter.update([(tuple(features),label) for (features,label) in zip(local_x,local_y)])

        x = np.empty([len(data_counter),self._num_features()])
        y = np.empty(len(data_counter))
        w = np.empty(len(data_counter))
        index = 0
        for (features,label) in data_counter.keys():
            x[index] = np.array(features)
            y[index] = label
            w[index] = data_counter[(features,label)]
            index += 1

        time_end = time.time()
        print('training data created in {0} seconds'.format(time_end-time_start))
        return (x,y,w) 

    def make_training_data(self, examples, use_transformations=False):
        ''' Make training data (x,y) from these examples using the setting for this LocalClassifier. '''
        # assume all examples have same size
        time_start = time.time()
        if len(examples)==0:
            raise ValueError('examples must be non-empty')
        if examples[0].end_board is None:
            raise ValueError('all examples must have non-None end_board')
        
        num_rows = examples[0].end_board.num_rows
        num_cols = examples[0].end_board.num_cols
        
        copies_per = 1
        if use_transformations:
            copies_per = 8

        x = np.empty([copies_per*num_rows*num_cols*len(examples), self._num_features()])
        y = np.empty(copies_per*num_rows*num_cols*len(examples))
        
        index = 0
        for example in examples:
            if example.start_board is None:
                raise ValueError('all examples must have non-None start_board')
            if example.end_board is None:
                raise ValueError('all examples must have non-None end_board')
            for t in range(copies_per):
                # do transform here to make sure features and labels line up
                local_board = transform_board(example.end_board.board,t)
                start_board = transform_board(example.start_board.board,t)
                
                x[index:(index+num_rows*num_cols)] = self._make_features_board(local_board)
                y[index:(index+num_rows*num_cols)] = start_board.flatten()
                index += num_rows*num_cols

        time_end = time.time()
        print('training data created in {0} seconds'.format(time_end-time_start))
        return (x,y) 

    def train(self, examples, use_transformations=False,use_weights=True):
        time_start = time.time()

        self.classifiers = dict()
        
        deltas = set([e.delta for e in examples])
        
        for delta in deltas:
            # create classifier with same params as base classifier
            clf = clone(self.base_classifier)
            
            if use_weights:
                # weighted training data for current delta
                # training data for current delta            
                (train_x,train_y,train_w) = self.make_weighted_training_data([e for e in examples if e.delta==delta],use_transformations=use_transformations) 
                print ('delta={0}, training with {1} weighted examples'.format(delta,len(train_x)))
                # fit
                clf.fit(train_x,train_y,train_w)

            else:
                # training data for current delta            
                (train_x,train_y) = self.make_training_data([e for e in examples if e.delta==delta],use_transformations=use_transformations)            
                print ('delta={0}, training with {1} examples'.format(delta,len(train_x)))
                # fit
                clf.fit(train_x,train_y)

            # store
            self.classifiers[delta] = clf

        time_end = time.time()
        print('training completed in {0} seconds'.format(time_end-time_start))

    def _score(self,clf,x,y,w=None):
        '''General score function to handle weighted or unweighted examples,
        returns accuracy.

        '''

        if w is None:
            return clf.score(x,y)
        else:
            return np.dot(clf.predict(x)==y,w)/np.sum(w)
        
    def _fit_grid_point(self, x_train,y_train,w_train,x_test,y_test,w_test,base_estimator,parameters):
        '''Fit a classifier for particular parameter settings. Modeled after
        grid_seary.py from sklearn.
        '''

        # setup classifierN
        clf = clone(base_estimator)
        clf.set_params(**parameters)
        # fit
        if w_train is None:
            clf.fit(x_train,y_train)
        else:
            clf.fit(x_train,y_train,w_train)
        
        # evaluate
        score = self._score(clf,x_test,y_test,w_test)
            
        return score,parameters

    def tune_and_train(self, examples, param_grid, use_transformations=False, use_weights=True,tune_perc=0.5,verbosity=0):
        '''Tune parameters and then train using best parameters. 

        param_grid - list of dicts with setting configurations to try,
        e.g. [{'n_estimators':[10,20,30],'max_depth':[5,10,15]}]
        
        tune_perc - percentage of examples to use for tuning

        '''

        time_start = time.time()
        
        self.classifiers = dict()
        self.grid_scores = dict()
        self.best_scores = dict()
        self.best_params = dict()
        
        deltas = set([e.delta for e in examples])
        
        for delta in deltas:
            cur_examples = [e for e in examples if e.delta==delta]
            
            # train/test split
            cutoff = int(len(cur_examples)*tune_perc)
            # random ordering
            shuffle(cur_examples)
            
            if use_weights:
                (x_train,y_train,w_train) = self.make_weighted_training_data(cur_examples[0:cutoff],use_transformations=use_transformations)
                (x_test,y_test,w_test) = self.make_weighted_training_data(cur_examples[cutoff:len(examples)],use_transformations=use_transformations)
            else:
                (x_train,y_train) = self.make_training_data(cur_examples[0:cutoff],use_transformations=use_transformations)
                (x_test,y_test) = self.make_training_data(cur_examples[cutoff:len(examples)],use_transformations=use_transformations)
                w_test = None
                w_train = None
                                        
            # fit each parameter setting using ParameterGrid from
            # sklearn.grid_search to cycle through the possibilities
            self.grid_scores[delta] = [self._fit_grid_point(x_train,y_train,w_train,x_test,y_test,w_test,self.base_classifier,parameters) for parameters in ParameterGrid(param_grid)]

            (best_scores,best_params) = sorted(self.grid_scores[delta],key=lambda x:x[0], reverse=True)[0]
            if verbosity>1:
                print('best params for delta={0} are {1} with score of {2}'.format(delta,str(best_params),str(best_scores)))
                
            self.best_scores[delta] = best_scores
            self.best_params[delta] = best_params

            # fit on entire data
            clf = clone(self.base_classifier)
            clf.set_params(**best_params)
            
            if use_weights:
                (x,y,w) = self.make_weighted_training_data(cur_examples,use_transformations=use_transformations)
                clf.fit(x,y,w)
            else:
                (x,y) = self.make_training_data(cur_examples,use_transformations=use_transformations)
                clf.fit(x,y)
                
            # store
            self.classifiers[delta] = clf
        
        time_end = time.time()
        if verbosity>0:
            print('tuning and training completed in {0} seconds'.format(time_end-time_start))
        
    def predict(self,end_board, delta):
        if delta not in self.classifiers:
            raise ValueError('Unable to predict delta='+str(delta)+', no training data for that delta')

        try:
            self.test_use_transforms
        except AttributeError:
            # old version without test_use_Transforms and test_threshold, so add them
            self.test_use_transforms = False
            self.test_threshold = 0.5
            
        clf = self.classifiers[delta]

        (num_rows,num_cols) = end_board.board.shape

        if self.test_use_transforms:
            # predict on all 8 transforms of board and average the probabilities
            y_hat = np.zeros((num_rows,num_cols))

            for t in range(8):
                #temp_array = np.array(transform_board(end_board.board,t))
                x = self._make_features_board(transform_board(end_board.board,t))
                #print('x.shape='+str(x.shape))
                y_cur = clf.predict_proba(x)[:,1]
                #y_cur = y_cur[:,1] # grab second column, probs for the positive (1) class
                #print('y_cur.shape=',str(y_cur.shape))
                y_cur = y_cur.reshape((num_rows,num_cols))
                #print('y_cur.shape=',str(y_cur.shape))
                # add to predictions, but transform back to original board
                y_add = transform_board(y_cur,inverse_transform(t))
                #print('y_add.shape=',str(y_add.shape))
                y_hat = y_hat+0.125*y_add
                

            predictions = np.empty((num_rows,num_cols),dtype=int)
            predictions[...] = DEAD
            predictions[y_hat>self.test_threshold] = ALIVE

            return predictions
        
        else:
            # make all cell predictions at once (row-major order)
            x = np.empty([num_rows*num_cols,self._num_features()])
            # populate x
            x = self._make_features_board(end_board.board)
            
            # predict
            y_hat = clf.predict(x)
            
            # reshape into board
            predictions = y_hat.reshape((num_rows,num_cols))
            return predictions



class FreshEnsembleClassifier(LocalClassifier):
    ''' Generate new examples for every tree in the forest. '''
    
    def __init__(self,window_size=1,off_board_value=0,clf=DecisionTreeClassifier(),n_estimators=10):
        # superclass constructor
        LocalClassifier.__init__(self,window_size=window_size,off_board_value=off_board_value,clf=clf)
        self.n_estimators=n_estimators
        
        
    def train(self,num_examples,deltas=list(range(1,6)),use_transformations=False,use_weights=True,verbosity=0):
        '''Train ensemble of classifiers using newly generated data for every
        member of the ensemble.

        '''
        time_start = time.time()
        
        self.classifiers = dict()

        for delta in deltas:
            self.classifiers[delta] = [] # list for ensemble of classifiers
            for i in range(self.n_estimators):
                # base classifier
                clf = clone(self.base_classifier)
                
                if use_weights:
                    train_x,train_y,train_w = self.make_weighted_training_data(create_examples(num_examples=num_examples,deltas=[delta]),use_transformations=use_transformations)
                    if verbosity>1:
                        print ('delta={0}, #{1}: training with {2} weighted examples'.format(delta,i,len(train_x)))
                    # fit
                    clf.fit(train_x,train_y,sample_weight=train_w)
                else:
                    train_x,train_y = self.make_training_data(create_examples(num_examples=num_examples,deltas=[delta]),use_transformations=use_transformations)
                    if verbosity>1:
                        print ('delta={0}, #{1}: training with {2} examples'.format(delta,i,len(train_x)))
                    clf.fit(train_x,train_y)
                    
                self.classifiers[delta].append(clf)
        
        time_end = time.time()
        if verbosity>0:
            print('training completed in {0:.1f} seconds'.format(time_end-time_start))

            

    def predict(self,end_board,delta):
        if delta not in self.classifiers:
            raise ValueError('Unable to predict delta='+str(delta)+', no classifiers for that delta')

        
        # make all predictions at once (row-major order)
        x = self._make_features_board(end_board.board)
        
        predictions = np.zeros([len(x),2])
        
        for clf in self.classifiers[delta]:
            predictions += clf.predict_proba(x)
            
        ret = np.empty(len(predictions))
        ret[...] = DEAD # default to all dead        
        ret[predictions[:,0]<predictions[:,1]] = ALIVE # grab alive predictions
        
        return ret.reshape(end_board.board.shape)

                
