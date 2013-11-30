import numpy as np
from sklearn.ensemble import RandomForestClassifier
import  time
from collections import Counter
from copy import copy
from ..reverse_conway import DEAD,ALIVE

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer



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

    def test(self,examples,detailed_output=False):
        ''' Evaluate performance on test examples. Returns mean error rate. If detailed_output is true also prints performance for each delta.'''
        time_start = time.time()
        total_error_rate = 0
        delta_counts = dict()
        delta_error_rates = dict()
        
        for example in examples:
            error_rate = example.evaluate(self.predict(example.end_board,example.delta))
            total_error_rate += error_rate
            if example.delta not in delta_counts:
                delta_counts[example.delta] = 0
                delta_error_rates[example.delta] = 0                
            delta_counts[example.delta] += 1
            delta_error_rates[example.delta] += error_rate

        time_end = time.time()            
        print('testing completed in {0} seconds'.format(time_end-time_start))
        
        # print detailed performance
        if detailed_output:
            print ('delta\tn\terror rate')
            for delta in sorted(delta_counts.keys()):
                print('{0}\t{1}\t{2}'.format(delta,delta_counts[delta],delta_error_rates[delta]/delta_counts[delta]))
            print('{0}\t{1}\t{2}'.format('all',len(examples),total_error_rate/len(examples)))



        return total_error_rate/len(examples)

            

class LocalClassifier(Classifier):
    ''' Predict each cell 1 at a time. '''

    def __init__(self,window_size=1,off_board_value=0,clf=RandomForestClassifier()):
        ''' LocalClassifier constructor. window_size is number of cells (in all 4 directions) to include in the features for predicting the center cell. So window_size=1 uses 3x3 chunks, =2 uses 5x5 chunks, and so on. off_board_value is the value to represent features that are outside the board, defaults to 0 (ie DEAD). Copies of clf will be made and separate classifiers used for each unique delta.'''
        self.window_size = window_size
        self.off_board_value = off_board_value
        self.base_classifier = clf
        self.classifiers = dict()
        
    def _transform_board(self,board,transform=0):
        if transform==0:
            return board
        else:
            if transform>=4:
                return np.rot90(np.fliplr(board),transform%4)
            else:
                return np.rot90(board,transform%4)
            

    def _num_features(self):
        ''' Number of features used by this instance to classify each cell. '''
        return (2*self.window_size+1)**2

    def _make_neighbor_features_cell(self,board,i,j):
        ''' Create features of neighborhood around the (i,j) position. Returns numpy array of features.  NOTE: If getting neighborhoods for all positions on a board, use the _make_neighbor_features_board method which is much more efficient. '''
        features = np.empty((self.window_size*2+1)**2)
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
                

    def _make_neighbor_features_board(self,board):
        ''' Create features of neighborhood around all cells in board. Returns numpy 2d array of features in row-major order, first row is for (0,0) cell, next row for (0,1) cell, and so on. '''
        # put board in larger 2d array with off board values around the edges so slice notation will grab the neighborhoods
        square_size = (self.window_size*2+1)
        (num_rows,num_cols) = board.shape
        embed = np.empty([num_rows+self.window_size*2,num_cols+self.window_size*2])
        embed.fill(self.off_board_value)
        embed[self.window_size:(num_rows+self.window_size),self.window_size:(num_cols+self.window_size)] = board
        
        features = np.empty([num_rows*num_cols,square_size**2])
        # populate features
        for row in range(0,num_rows):
            for col in range(0,num_cols):
                features[row*num_cols + col] = embed[row:(row+square_size),col:(col+square_size)].flatten()
        
        return features
        

    def _make_features_cell(self,board,i,j):
        ''' Creates features describing the (i,j) position. Returns numpy array of features.'''        
        # just use neighborhood features
        features = self._make_neighbor_features_cell(board,i,j)
        return features

    def _make_features_board(self,board):
        ''' Create features describing all cells on board. Returns 2d numpy array of features in row-major order, so first row is for (0,0), second row for (0,1), etc. '''
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
                local_board = self._transform_board(example.end_board.board,t)
                start_board = self._transform_board(example.start_board.board,t)
                
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
                local_board = self._transform_board(example.end_board.board,t)
                start_board = self._transform_board(example.start_board.board,t)
                
                # faster, hopefully
                x[index:(index+num_rows*num_cols)] = self._make_features_board(local_board)
                y[index:(index+num_rows*num_cols)] = start_board.flatten()
                index += num_rows*num_cols
                # slower, don't use
                #for i in range(num_rows):
                #    for j in range(num_cols):
                #        # training features (the window around i,j) for
                #        # the i,j cell in this example
                #        x[index] = self._make_features_cell(local_board,i,j)
                #        y[index] = local_board[i][j]
                #        index += 1

        time_end = time.time()
        print('training data created in {0} seconds'.format(time_end-time_start))
        return (x,y) 

    def train(self, examples, use_transformations=False,use_weights=True):
        time_start = time.time()

        self.classifiers = dict()
        
        deltas = set([e.delta for e in examples])
        
        for delta in deltas:
            # create classifier with same params as base classifier
            clf = copy(self.base_classifier)
            
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

        
        
    
    def predict(self,end_board, delta):
        if delta not in self.classifiers:
            raise ValueError('Unable to predict delta='+str(delta)+', no training data for that delta')
        
        clf = self.classifiers[delta]

        (num_rows,num_cols) = end_board.board.shape
        
        # make all cell predictions at once (row-major order)
        x = np.empty([num_rows*num_cols,self._num_features()])
        # populate x
        x = self._make_features_board(end_board.board)
        
        # predict
        y_hat = clf.predict(x)
        # reshape into board
        predictions = y_hat.reshape((num_rows,num_cols))
        return predictions


class ANNClassifier(Classifier):
    ''' Predict entire board at once using neural network. 

    Note training continues from the current neural network.

    example usage:
    >>> from reverse_game_of_life import *
    >>> from pybrain.tools.shortcuts import buildNetwork
    # create 1000 examples all with delta=1
    >>> examples = create_examples(num_examples=1000,deltas=[1])
    # create classifier object
    >>> c = ANNClassifier(learning_rate=0.01,momentum=0.0,weight_decay=0.0,lr_decay=1.0)
    # set network with 400 inputs, 800 node hidden layer, 800 node hidden layer, 400 node output layer
    >>> c.net = buildNetwork(400,800,800,400)
    # train
    >>> c.train(examples[0:500],verbose=True)
    # training set error
    >>> c.test(examples[0:500])
    # test set error
    >>> c.test(examples[500:1000])
    '''
    
    def __init__(self,num_rows=20,num_cols=20,learning_rate=0.01,lr_decay=1.0,momentum=0.0,weight_decay=0.0):
        self.num_values = num_rows*num_cols
        self.net = buildNetwork(self.num_values,self.num_values,self.num_values)
        self.threshold = 0.5
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.weight_decay = weight_decay
        
    
    def set_network(self,net):
        self.net = net
        
    def train(self,examples,max_epochs=10,verbose=False):
        # convert examples to data set
        dataset = SupervisedDataSet(self.num_values,self.num_values)
        
        for e in examples:
            dataset.appendLinked(e.end_board.board.flatten(),e.start_board.board.flatten())
            

        trainer = BackpropTrainer(self.net,dataset,learningrate=self.learning_rate,lrdecay=self.lr_decay,momentum=self.momentum,weightdecay=self.weight_decay,verbose=verbose)
        trainer.trainUntilConvergence(maxEpochs=max_epochs, verbose=verbose, continueEpochs=2)

            
        # choose threshold to maximize accuracy
        
        # get predictions
        values = np.empty([len(dataset),self.num_values])

        for i in range(len(dataset)):
            values[i] = self.net.activate(dataset['input'][i])
            
        self.values = values

            
        targets = (dataset['target']>0.5).flatten() # make sure getting 1.0s as True and 0.0s as False
        self.targets = targets
        values = values.flatten()
        print(values.shape)
        
        indices = values.argsort()
        values = values[indices]
        targets = targets[indices]
        
        num_positives = sum(targets)
        num_negatives = len(targets)-num_positives
        tp = num_positives
        fp = num_negatives
        best_threshold = None
        best_error_rate = 1.0
        fpr_prev = 1.0
        tpr_prev = 1.0
        roc_area = 0.0
        # starting at smallest value
        for i in range(len(values)):
            if targets[i]:
                tp -= 1
            else:
                fp -= 1
            # can we put a break between the ith and i+1st value?
            if i+1<len(values) and values[i]!=values[i+1]:
                error_rate = (fp+num_positives-tp)/len(values)
                fpr = fp/num_negatives
                tpr = tp/num_positives
                # add to roc area
                roc_area += (fpr_prev-fpr)*(tpr+tpr_prev)/2
                fpr_prev = fpr
                tpr_prev = tpr
                if error_rate < best_error_rate:
                    best_threshold = (values[i]+values[i+1])/2
                    best_error_rate = error_rate
        
        if verbose:
            print('ROC Area: '+str(roc_area))
            print('Tuned threshold to '+str(best_threshold)+' with error rate of '+str(best_error_rate))
            
        self.threshold = best_threshold
        
        

    def predict(self,end_board,delta):
        ''' Make predictions on single board. '''
        # get predicted values from network
        values = self.net.activate(end_board.board.flatten())
        predictions = np.zeros(len(values))
        predictions[values>self.threshold] = ALIVE
        
        return predictions.reshape(end_board.board.shape)
        
