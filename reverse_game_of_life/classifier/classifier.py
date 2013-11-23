import numpy as np
from sklearn.ensemble import RandomForestClassifier

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


    def _num_features(self):
        ''' Number of features used by this instance to classify each cell. '''
        return (2*self.window_size+1)**2

    def _make_features(self,board,i,j,transform=0):
        ''' Creates features describing the (i,j) position. Returns numpy array of features. Use transformation to convert feature representation to the 8 possible rotations and reflections.'''
        features = np.empty(self._num_features())
        index = 0
        (num_rows,num_cols) = board.shape
        for delta_row in range(-self.window_size,self.window_size+1):
            for delta_col in range(-self.window_size,self.window_size+1):
                if i+delta_row>=0 and i+delta_row<num_rows and j+delta_col>=0 and j+delta_col<num_cols:
                    features[index] = board[i+delta_row][j+delta_col]
                else:
                    features[index] = self.off_board_value
                index += 1
                
        if transform>0:
            # convert back to 2d array
            a = features.reshape(2*self.window_size+1,2*self.window_size+1)
            # transform
            # flip if needed
            if transform>=4:
                a = np.fliplr(a)
            # rotate
            a = np.rot90(a,transform%4)
            # back to 1d
            features = a.flatten()

        return features

    
    def make_training_data(self, examples, use_transformations=False):
        ''' Make training data (x,y) from these examples using the setting for this LocalClassifier. '''
        # assume all examples have same size
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
            for i in range(num_rows):
                for j in range(num_cols):
                    # training features (the window around i,j) for
                    # the i,j cell in this example
                    for t in range(copies_per):
                        x[index] = self._make_features(example.end_board.board,i,j,t)
                        y[index] = example.start_board.board[i][j]
                        index += 1
        return (x,y) 

    def train(self, examples, use_transformations=False):
        time_start = time.time()

        self.classifiers = dict()
        
        deltas = set([e.delta for e in examples])
        
        for delta in deltas:
            # training data for current delta            
            (train_x,train_y) = self.make_training_data([e for e in examples if e.delta==delta],use_transformations=use_transformations)
            # create classifier with same params as base classifier
            clf = copy(self.base_classifier)
            
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
        index = 0
        for row in range(num_rows):
            for col in range(num_cols):
                x[index] = self._make_features(end_board.board,row,col)
                index += 1
        
        # predict
        y_hat = clf.predict(x)
        # reshape into board
        predictions = y_hat.reshape((num_rows,num_cols))
        return predictions

