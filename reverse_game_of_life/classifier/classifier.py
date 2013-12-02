import  time
import numpy as np

from ..conway_board import DEAD,ALIVE

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
        pass # no training neededo

    def predict(self,end_board, delta):
        ''' Returns classifiers prediction for ConwayBoard and given delta. '''
        # this dummy classifier predicts all dead
        prediction = np.empty([end_board.num_rows,end_board.num_cols],dtype=int)
        prediction.fill(DEAD)
        return prediction

    def test(self,examples,detailed_output=False):
        '''
        Evaluate performance on test examples. Returns mean error rate. If detailed_output is true also prints
        performance for each delta.
        '''
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
