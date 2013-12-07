import  time
import numpy as np

from math import sqrt

from ..conway_board import DEAD,ALIVE
from ..tools import bootstrap_confidence_interval

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

    def test(self,examples,verbosity=0,detailed_output=False,output_precision=4,conf_int_method='normal'):
        '''Evaluate performance on test examples. Returns mean error rate. If
        detailed output is True or verbosity>1, displays error rate,
        standard deviation (sd), and 95% confidence interval  for each
        delta.

        detailed_output - prints per delta stats if True
        output_precision - number of decimal places to use in detailed
        output verbosity - 0 prints nothing (default),
                           1 prints timing info,
                           >1 more details 
        conf_int_method - 'bootstrap' for empirical bootstrap CI
                          'normal' to assume normally distributed error rates 
                          (not actually true) but much faster than bootstrap 
                          and appears to provide similar intervals (default)

        '''
        time_start = time.time()
        total_error_rate = 0
        delta_counts = dict()
        delta_error_rates = dict()
        
        error_rates = [e.evaluate(self.predict(e.end_board,e.delta)) for e in examples]

        time_end = time.time()            
        if verbosity>0:
            print('testing completed in {0:.1f} seconds'.format(time_end-time_start))

        if verbosity>1 or detailed_output:            
            # print detailed performance
            z_value = 1.96 # for 95% confidence intervals
            # create format strings
            header_format = '{0:<7} {1:<7} {2:<12} {3:<'+str(output_precision+4)+'} {4}'
            data_format = '{0:<7} {1:<7d} {2:<12.'+str(output_precision)+'f} {3:<'+str(output_precision+4)+'.'+str(output_precision)+'f} ({4:.'+str(output_precision)+'f},{5:.'+str(output_precision)+'f})'
            
            print(header_format.format('delta','n','error rate','sd','95% CI'))
            for delta in sorted(set([e.delta for e in examples])):
                cur_error_rates = [error_rate for error_rate,example in zip(error_rates,examples) if example.delta==delta]
               

                mean = sum(cur_error_rates)/len(cur_error_rates)
                variance = sum([x*x for x in cur_error_rates])/len(cur_error_rates) - mean**2
                standard_error_mean = sqrt(variance/len(cur_error_rates))
                if conf_int_method=='normal':
                    lb = mean-z_value*standard_error_mean
                    ub = mean+z_value*standard_error_mean
                elif conf_int_method=='bootstrap':
                    lb,ub = bootstrap_confidence_interval(cur_error_rates)
                else:
                    raise ValueError('Unknown conf_int_method='+str(conf_int_method))
                
                print(data_format.format(delta,len(cur_error_rates),mean,sqrt(variance),lb,ub))
                
            # overall
            all_mean = sum(error_rates)/len(error_rates)
            all_variance = sum([x*x for x in error_rates])/len(error_rates) - all_mean**2
            all_standard_error_mean = sqrt(all_variance/len(error_rates))
            if conf_int_method=='normal':
                lb = all_mean-z_value*all_standard_error_mean
                ub = all_mean+z_value*all_standard_error_mean
            elif conf_int_method=='bootstrap':
                lb,ub = bootstrap_confidence_interval(error_rates)
            else:
                raise ValueError('Unknown conf_int_method='+str(conf_int_method))
                
            print(data_format.format('all',len(error_rates),all_mean,sqrt(all_variance),lb,ub))
            

        return sum(error_rates)/len(error_rates)
