import numpy as np

from ..conway_board import DEAD,ALIVE

from .classifier import Classifier
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

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

        trainer = BackpropTrainer(
            self.net,dataset,
            learningrate=self.learning_rate,
            lrdecay=self.lr_decay,
            momentum=self.momentum,
            weightdecay=self.weight_decay,
            verbose=verbose
        )
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
        predictions.fill(DEAD)
        predictions[values>self.threshold] = ALIVE
        
        return predictions.reshape(end_board.board.shape)
