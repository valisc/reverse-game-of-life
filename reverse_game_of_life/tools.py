# utility methods

from random import choice
import numpy as np

from .conway_board import DEAD,ALIVE

def bootstrap_confidence_interval(a,f=lambda x:x.mean(),num_replicates=1000,alpha=0.05):
    ''' Empirical confidence interval for a statistic using the bootstrap.
    
    See http://en.wikipedia.org/wiki/Bootstrapping_(statistics) for details of the method.

    a - sample data to resample from
    f - function to calculate statistic on an np.array of samples, default: mean
    num_replicates - number of bootstrap replicates to create: default: 1000
    alpha - alpha from hypothesis testing to control the confidence interval width

    returns - (1-alpha)% confidence interval for the statistic
    
    '''
    # iterable to perform a resamples and calculation of the statistics
    iterable = (f(np.random.choice(a,size=len(a))) for i in range(num_replicates))
    # store in np.array
    d = np.fromiter(iterable,np.float,count=num_replicates)
    # in place sort
    np.ndarray.sort(d)
    # return the alpha/2 and 1-alpha/2 quantiles for the confidence interval
    return d[int(num_replicates*alpha/2)],d[int(num_replicates*(1-alpha/2))]


def board_to_int(board):
    '''Convert a board of ALIVE and DEAD to an (arbitrary sized) int for
    efficient lookup and storage. Can be reversed with int_to_board
    with the original shape.

    '''

    base = 1
    value = 0
    # order='C' to ensure consistent order through arrays of same size
    for c in np.nditer(board,order='C'):
        if c==ALIVE:
            value += base
        base *= 2
    return value

def int_to_board(value,num_rows,num_cols):
    '''Convert an int (arbitrary sized) representation to the original
    ALIVE and DEAD board. Can be reversed with board_to_int.

    '''
    a = np.empty(num_rows*num_cols)
    a[...] = DEAD
    temp_value = value
    for i in range(num_rows*num_cols):
        if temp_value%2 == 1:
            a[i] = ALIVE
        temp_value //= 2
    return a.reshape(num_rows,num_cols,order='C')
    

def transform_board(board,transform=0):
    '''
    Transforms board to one of the 8 possible rotations/flips.
    '''
    
    if transform<0 or transform>8:
        raise ValueError('transform_board only accepts transformations between 0 and 7 (inclusive)')
    
    if transform==0:
        return board
    else:
        if transform>=4:
            return np.rot90(np.fliplr(board),transform%4)
        else:
            return np.rot90(board,transform%4)


_inverse_transform = [0,3,2,1,4,5,6,7]
def inverse_transform(transform):
    '''
    Returns transform t' such that transform_board(transform_board(b,t),t')==b.
    '''
    if transform<0 or transform>8:
        raise ValueError('inverse_transform only accepts ints between 0 and 7 (inclusive)')
    
    return _inverse_transform[transform]
