# utility methods

from random import choice
import numpy as np

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
