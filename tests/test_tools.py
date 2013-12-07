import unittest

import numpy as np

from reverse_game_of_life.tools import *

class ToolsTestCase(unittest.TestCase):
    ''' Test for tools module. '''

    def test_bootstrap_confidence_interval(self):
        # non-deterministic method, so just do sanity checks
        a = np.random.randn(1000)
        (lb,ub) = bootstrap_confidence_interval(a)
        # lower bound is less than or equal to upper bound
        self.assertLessEqual(lb,ub)

        # obtain interval for the median
        (lb,ub) = bootstrap_confidence_interval(a,lambda x:np.median(x))
        self.assertLessEqual(lb,ub)

        
        
        
