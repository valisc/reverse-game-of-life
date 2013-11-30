import unittest
import numpy as np

from reverse_game_of_life import *

class ClassifierTestCase(unittest.TestCase):
    ''' Test Classifier. '''
    
    def assert_array_equal(self, a1, a2):        
        self.assertTrue(np.array_equal(a1,a2),repr(a1) + " != " + repr(a2))

    def setUp(self):
        self.classifier = Classifier()
        
    def test_train(self):
        # shouldn't do anything, just make sure it doesn't crash
        self.classifier.train(create_examples(num_examples=1,deltas=[1]))

    def test_predict(self):
        board = ConwayBoard(rows=3,cols=3)
        expected = np.zeros(9).reshape(3,3)
        received = self.classifier.predict(board,1)
        self.assert_array_equal(received,expected)
        
    def test_test(self):
        examples = [KaggleExample(delta=1,start_board=[[0,1,1],[1,1,0],[0,0,1]])]
        self.assertAlmostEqual(self.classifier.test(examples), 5./9)
        

        
        
