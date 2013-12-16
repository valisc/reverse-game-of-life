import unittest
import numpy as np

from reverse_game_of_life import *


class TileClassifierTestCase(unittest.TestCase):
    ''' Test TileClassifier. '''
    
    def assert_array_equal(self, a1, a2):        
        self.assertTrue(np.array_equal(a1,a2),repr(a1) + " != " + repr(a2))

    def setUp(self):
        self.classifier = TileClassifier(tile_size=2)

    def test_store_tiles(self):
        examples = [KaggleExample(start_board=[[0,1],[1,1]],end_board=[[1,1],[1,1]],delta=1)]
        self.classifier.store_tiles(examples)

            
    def test_predict(self):
        examples = create_examples(num_examples=10,deltas=[1])

        self.classifier.store_tiles(examples)

        result = self.classifier.predict(examples[0].end_board,examples[0].delta)
        # right size
        self.assertEqual(result.shape,(20,20))
        # all 0s or 1s
        self.assertTrue(((result==1) | (result==0)).all())

        
