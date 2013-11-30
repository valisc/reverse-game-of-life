import numpy as np
import unittest

from reverse_game_of_life import *

class LocalClassifierTestCase(unittest.TestCase):
    ''' Test LocalClassifier. '''
    
    def assert_array_equal(self, a1, a2):        
        self.assertTrue(np.array_equal(a1,a2),repr(a1) + " != " + repr(a2))

    def setUp(self):
        self.classifier = LocalClassifier(window_size=1,off_board_value=-1)
        

    def test_make_features(self):
        # 000
        # 011
        # 011
        board = np.array([[0,0,0],[0,1,1],[0,1,1]])        
        self.assert_array_equal(self.classifier._make_neighbor_features_cell(board,0,0),
                                [-1,-1,-1,-1,0,0,-1,0,1])
        self.assert_array_equal(self.classifier._make_neighbor_features_cell(board,1,1),
                                [0,0,0,0,1,1,0,1,1])
        

    def test_transform_board(self):
        # 000
        # 011
        # 100
        board = np.array([[0,0,0],[0,1,1],[1,0,0]])
        
        # order of transforms is not important, just that we get all 8
        all_transforms = [
            [0,0,0,0,1,1,1,0,0], # original
            [1,0,0,0,1,0,0,1,0], # clockwise 90
            [0,0,1,1,1,0,0,0,0], # clockwise 180
            [0,1,0,0,1,0,0,0,1], # clockwise 270
            [0,0,0,1,1,0,0,0,1], # flip vertical
            [0,1,0,0,1,0,1,0,0], # flip and 90
            [1,0,0,0,1,1,0,0,0], # flip and 180
            [0,0,1,0,1,0,0,1,0]] # flip and 270

        received = [list(self.classifier._transform_board(board,i).flatten()) for i in range(8)]
        self.assertEqual(sorted(received),sorted(all_transforms))

    def test_make_training_data(self):
        # 10
        # 01
        lc = LocalClassifier(window_size=1,off_board_value=0)
        # NOTE: end board is not one conway step of start board
        examples = [KaggleExample(delta=1,start_board=[[1,1],[0,1]],end_board=[[1,0],[0,1]])]
        expected_x = np.array([ [-1,-1,-1,-1,1,0,-1,0,1],
                                [-1,-1,-1,1,0,-1,0,1,-1],
                                [-1,1,0,-1,0,1,-1,-1,-1],
                                [1,0,-1,0,1,-1,-1,-1,-1]])
        expected_y = np.array([1,1,0,1])
        
        (x,y) = self.classifier.make_training_data(examples)
        self.assert_array_equal(x,expected_x)
        self.assert_array_equal(y,expected_y)
                        
    def test_make_weighted_training_data(self):
        # 10
        # 01
        lc = LocalClassifier(window_size=1,off_board_value=0)
        # NOTE: end board is not one conway step of start board
        examples = [KaggleExample(delta=1,start_board=[[1,1],[0,1]],end_board=[[1,0],[0,1]])]
        expected_x = np.array([ [0,0,0,0,1,0,0,0,1],
                                [0,0,0,1,0,0,0,1,0],
                                [0,1,0,0,0,1,0,0,0],
                                [1,0,0,0,1,0,0,0,0]])
        expected_y = np.array([1,1,0,1])
        expected_w = np.array([1,1,1,1])

        (x,y,w) = lc.make_weighted_training_data(examples)
        # order may change ...
        # turn into sets
        expected = set([(tuple(a),b,c) for (a,b,c) in zip(expected_x,expected_y,expected_w)])
        received = set([(tuple(a),b,c) for (a,b,c) in zip(x,y,w)])
        self.assertEqual(received,expected)


    def test_predict(self):
        # classifier repeatability is doable but seems overkill right now
        # so just check that return is sensible and method doesn't error out
        examples = create_examples(num_examples=10,deltas=[1])
        self.assertRaises(ValueError,self.classifier.predict,examples[0].start_board,examples[0].delta)
        
        self.classifier.train(examples)
        
        result = self.classifier.predict(examples[0].end_board,examples[0].delta)
        # right size
        self.assertEqual(result.shape,(20,20))
        # all 0s or 1s
        self.assertTrue(((result==1) | (result==0)).all())
