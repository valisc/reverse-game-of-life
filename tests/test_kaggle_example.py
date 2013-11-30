import unittest
import numpy as np

from reverse_game_of_life import *

class KaggleExampleTestCase(unittest.TestCase):
    ''' Test KaggleExample class and associated functions. '''
    def assert_array_equal(self, a1, a2):        
        self.assertTrue(np.array_equal(a1,a2),repr(a1) + " != " + repr(a2))

    def test_kaggle_example_constructor(self):
        start_board = [[0,0],[1,0]]
        end_board = [[0,0],[0,0]]
        example = KaggleExample(delta=1,start_board=start_board, end_board=end_board)
        self.assertEqual(example.delta,1)
        self.assert_array_equal(example.start_board.board,start_board)
        self.assert_array_equal(example.end_board.board,end_board)
        
    def test_kaggle_example_constructor_error(self):
        self.assertRaises(ValueError, KaggleExample,(1))
        
    def test_example_evaluate(self):
        start_board = np.array([[0,1,0],[1,1,0],[0,1,1]])
        end_board = np.array([[1,1,0],[1,0,0],[1,1,1]])
        e = KaggleExample(delta=1,start_board=start_board,end_board=end_board)
        self.assertAlmostEqual(e.evaluate([[0,0,0],[0,0,0],[0,0,0]]),5.0/9)
        self.assertAlmostEqual(e.evaluate(start_board),0)
        self.assertAlmostEqual(e.evaluate(end_board),3.0/9)



    def test_create_examples(self):
        examples = create_examples(num_examples=1,deltas=[1])
        self.assertEqual(len(examples),1)
        self.assertEqual(examples[0].delta,1)
