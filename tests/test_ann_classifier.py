import unittest

from reverse_game_of_life import *

class ANNClassifierTestCase(unittest.TestCase):
    ''' Test ANNClassifier. '''

    def setUp(self):
        self.classifier = ANNClassifier()
    
    def test_train(self):
        # just make sure it works with no errors
        self.classifier.train(create_examples(num_examples=10,deltas=[1]),max_epochs=1)

    def test_predict(self):
        examples = create_examples(num_examples=1,deltas=[1])
                
        result = self.classifier.predict(examples[0].end_board,examples[0].delta)
        
        # right size
        self.assertEqual(result.shape,(20,20))
        # all 0s or 1s
        self.assertTrue(((result==1) | (result==0)).all())
