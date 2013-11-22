reverse-game-of-life
====================

http://www.kaggle.com/c/conway-s-reverse-game-of-life

====================
INSTALLING
====================
We recommend installing this under a virtual environment

Install dependencies

```
$ pip install -r dependencies.txt
```

====================
Running tests
====================

```
$ nosetests
```

====================
Local Classifier
====================
LocalClassifier creates a solution based on a scikit-learn classifier predicting each cell from a window around the cell.

Train and test a local classifier

```
>>> lc = LocalClassifier()
>>> lc.train(create_examples(num_examples=100))
>>> lc.test(create_examples(num_examples=100))
```

Customize the classifier used

```
>>> from sklearn.ensembles import RandomForestClassifier
>>> lc = LocalClassifier(clf=RandomForestClassifier(n_estimators=100))
>>> lc.train(create_examples(num_examples=100))
>>> lc.test(create_examples(num_examples=100))
```

Testing with 100 random trees in the forest is rather slow right now, even slower than training. Not sure what's going on.

Get the raw X,Y data for more interactive experiments

```
>>> from sklearn.ensembles import RandomForestClassifier
>>> examples = create_examples(num_examples=100, deltas=[1]) # 100 examples all with delta=1
>>> lc = LocalClassifier(window_size=2, off_board_value=2) # options for features to use
>>> (x,y) = lc.make_training_data(examples)
>>> clf = RandomForestClassifier(n_estimators=100)
>>> clf.fit(x,y)
>>> clf.score(x,y) # get training set accuracy
>>> (x_test,y_test) = lc.make_training_data(create_examples(num_examples=100,deltas=[1])) # some test data
>>> clf.score(x_test,y_test) # test better estimate of accuracy on unseen test data
```
