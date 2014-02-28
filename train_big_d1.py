# recommend running from within an interpreter so you can do something
# with the resulting classifier
# e.g.
# exec(open("./train_big_d1.py").read())


from reverse_game_of_life import *
import gzip
import pickle
from sklearn.ensemble import RandomForestClassifier

lc_rf64_w3_d1_40k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=64,bootstrap=False,min_samples_split=256,max_features=20,n_jobs=16))
# train on 40k examples all for delta=1
ex_train = create_examples(40000,deltas=[1])
lc_rf64_w3_d1_40k.train(ex_train,use_transformations=True)
