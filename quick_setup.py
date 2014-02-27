from reverse_game_of_life import *
from sklearn.ensemble import RandomForestClassifier
from sys import argv


example_size = int(argv[1])
examples = create_examples(example_size)
rf_params = {'max_depth':[4,8,12,16,20],'max_features':[4,8,12,16,20,24,28,32,36]}

lc_rf10_w3_1k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=10,n_jobs=-1))
lc_rf10_w3_1k.tune_and_train(examples[0:int(example_size/100)],rf_params,use_transformations=True,verbosity=1)
lc_rf10_w3_1k.test(examples[int(example_size/2):],detailed_output=True)
