
Random forests and window size of 3
===================================

With some param tuning for max_depth and max_features
|         | Training set size                 |
| # trees | 1k     | 2k     | 5k     | 10k    |
| 10      | 0.1268 | 0.1254 | 0.1240 | 0.1225 |
| 20      | 0.1255 | 0.1246 | 0.1221 |
| 50      | 

# necessary imports
>>> from reverse_game_of_life import *
>>> from sklearn.ensemble import RandomForestClassifier

# throughout examples is 100k list of kaggle examples created with
>>> examples = create_examples(100000)
# but using the same examples so testing is on exact same test data

>>> rf_params = {'max_depth':[4,8,12,16,20],'max_features':[4,8,12,16,20,24,28,32,36]}
>>> lc_rf10_w3_1k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomF
orestClassifier(n_estimators=10))
>>> examples = pickle.load(open('examples_100k_0.p','rb'))
>>> lc_rf10_w3_1k.tune_and_train(examples[0:1000],rf_params,use_transformations=
True,verbosity=1)
training data created in 8.024150848388672 seconds
training data created in 7.778686046600342 seconds
training data created in 15.897156000137329 seconds
training data created in 7.138728141784668 seconds
training data created in 7.127840995788574 seconds
training data created in 14.26283311843872 seconds
training data created in 7.419855833053589 seconds
training data created in 7.362637996673584 seconds
training data created in 14.392120599746704 seconds
training data created in 6.445016622543335 seconds
training data created in 6.598558664321899 seconds
training data created in 12.788190841674805 seconds
training data created in 7.285831928253174 seconds
training data created in 7.104708909988403 seconds
training data created in 14.066787958145142 seconds
tuning and training completed in 5427.444009542465 seconds
>>> lc_rf10_w3_1k.test(examples[50000:100000],detailed_output=True)
testing completed in 230.843820810318 seconds
delta   n       error rate
1       10054   0.09675800676347797
2       9944    0.12467467819790917
3       10116   0.1360085508105978
4       9961    0.13831367332597233
5       9925    0.13845138539042925
all     50000   0.12680610000000214
0.12680610000000214



>>> rf_params = {'max_depth':[4,8,12,16,20],'max_features':[4,8,12,16,20,24,28,32,36]}
>>> lc_rf10_w3_2k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=10))
>>> lc_rf10_w3_2k.tune_and_train(examples[0:2000],rf_params,use_transformations=True,verbosity=1)
training data created in 14.605018854141235 seconds
training data created in 15.39931058883667 seconds
training data created in 30.558761596679688 seconds
training data created in 15.662081956863403 seconds
training data created in 16.13107919692993 seconds
training data created in 32.24018478393555 seconds
training data created in 13.513206243515015 seconds
training data created in 13.701289176940918 seconds
training data created in 27.018649578094482 seconds
training data created in 13.560807943344116 seconds
training data created in 13.780156135559082 seconds
training data created in 27.029273509979248 seconds
training data created in 13.117354393005371 seconds
training data created in 13.31007194519043 seconds
training data created in 26.16680645942688 seconds
tuning and training completed in 11378.419252872467 seconds
>>> lc_rf10_w3_2k.test(examples[50000:100000],detailed_output=True)
testing completed in 236.9838252067566 seconds
delta   n       error rate
1       10054   0.09390068629401238
2       9944    0.12295705953338747
3       10116   0.13503929418742683
4       9961    0.13724851922497797
5       9925    0.13816775818639862
all     50000   0.12542535000000332
0.12542535000000332



>>> rf_params = {'max_depth':[4,8,12,16,20],'max_features':[4,8,12,16,20,24,28,32,36]}
>>> lc_rf10_w3_5k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=10))
>>> lc_rf10_w3_5k.tune_and_train(examples[0:5000],rf_params,use_transformations=True,verbosity=1)
training data created in 36.629271507263184 seconds
training data created in 36.3157844543457 seconds
training data created in 73.81269526481628 seconds
training data created in 36.307194232940674 seconds
training data created in 36.85136580467224 seconds
training data created in 73.68338465690613 seconds
training data created in 35.18715476989746 seconds
training data created in 34.531614542007446 seconds
training data created in 69.85783123970032 seconds
training data created in 34.45726227760315 seconds
training data created in 34.802769899368286 seconds
training data created in 69.75554871559143 seconds
training data created in 34.57942867279053 seconds
training data created in 34.50123429298401 seconds
training data created in 68.65994763374329 seconds
tuning and training completed in 34136.422243118286 seconds
>>> lc_rf10_w3_5k.test(examples[50000:100000],detailed_output=True)
testing completed in 242.98559856414795 seconds
delta   n       error rate
1       10054   0.09096677939128761
2       9944    0.12101065969428869
3       10116   0.13433447014630398
4       9961    0.1359348960947706
5       9925    0.13787909319899266
all     50000   0.12398670000000347
0.12398670000000347


>>> rf_params = {'max_depth':[4,8,12,16,20],'max_features':[4,8,12,16,20,24,28,32,36]}
>>>lc_rf20_w3_1k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=20))
>>> lc_rf20_w3_1k.tune_and_train(examples[0:1000],rf_params,use_transformations=True,verbosity=1)
training data created in 7.942469835281372 seconds
training data created in 7.434887647628784 seconds
training data created in 15.882714986801147 seconds
training data created in 6.8850789070129395 seconds
training data created in 7.038900852203369 seconds
training data created in 13.954154014587402 seconds
training data created in 7.042872667312622 seconds
training data created in 7.085495710372925 seconds
training data created in 14.06035041809082 seconds
training data created in 6.0669097900390625 seconds
training data created in 6.305178880691528 seconds
training data created in 12.588786840438843 seconds
training data created in 6.824497699737549 seconds
training data created in 6.942764043807983 seconds
training data created in 13.817763090133667 seconds
tuning and training completed in 10351.518267393112 seconds
>>> lc_rf20_w3_1k.test(examples[50000:100000],detailed_output=True)
testing completed in 288.84096813201904 seconds
delta   n       error rate
1       10054   0.09416401432265824
2       9944    0.12317553298471526
3       10116   0.13527283511269247
4       9961    0.13701761871298138
5       9925    0.13795591939546656
all     50000   0.1254809500000032
0.1254809500000032

>>> rf_params = {'max_depth':[4,8,12,16,20],'max_features':[4,8,12,16,20,24,28,32,36]}
>>> lc_rf20_w3_2k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomF
orestClassifier(n_estimators=20))
>>> lc_rf20_w3_2k.tune_and_train(examples[0:2000],rf_params,use_transformations=
True,verbosity=1)
training data created in 15.057081460952759 seconds
training data created in 14.827362060546875 seconds
training data created in 30.206374406814575 seconds
training data created in 15.54614782333374 seconds
training data created in 15.703454732894897 seconds
training data created in 31.57854914665222 seconds
training data created in 13.126159191131592 seconds
training data created in 13.264602661132812 seconds
training data created in 25.801635265350342 seconds
training data created in 13.170904636383057 seconds
training data created in 13.19525146484375 seconds
training data created in 26.861876726150513 seconds
training data created in 13.046380519866943 seconds
training data created in 12.869215965270996 seconds
training data created in 25.78421115875244 seconds
tuning and training completed in 21754.30801010132 seconds
>>> lc_rf20_w3_2k.test(examples[50000:100000],detailed_output=True)             
              testing completed in 304.08431100845337 seconds
delta   n       error rate
1       10054   0.09228839267953125
2       9944    0.12189134151247073
3       10116   0.1343354586793211
4       9961    0.1364689790181713
5       9925    0.13807858942065554
all     50000   0.12457380000000168
0.12457380000000168




>>> rf_params = {'max_depth':[4,8,12,16,20],'max_features':[4,8,12,16,20,24,28,32,36]}
>>> lc_rf20_w3_5k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=20))
>>> lc_rf20_w3_5k.tune_and_train(examples[0:5000],rf_params,use_transformations=True,verbosity=1)
training data created in 36.54121971130371 seconds
training data created in 36.03036117553711 seconds
training data created in 72.99042558670044 seconds
training data created in 36.19559073448181 seconds
training data created in 35.9888072013855 seconds
training data created in 71.06846714019775 seconds
training data created in 33.889548778533936 seconds
training data created in 33.67138051986694 seconds
training data created in 68.12282133102417 seconds
training data created in 33.5337553024292 seconds
training data created in 33.814897775650024 seconds
training data created in 67.778249502182 seconds
training data created in 33.38223600387573 seconds
training data created in 33.32728314399719 seconds
training data created in 67.8635516166687 seconds
tuning and training completed in 64481.52228832245 seconds
>>> lc_rf20_w3_5k.test(examples[50000:100000],detailed_output=True)             testing completed in 325.90638542175293 seconds
delta   n       error rate
1       10054   0.0877852098667203
2       9944    0.1190162409493168
3       10116   0.13195259984183585
4       9961    0.13505621925509595
5       9925    0.13686297229219252
all     50000   0.12209165000000305
0.12209165000000305


>>> rf_params = {'max_depth':[8,12,16,20],'max_features':[8,12,16,20,24,28,32,36]}
>>> lc_rf10_w3_10k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=10))
>>> lc_rf10_w3_10k.tune_and_train(examples[0:10000],rf_params,use_transformations=True,verbosity=1)
training data created in 49.759636640548706 seconds
training data created in 49.20284295082092 seconds
training data created in 106.65880489349365 seconds
training data created in 51.60172462463379 seconds
training data created in 50.533955335617065 seconds
training data created in 98.59237027168274 seconds
training data created in 48.257474422454834 seconds
training data created in 47.83779788017273 seconds
training data created in 104.72760200500488 seconds
training data created in 50.232829332351685 seconds
training data created in 50.99352240562439 seconds
training data created in 101.9321756362915 seconds
training data created in 50.333924770355225 seconds
training data created in 51.15891170501709 seconds
training data created in 141.00279211997986 seconds
tuning and training completed in 73638.14185190201 seconds
>>> lc_rf10_w3_10k.test(examples[50000:100000],verbosity=2)
testing completed in 137.7 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0885       0.0736   (0.0870,0.0899)
2       9944    0.1190       0.0934   (0.1171,0.1208)
3       10116   0.1323       0.0982   (0.1304,0.1342)
4       9961    0.1355       0.0982   (0.1336,0.1375)
5       9925    0.1376       0.0977   (0.1357,0.1395)
all     50000   0.1225       0.0945   (0.1217,0.1234)
0.12253265000000158
