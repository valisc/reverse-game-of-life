
Random forests and window size of 3
===================================

v1 - param tuning for max_depth and max_features
v2 - max_depth=None and min_samples_split=64 and param tuning for max_features

| All     | Training set size                 |
| # trees | 1k     | 2k     | 5k     | 10k    | 20k 
| 0 (dead)| 0.1439 |
| 10      | 0.1268 | 0.1254 | 0.1240 | 0.1225 | 0.1212 |
| 20      | 0.1255 | 0.1246 | 0.1221 | 0.1210 |
| 50      | 0.1240 | 0.1225 | 0.1206 | 
| 10 v2   | 0.1234 | 0.1220 | 0.1202 | 0.1190 |
| 20 v2   | 0.1226 | 0.1213 | 0.1195 | 0.1184 |


| delta=1 | Training set sizen                 |
| # trees | 1k     | 2k     | 5k     | 10k    | 20k    |
| 0(dead) | 0.1443 |n
| 10      | 0.0968 | 0.0939 | 0.0910 | 0.0885 | 0.0858 |
| 20      | 0.0942 | 0.0923 | 0.0878 | 0.0860 |
| 50      | 0.0912 | 0.0886 | 0.0858 |  
| 10 v2   | 0.0925 | 0.0895 | 0.0862 | 0.0844 |
| 20 v2   | 0.0911 | 0.0883 | 0.0848 | 0.0830 |

| delta=2
| # trees | 1k     | 2k     | 5k     | 10k    | 20k    |
| 0(dead) | 0.1429 |
| 10      | 0.1247 | 0.1230 | 0.1210 | 0.1190 | 0.1180 |
| 20      | 0.1232 | 0.1219 | 0.1190 | 0.1179 |
| 50      | 0.1213 | 0.1194 | 0.1173 |
| 10 v2   | 0.1208 | 0.1190 | 0.1172 | 0.1152 |
| 20 v2   | 0.1201 | 0.1180 | 0.1161 | 0.1148 |

| delta=3
| # trees | 1k     | 2k     | 5k     | 10k    | 20k    |
| 0(dead) | 0.1449 |
| 10      | 0.1360 | 0.1350 | 0.1343 | 0.1323 | 0.1313 |
| 20      | 0.1353 | 0.1343 | 0.1320 | 0.1312 |
| 50      | 0.1334 | 0.1325 | 0.1307 |
| 10 v2   | 0.1323 | 0.1317 | 0.1300 | 0.1290 |
| 20 v2   | 0.1317 | 0.1309 | 0.1293 | 0.1284 |

| delta=4 
| # trees | 1k     | 2k     | 5k     | 10k    | 20k    |
| 0(dead) | 0.1439 |
| 10      | 0.1383 | 0.1372 | 0.1359 | 0.1355 | 0.1344 |
| 20      | 0.1370 | 0.1365 | 0.1351 | 0.1341 |
| 50      | 0.1367 | 0.1352 | 0.1339 |
| 10 v2   | 0.1356 | 0.1348 | 0.1331 | 0.1326 |
| 20 v2   | 0.1350 | 0.1345 | 0.1328 | 0.1321 |
 
| delta=5
| # trees | 1k     | 2k     | 5k     | 10k    | 20k    |
| 0(dead) | 0.1436 |
| 10      | 0.1385 | 0.1382 | 0.1379 | 0.1376 | 0.1366 |
| 20      | 0.1380 | 0.1381 | 0.1369 | 0.1362 |
| 50      | 0.1377 | 0.1369 | 0.1357 |
| 10 v2   | 0.1359 | 0.1353 | 0.1348 | 0.1342 |
| 20 v2   | 0.1355 | 0.1351 | 0.1345 | 0.1339 |


# necessary imports
>>> from reverse_game_of_life import *
>>> from sklearn.ensemble import RandomForestClassifier

# throughout examples is 100k list of kaggle examples created with
>>> examples = create_examples(100000)

# but using the same examples (examples_100k_0.p on spider) so testing
# is on exact same test data
>>> Classifier().test(examples[50000:100000],detailed_output=True,verbosity=2)
testing completed in 18.5 seconds
delta   n       error rate   sd       95% CI
1       10054   0.1443       0.0942   (0.1424,0.1461)
2       9944    0.1429       0.0945   (0.1410,0.1447)
3       10116   0.1449       0.0938   (0.1431,0.1467)
4       9961    0.1439       0.0936   (0.1420,0.1457)
5       9925    0.1436       0.0943   (0.1418,0.1455)
all     50000   0.1439       0.0941   (0.1431,0.1447)
0.1439094500000118



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


>>> lc_rf50_w3_1k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=50))
>>> lc_rf50_w3_1k.tune_and_train(examples[0:1000],rf_params,use_transformations=True,verbosity=1)
training data created in 6.395838975906372 seconds
training data created in 6.385439395904541 seconds
training data created in 12.695619583129883 seconds
training data created in 5.575204610824585 seconds
training data created in 5.68770694732666 seconds
training data created in 11.259097814559937 seconds
training data created in 5.826438665390015 seconds
training data created in 5.890855312347412 seconds
training data created in 11.44253396987915 seconds
training data created in 5.1117103099823 seconds
training data created in 5.244232654571533 seconds
training data created in 10.17555856704712 seconds
training data created in 5.759449243545532 seconds
training data created in 5.6962034702301025 seconds
training data created in 11.329783916473389 seconds
tuning and training completed in 29797.53557395935 seconds
>>> lc_rf50_w3_1k.test(examples[50000:100000],verbosity=2)
testing completed in 604.9 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0912       0.0742   (0.0898,0.0927)
2       9944    0.1213       0.0945   (0.1194,0.1231)
3       10116   0.1334       0.0973   (0.1315,0.1353)
4       9961    0.1367       0.0974   (0.1347,0.1386)
5       9925    0.1377       0.0978   (0.1358,0.1397)
all     50000   0.1240       0.0943   (0.1232,0.1248)
0.12401195000000285


>>> lc_rf50_w3_2k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=50))
>>> lc_rf50_w3_2k.tune_and_train(examples[0:2000],rf_params,use_transformations=True,verbosity=1)
training data created in 11.938565254211426 seconds
training data created in 12.26734209060669 seconds
training data created in 23.84203553199768 seconds
training data created in 12.890844821929932 seconds
training data created in 12.83693265914917 seconds
training data created in 60.965598583221436 seconds
training data created in 10.666390180587769 seconds
training data created in 11.129401922225952 seconds
training data created in 20.256810188293457 seconds
training data created in 10.532788515090942 seconds
training data created in 10.30793809890747 seconds
training data created in 21.679993867874146 seconds
training data created in 31.824877500534058 seconds
training data created in 10.51926589012146 seconds
training data created in 20.769895315170288 seconds
tuning and training completed in 64604.39176058769 seconds
>>> lc_rf50_w3_2k.test(examples[50000:100000],verbosity=2)
testing completed in 546.1 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0886       0.0728   (0.0872,0.0900)
2       9944    0.1194       0.0923   (0.1176,0.1212)
3       10116   0.1325       0.0973   (0.1306,0.1344)
4       9961    0.1352       0.0984   (0.1333,0.1371)
5       9925    0.1369       0.0985   (0.1350,0.1388)
all     50000   0.1225       0.0941   (0.1217,0.1233)
0.12249245000000399

>>> rf_params = {'max_depth':[8,12,16,20],'max_features':[8,12,16,20,24,28,32,36]}
>>> lc_rf20_w3_10k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=20))
>>> lc_rf20_w3_10k.tune_and_train(examples[0:10000],rf_params,use_transformations=True,verbosity=1)
training data created in 50.56372332572937 seconds
training data created in 49.85594177246094 seconds
training data created in 100.78403091430664 seconds
training data created in 51.22110891342163 seconds
training data created in 48.54091286659241 seconds
training data created in 116.23162865638733 seconds
training data created in 51.77469086647034 seconds
training data created in 54.66222286224365 seconds
training data created in 113.24372982978821 seconds
training data created in 50.06851315498352 seconds
training data created in 50.87463068962097 seconds
training data created in 107.09537482261658 seconds
training data created in 47.9894073009491 seconds
training data created in 48.87774443626404 seconds
training data created in 208.3990604877472 seconds
tuning and training completed in 144486.12302398682 seconds
>>> lc_rf20_w3_10k.test(examples[50000:100000],verbosity=2)
testing completed in 238.5 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0860       0.0715   (0.0846,0.0874)
2       9944    0.1179       0.0927   (0.1161,0.1197)
3       10116   0.1312       0.0972   (0.1293,0.1330)
4       9961    0.1341       0.0988   (0.1322,0.1361)
5       9925    0.1362       0.0991   (0.1342,0.1381)
all     50000   0.1210       0.0943   (0.1202,0.1218)
0.12101670000000125


>>> lc_rf50_w3_5k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=50))
>>> rf_params = {'max_depth':[8,12,16,20],'max_features':[8,12,16,20,24,28,32,36]}
>>> lc_rf50_w3_5k.tune_and_train(examples[0:5000],rf_params,use_transformations=True,verbosity=1)
training data created in 28.53341555595398 seconds
^[[1~training data created in 27.646077871322632 seconds
training data created in 272.8837249279022 seconds
training data created in 28.500855445861816 seconds
training data created in 28.56894040107727 seconds
training data created in 329.42533111572266 seconds
training data created in 27.203734397888184 seconds
training data created in 26.73308491706848 seconds
training data created in 88.07786512374878 seconds
training data created in 25.790913343429565 seconds
training data created in 24.916589975357056 seconds
training data created in 756.3030087947845 seconds
training data created in 40.65700888633728 seconds
training data created in 26.35893702507019 seconds
training data created in 56.99277186393738 seconds
tuning and training completed in 229469.81395864487 seconds
>>> lc_rf50_w3_5k.test(examples[50000:100000],detailed_output=True)
delta   n       error rate   sd       95% CI
1       10054   0.0858       0.0711   (0.0844,0.0872)
2       9944    0.1173       0.0930   (0.1155,0.1192)
3       10116   0.1307       0.0986   (0.1287,0.1326)
4       9961    0.1339       0.0990   (0.1319,0.1358)
5       9925    0.1357       0.0999   (0.1338,0.1377)
all     50000   0.1206       0.0948   (0.1198,0.1215)
0.12063600000000377



>>> rf_params = {'max_depth':[12,16,20],'max_features':[8,12,16,20,24,28,32,36]}
>>> lc_rf10_w3_20k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=10))
>>> lc_rf10_w3_20k.tune_and_train(examples[0:20000],rf_params,use_transformations=True)
training data created in 102.13056993484497 seconds
training data created in 100.57641696929932 seconds
training data created in 203.28278613090515 seconds
training data created in 95.4451813697815 seconds
training data created in 95.11247038841248 seconds
training data created in 193.48292016983032 seconds
training data created in 93.84220147132874 seconds
training data created in 94.36005306243896 seconds
training data created in 284.34112191200256 seconds
training data created in 96.8078966140747 seconds
training data created in 98.22176742553711 seconds
training data created in 189.39057302474976 seconds
training data created in 95.31740832328796 seconds
training data created in 95.8582022190094 seconds
training data created in 193.12487864494324 seconds
>>> lc_rf10_w3_20k.test(examples[50000:100000],detailed_output=True,verbosity=2)
testing completed in 125.1 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0858       0.0715   (0.0844,0.0872)
2       9944    0.1180       0.0927   (0.1162,0.1199)
3       10116   0.1313       0.0974   (0.1294,0.1332)
4       9961    0.1344       0.0983   (0.1325,0.1364)
5       9925    0.1366       0.0987   (0.1347,0.1386)
all     50000   0.1212       0.0942   (0.1204,0.1220)
0.12118495000000197


>>> lc_rf10_w3_v2_1k = LocalClassifier(window_size=3,off_board_value=-1, clf=RandomForestClassifier(n_estimators=10,max_depth=None,bootstrap=False,min_samples_split=64))
>>> rf_v2_params = {'max_features':[4,8,12,16,20,24,28,32,36,40]}
>>> lc_rf10_w3_v2_1k.tune_and_train(examples[0:1000],rf_v2_params,use_transformations=True,verbosity=1)
training data created in 5.587444543838501 seconds
training data created in 5.614023447036743 seconds
training data created in 11.943026304244995 seconds
training data created in 5.122316360473633 seconds
training data created in 5.193070888519287 seconds
training data created in 10.441855430603027 seconds
training data created in 5.377492427825928 seconds
training data created in 5.321816444396973 seconds
training data created in 9.988096237182617 seconds
training data created in 4.428252458572388 seconds
training data created in 4.6205973625183105 seconds
training data created in 8.898418664932251 seconds
training data created in 5.053190231323242 seconds
training data created in 5.091525316238403 seconds
training data created in 9.884079217910767 seconds
tuning and training completed in 3596.829959154129 seconds
>>> lc_rf10_w3_v2_1k.test(examples[50000:100000],verbosity=2)
testing completed in 151.1 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0925       0.0749   (0.0911,0.0940)
2       9944    0.1208       0.0941   (0.1189,0.1226)
3       10116   0.1323       0.0980   (0.1304,0.1342)
4       9961    0.1356       0.0980   (0.1337,0.1375)
5       9925    0.1359       0.0999   (0.1339,0.1378)
all     50000   0.1234       0.0949   (0.1225,0.1242)
0.12337145000000276


>>> lc_rf10_w3_v2_2k = LocalClassifier(window_size=3,off_board_value=-1, clf=RandomForestClassifier(n_estimators=10,max_depth=None,bootstrap=False,min_samples_split=64))
>>> lc_rf10_w3_v2_2k.tune_and_train(examples[0:2000],rf_v2_params,use_transformations=True,verbosity=1)                                       training data created in 10.583369493484497 seconds
training data created in 10.432039976119995 seconds
training data created in 21.1219265460968 seconds
training data created in 11.028483867645264 seconds
training data created in 11.176833629608154 seconds
training data created in 22.441216945648193 seconds
training data created in 9.180810928344727 seconds
training data created in 9.58297848701477 seconds
training data created in 18.708654403686523 seconds
training data created in 9.62415599822998 seconds
training data created in 9.569076299667358 seconds
training data created in 18.94559621810913 seconds
training data created in 8.91556191444397 seconds
training data created in 9.54601240158081 seconds
training data created in 18.089386701583862 seconds
tuning and training completed in 7306.3845019340515 seconds
>>> lc_rf10_w3_v2_2k.test(examples[50000:100000],verbosity=2)
testing completed in 167.9 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0895       0.0732   (0.0881,0.0910)
2       9944    0.1190       0.0934   (0.1171,0.1208)
3       10116   0.1317       0.0980   (0.1298,0.1336)
4       9961    0.1348       0.0986   (0.1328,0.1367)
5       9925    0.1353       0.0999   (0.1333,0.1373)
all     50000   0.1220       0.0947   (0.1212,0.1228)
0.12201130000000251


>>> lc_rf10_w3_v2_5k = LocalClassifier(window_size=3,off_board_value=-1, clf=RandomForestClassifier(n_estimators=10,max_depth=None,bootstrap=False,min_samples_split=64))
>>> lc_rf10_w3_v2_5k.tune_and_train(examples[0:5000],rf_v2_params,use_transformations=True,verbosity=1)                                                         training data created in 25.737348079681396 seconds
training data created in 26.069751977920532 seconds
training data created in 51.76441216468811 seconds
training data created in 25.11500120162964 seconds
training data created in 25.541069269180298 seconds
training data created in 53.185325145721436 seconds
training data created in 24.422606468200684 seconds
training data created in 24.582801818847656 seconds
training data created in 51.424943685531616 seconds
training data created in 24.278393030166626 seconds
training data created in 24.262994527816772 seconds
training data created in 50.605427980422974 seconds
training data created in 24.413602113723755 seconds
training data created in 24.152410745620728 seconds
training data created in 49.01412510871887 seconds
tuning and training completed in 23505.870305776596 seconds
>>> lc_rf10_w3_v2_5k.test(examples[50000:100000],verbosity=2)                   testing completed in 180.6 seconds                                              delta   n       error rate   sd       95% CI
1       10054   0.0862       0.0715   (0.0848,0.0876)
2       9944    0.1172       0.0929   (0.1153,0.1190)
3       10116   0.1300       0.0982   (0.1281,0.1319)
4       9961    0.1331       0.0991   (0.1312,0.1351)
5       9925    0.1348       0.1005   (0.1329,0.1368)
all     50000   0.1202       0.0948   (0.1194,0.1211)
0.1202295500000023



>>> lc_rf20_w3_v2_1k = LocalClassifier(window_size=3,off_board_value=-1, clf=RandomForestClassifier(n_estimators=20,max_depth=None,bootstrap=False,min_samples_split=64))
>>> lc_rf20_w3_v2_1k.tune_and_train(examples[0:1000],rf_v2_params,use_transformations=True,verbosity=1)
training data created in 5.7874860763549805 seconds
training data created in 5.833743333816528 seconds
training data created in 11.65848994255066 seconds
training data created in 5.216526031494141 seconds
training data created in 5.348188877105713 seconds
training data created in 9.774640560150146 seconds
training data created in 5.018142461776733 seconds
training data created in 5.108093738555908 seconds
training data created in 9.827835083007812 seconds
training data created in 4.3721349239349365 seconds
training data created in 4.461880922317505 seconds
training data created in 8.690989255905151 seconds
training data created in 5.080218315124512 seconds
training data created in 4.914518117904663 seconds
training data created in 9.722307682037354 seconds
tuning and training completed in 7031.467175722122 seconds
>>> lc_rf20_w3_v2_1k.test(examples[50000:100000],verbosity=2)
testing completed in 261.8 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0911       0.0739   (0.0897,0.0926)
2       9944    0.1201       0.0936   (0.1183,0.1219)
3       10116   0.1317       0.0975   (0.1298,0.1336)
4       9961    0.1350       0.0983   (0.1330,0.1369)
5       9925    0.1355       0.0998   (0.1336,0.1375)
all     50000   0.1226       0.0946   (0.1218,0.1235)
0.1226371500000035


>>> lc_rf20_w3_v2_2k = LocalClassifier(window_size=3,off_board_value=-1, clf=RandomForestClassifier(n_estimators=20,max_depth=None,bootstrap=False,min_samples_split=64))
>>> lc_rf20_w3_v2_2k.tune_and_train(examples[0:2000],rf_v2_params,use_transformations=True,verbosity=1)                                     training data created in 10.573201417922974 seconds
training data created in 10.77562689781189 seconds
training data created in 21.166105031967163 seconds
training data created in 11.200411081314087 seconds
training data created in 11.31595253944397 seconds
training data created in 22.480254650115967 seconds
training data created in 9.413429975509644 seconds
training data created in 9.163946628570557 seconds
training data created in 18.245206832885742 seconds
training data created in 9.222099542617798 seconds
training data created in 9.423505544662476 seconds
training data created in 18.384167194366455 seconds
training data created in 8.920380353927612 seconds
training data created in 9.093026399612427 seconds
training data created in 17.74782705307007 seconds
tuning and training completed in 14237.202942371368 seconds
>>> lc_rf20_w3_v2_2k.test(examples[50000:100000],verbosity=2)
testing completed in 278.2 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0883       0.0724   (0.0869,0.0898)
2       9944    0.1180       0.0927   (0.1162,0.1198)
3       10116   0.1309       0.0975   (0.1290,0.1328)
4       9961    0.1345       0.0983   (0.1326,0.1364)
5       9925    0.1351       0.1000   (0.1331,0.1370)
all     50000   0.1213       0.0944   (0.1205,0.1221)
0.1213195000000015


>>> lc_rf20_w3_v2_5k = LocalClassifier(window_size=3,off_board_value=-1, clf=RandomForestClassifier(n_estimators=20,max_depth=None,bootstrap=False,min_samples_split=64))
>>> lc_rf20_w3_v2_5k.tune_and_train(examples[0:5000],rf_v2_params,use_transformations=True,verbosity=1)
training data created in 25.82967472076416 seconds
training data created in 25.354238748550415 seconds
training data created in 52.083269357681274 seconds
training data created in 25.592249155044556 seconds
training data created in 25.368857383728027 seconds
training data created in 51.21273851394653 seconds
training data created in 23.92716407775879 seconds
training data created in 23.526536226272583 seconds
training data created in 47.84968972206116 seconds
training data created in 23.62741446495056 seconds
training data created in 23.686455488204956 seconds
training data created in 48.14885663986206 seconds
training data created in 23.44158697128296 seconds
training data created in 23.380560636520386 seconds
training data created in 47.544256925582886 seconds
tuning and training completed in 50827.602601766586 seconds
>>> lc_rf20_w3_v2_5k.test(examples[50000:100000],verbosity=2)
testing completed in 308.3 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0848       0.0705   (0.0835,0.0862)
2       9944    0.1161       0.0923   (0.1143,0.1179)
3       10116   0.1293       0.0977   (0.1274,0.1312)
4       9961    0.1328       0.0988   (0.1309,0.1347)
5       9925    0.1345       0.1003   (0.1325,0.1365)
all     50000   0.1195       0.0944   (0.1186,0.1203)
0.11946670000000262


>>> rf_v2_params =  {'max_features':[4,8,12,16,20,24,28,32,36,40]}
>>> lc_rf10_w3_v2_10k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=10,max_depth=None,bootstrap=False,min_samples_split=64))
>>> lc_rf10_w3_v2_10k.tune_and_train(examples[0:10000],rf_v2_params,use_transformations=True,verbosity=1)
training data created in 49.557963848114014 seconds
training data created in 49.99402713775635 seconds
training data created in 108.36565709114075 seconds
training data created in 51.9759578704834 seconds
training data created in 51.52695417404175 seconds
training data created in 104.87696766853333 seconds
training data created in 51.99434542655945 seconds
training data created in 51.70567321777344 seconds
training data created in 98.24357295036316 seconds
training data created in 47.4780969619751 seconds
training data created in 48.11114501953125 seconds
training data created in 96.66214656829834 seconds
training data created in 47.570565938949585 seconds
training data created in 48.58940052986145 seconds
training data created in 95.8906717300415 seconds
tuning and training completed in 53647.33166694641 seconds
>>> lc_rf10_w3_v2_10k.test(examples[50000:100000],verbosity=2)
testing completed in 182.8 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0839       0.0700   (0.0825,0.0853)
2       9944    0.1154       0.0923   (0.1136,0.1172)
3       10116   0.1291       0.0979   (0.1272,0.1310)
4       9961    0.1327       0.0990   (0.1308,0.1347)
5       9925    0.1342       0.1004   (0.1322,0.1362)
all     50000   0.1190       0.0945   (0.1182,0.1198)
0.11902095000000229
>>> [(c,lc_rf10_w3_v2_10k.classifiers[c].max_features) for c in lc_rf10_w3_v2_10k.classifiers]
[(1, 20), (2, 16), (3, 12), (4, 8), (5, 8)]
>>> lc_rf10_w3_v2_10k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=10,max_depth=None,bootstrap=False,min_samples_split=64))
>>> lc_rf10_w3_v2_10k.tune_and_train(examples[0:10000],rf_v2_params,use_transformations=True,verbosity=1)
training data created in 51.269888401031494 seconds
training data created in 49.80906081199646 seconds
training data created in 99.93267631530762 seconds
training data created in 48.53078317642212 seconds
training data created in 48.96663975715637 seconds
training data created in 98.0561671257019 seconds
training data created in 48.15947508811951 seconds
training data created in 48.518390417099 seconds
training data created in 94.444509267807 seconds
training data created in 45.929556131362915 seconds
training data created in 46.53597593307495 seconds
training data created in 93.52166032791138 seconds
training data created in 46.14307880401611 seconds
training data created in 46.37464118003845 seconds
training data created in 93.6843204498291 seconds
tuning and training completed in 54645.2928276062 seconds
>>> lc_rf10_w3_v2_10k.test(examples[50000:100000],verbosity=2)
testing completed in 184.6 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0844       0.0705   (0.0830,0.0858)
2       9944    0.1152       0.0922   (0.1134,0.1170)
3       10116   0.1290       0.0981   (0.1271,0.1309)
4       9961    0.1326       0.0992   (0.1306,0.1345)
5       9925    0.1342       0.1004   (0.1322,0.1362)
all     50000   0.1190       0.0946   (0.1182,0.1199)
0.11902275000000362
>>> [(c,lc_rf10_w3_v2_10k.classifiers[c].max_features) for c in lc_rf10_w3_v2_10k.classifiers]
[(1, 36), (2, 16), (3, 16), (4, 12), (5, 8)]



>>> rf_v2_params =  {'max_features':[4,8,12,16,20,24,28,32,36,40]}
>>> lc_rf20_w3_v2_10k = LocalClassifier(window_size=3,off_board_value=-1,clf=RandomForestClassifier(n_estimators=20,max_depth=None,bootstrap=False,min_samples_split=64))
>>> lc_rf20_w3_v2_10k.tune_and_train(examples[0:10000],rf_v2_params,use_transformations=True,verbosity=1)
training data created in 54.63200902938843 seconds
training data created in 54.41248559951782 seconds
training data created in 111.36985993385315 seconds
training data created in 50.70378851890564 seconds
training data created in 49.71528196334839 seconds
training data created in 101.83740568161011 seconds
training data created in 51.04755449295044 seconds
training data created in 50.89416575431824 seconds
training data created in 98.45438170433044 seconds
training data created in 47.79933214187622 seconds
training data created in 47.804333209991455 seconds
training data created in 97.47762393951416 seconds
training data created in 50.01238489151001 seconds
training data created in 48.46520400047302 seconds
training data created in 98.91297936439514 seconds
tuning and training completed in 107720.34962129593 seconds
>>> lc_rf20_w3_v2_10k.test(examples[50000:100000],verbosity=2)
testing completed in 315.4 seconds
delta   n       error rate   sd       95% CI
1       10054   0.0830       0.0694   (0.0816,0.0843)
2       9944    0.1148       0.0918   (0.1130,0.1166)
3       10116   0.1284       0.0976   (0.1265,0.1303)
4       9961    0.1321       0.0989   (0.1302,0.1340)
5       9925    0.1339       0.1003   (0.1319,0.1359)
all     50000   0.1184       0.0942   (0.1176,0.1192)
0.11838885000000278
>>> pickle.dump(lc_rf20_w3_v2_10k,gzip.open('lc_rf20_w3_v2_10k.p.gz','wb'))
>>> [(c,lc_rf20_w3_v2_10k.classifiers[c].max_features) for c in lc_rf20_w3_v2_10k.classifiers]
[(1, 28), (2, 12), (3, 16), (4, 16), (5, 12)]
