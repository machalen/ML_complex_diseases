# Definition of Hyperparameters

Definition of the set of parameters in the __*parameters.txt*__ file.

### CNN

  * numEpV:200,300,500
  * learningRateV:0.0001,0.001,0.01
  * selfdrV:0.1,0.2,0.4
  * nUnitsV:100,200
  * nLayersV:1,2,3
  * BalanceV:50,70
  * SamplingV:random,ENN,SMOTE_random,SMOTE_ENN

### FFN
  * numEpV:200,300,500
  * learningRateV:0.0001,0.001,0.01
  * selfdrV:0.1,0.2,0.4
  * nUnitsV:100,200
  * nLayersV:1,2,3
  * BalanceV:50,70
  * SamplingV:random,ENN,SMOTE_random,SMOTE_ENN

### GB
  * n_estimators:70,80,90
  * learning_rate:0.001,0.01,0.1
  * subsample:1.0
  * max_depth:7,10,12
  * loss:exponential
  * balance:50,70
  * sampl_strategy:random,ENN,SMOTE_random,SMOTE_ENN

### ET
  * n_estimators:50,60,70,80,100
  * min_samples_split:2,5,8
  * min_samples_leaf:1,2,5
  * max_depth:None
  * balance:50,60,70
  * sampl_strategy:random,ENN,SMOTE_random,SMOTE_ENN

### LR
  * solver:newton-cg,liblinear,sag,saga
  * creg:0.0001,0.001,0.01,1,10,100
  * balance:50,60,70
  * sampl_strategy:random,ENN,SMOTE_random,SMOTE_ENN

### RF
  * n_estimators:50,60,70,80,100
  * min_samples_split:2,5,8
  * min_samples_leaf:1,2,5
  * max_depth:None
  * balance:50,60,70
  * sampl_strategy:random,ENN,SMOTE_random,SMOTE_ENN
