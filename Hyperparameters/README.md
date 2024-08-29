# Definition of Hyperparameters

Definition of the set of hyperparameters in the __*parameters.txt*__ file.

---

### Deep learning (DL) methods

The hyperparameters corresponding to the DL methods, CNN and FFN, are illustrated in the image below:

![Figure](../images/Figure_FFN_Architecture_V1.png)\

#### FFN and CNN

  * __numepv:__ Number of epochs in the model. An epoch is one complete cycle through the entire training dataset by the learning algorithm. In each epoch, the model processes every sample in the dataset once. Values are positive integers.
  * __learningratev:__ Learning rate corresponding to the parameter _'lr'_ in the [torch.optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) class.
  * __selfdrv:__ Probability of an element to be zeroed in dropout. This is controlled by the parameter _'p'_ in the [torch.nn.functional.dropout](https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html) function.
  * __nunitsv:__ Number of units, depicted in green in the figure, which determine the width of the FFN. Values are positive integers.
  * __nlayersv:__ Number of layers, depicted in blue in the figure, which determine the depth of the FFN. Values are positive integers.
  * __balancev:__ Balancing of cases and controls is specified as the percentage of controls relative to the total number of cases and controls.(Ex. a value of 60 indicates that 60% of the samples are controls and 40% are cases). Values range from 0 to 100.
  * __samplingv:__ Sampling strategy. Can be defined with 4 different input strings:
    * _random:_ Random under-sampling implemented with the [imblearn.under_sampling.RandomUnderSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html) class.
    * _ENN:_ Edited Nearest Neighbors (ENN) method for under-sampling implemented with the [imblearn.under_sampling.EditedNearestNeighbours](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html) class.
    * _SMOTE_random:_ Synthetic Minority Over-sampling Technique (SMOTE) method for over-sampling implemented with the [imblearn.over_sampling.SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) class, and combined with the random under-sampling.
    * _SMOTE_ENN:_ Synthetic Minority Over-sampling Technique (SMOTE) method for over-sampling implemented with the [imblearn.over_sampling.SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) class, and combined with the ENN under-sampling.

--------------------------------------------------------
### Machine learning (ML) methods


#### LR
  * __solver:__ newton-cg,liblinear,sag,saga
  * __creg:__ 0.0001,0.001,0.01,1,10,100
  * __balance:__ Balancing of cases and controls is specified as the percentage of controls relative to the total number of cases and controls.(Ex. a value of 60 indicates that 60% of the samples are controls and 40% are cases). Values range from 0 to 100.
  * __sampl_strategy:__ Sampling strategy. Can be defined with 4 different input strings:
    * _random:_ Random under-sampling implemented with the [imblearn.under_sampling.RandomUnderSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html) class.
    * _ENN:_ Edited Nearest Neighbors (ENN) method for under-sampling implemented with the [imblearn.under_sampling.EditedNearestNeighbours](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html) class.
    * _SMOTE_random:_ Synthetic Minority Over-sampling Technique (SMOTE) method for over-sampling implemented with the [imblearn.over_sampling.SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) class, and combined with the random under-sampling.
    * _SMOTE_ENN:_ Synthetic Minority Over-sampling Technique (SMOTE) method for over-sampling implemented with the [imblearn.over_sampling.SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) class, and combined with the ENN under-sampling.

#### GB
  * __n_estimators:__ 70,80,90
  * __learning_rate:__ 0.001,0.01,0.1
  * __subsample:__ 1.0
  * __max_depth:__ 7,10,12
  * __loss:__ exponential
  * __balance:__ Balancing of cases and controls is specified as the percentage of controls relative to the total number of cases and controls.(Ex. a value of 60 indicates that 60% of the samples are controls and 40% are cases). Values range from 0 to 100.
  * __sampl_strategy:__ Sampling strategy. Can be defined with 4 different input strings:
    * _random:_ Random under-sampling implemented with the [imblearn.under_sampling.RandomUnderSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html) class.
    * _ENN:_ Edited Nearest Neighbors (ENN) method for under-sampling implemented with the [imblearn.under_sampling.EditedNearestNeighbours](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html) class.
    * _SMOTE_random:_ Synthetic Minority Over-sampling Technique (SMOTE) method for over-sampling implemented with the [imblearn.over_sampling.SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) class, and combined with the random under-sampling.
    * _SMOTE_ENN:_ Synthetic Minority Over-sampling Technique (SMOTE) method for over-sampling implemented with the [imblearn.over_sampling.SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) class, and combined with the ENN under-sampling.

#### ET and RF
  * __n_estimators:__ 50,60,70,80,100
  * __min_samples_split:__ 2,5,8
  * __min_samples_leaf:__ 1,2,5
  * __max_depth:__ None
  * __balance:__ Balancing of cases and controls is specified as the percentage of controls relative to the total number of cases and controls.(Ex. a value of 60 indicates that 60% of the samples are controls and 40% are cases). Values range from 0 to 100.
  * __sampl_strategy:__ Sampling strategy. Can be defined with 4 different input strings:
    * _random:_ Random under-sampling implemented with the [imblearn.under_sampling.RandomUnderSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html) class.
    * _ENN:_ Edited Nearest Neighbors (ENN) method for under-sampling implemented with the [imblearn.under_sampling.EditedNearestNeighbours](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html) class.
    * _SMOTE_random:_ Synthetic Minority Over-sampling Technique (SMOTE) method for over-sampling implemented with the [imblearn.over_sampling.SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) class, and combined with the random under-sampling.
    * _SMOTE_ENN:_ Synthetic Minority Over-sampling Technique (SMOTE) method for over-sampling implemented with the [imblearn.over_sampling.SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) class, and combined with the ENN under-sampling.

