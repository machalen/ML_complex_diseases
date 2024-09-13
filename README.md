# Machine learning methods applied to classify complex diseases using genomic data

Python scripts for running the machine learning (ML) and deep learning (DL) models reported in the manuscript:
Magdalena Arnal Segura, 2024
https://doi.org/10.1101/2024.03.18.585541

This work is part of the PhD thesis titled [*"Machine Learning Methods Applied to Classify Complex Diseases Using Genomic Data"*](https://iris.uniroma1.it/handle/11573/1706863) defended by Magdalena Arnal Segura in March 2024 at Sapienza Università di Roma.

### Author/Support

https://github.com/machalen/ML_complex_diseases/issues </br>

### Directory Contents

  * __Hyperparameters:__ Definitions of the parameters used in the hyperparameter selection process.
  * __Python_scripts:__ Python scripts divided into different subdirectories corresponding to the different ML and DL methods.
  * __requirements.txt:__ List of required packages for installation.

### Running The Code

All the Python scripts are located in the _'Python_scripts'_ folder and are organized into subdirectories named according to the ML and DL methods used:

  * [Logistic Regression (LR)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  * [Gradient-Boosted Decision Trees (GB)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
  * [Random Forest (RF)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  * [Extremely Randomized Trees (ET)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
  * [Feedforward neural network (FFN)](https://pytorch.org/docs/stable/index.html)
  * [Convolutional neural networks (CNN)](https://pytorch.org/docs/stable/index.html)

Reference input tables are provided in the _'example_input_tables'_ folder. An example to run the scripts is provided in [__*ScriptsToCallSampleData.sh*__](./ScriptsToCallSampleData.sh).

The strategy employed for training the models is nested cross-validation (nested CV), which is an adaptation of the K-fold CV that consists in setting one outer loop and one inner loop of CV. In this approach, the CV in the inner loop is performed on the training set of the outer loop and is used to select the optimum hyperparameter configuration. This step is implemented in the scripts named __*1_XX_With_nestedCV.py*__.

Conversely, the CV in the outer loop is used to train the final model with the selected hyperparameter configuration obtained from the inner loop, and to test the model with the remaining test set that has not been used for hyperparameter selection or training the model. This step is implemented in the scripts named __*2_XX_FinalModel.py*__.

Iterating over different folds in the inner and outer loop allows for the use of different samples in training, validation, and testing in each iteration, optimizing the use of all the available samples. At the end, nested CV generate as many final models as number of folds in the outer loop.

![The figure represents the nested CV approach used in this repository, which consists of a 10-fold CV for the inner loop, and a 5-fold CV for the outer loop.](./images/Figure_NestedCV.png)

--------------------------------------------------------

#### Hyperparameter selection in the inner loop of the nested CV

The grid search approach is employed for hyperparameter selection, which involves listing a set of values for each hyperparameter, and testing all possible combinations. The set of hyperparameters to test is listed in the __*parameters.txt*__ file located in the __Hyperparameters__ directory and is divided into subsections corresponding to each ML method. The [__*parameters.txt*__](Hyperparameters/parameters.txt) file can be modified to define the [values](Hyperparameters/README.md) to test in the grid search.

The first step is to run the inner loop of the nested CV (10-fold CV) for hyperparameter selection. This step is run 5 times, one for each fold in the outer loop ($manualFold parameter in the code from 0 to 4). The example below is to run LR, but the same input parameters apply to all methods except for CNN.

```bash
#Input numeric matrix in .txt format where columns are predictors and rows are samples.
inMtrx=/FullPath/Mtrx.txt
#Labels corresponding to the sample IDs in the input matrix in .txt format. The first column named eid contains sample IDs and the second column named cond contains binary categories: control or disease condition.
inLab=/FullPath/labels.txt 
#Full path to the results directory.
outDir=/FullPath/ToResults/Name
#Name of the disease condition in the input Labels.
PatCond=MS
#Fold from the outer loop of the nested CV used in the job (0 to 4).
manualFold=0

#Run the code for LR
python ./Python_scripts/LR/1_LR_With_nestedCV.py -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold"
```

For the CNN, the command is as follows:

```bash
#Input numeric matrix in .txt format where columns are predictors and rows are samples.
inMtrx=/FullPath/Mtrx.txt
#Labels corresponding to the sample IDs in the input matrix in .txt format. The first column named eid contains sample IDs and the second column named cond contains binary categories: control or disease condition.
inLab=/FullPath/labels.txt 
#Matrix with first column containing predictor IDs named snvIDs and the second column containing chromosome numbers. The columns are tab-separated.
conv_chr=FullPath/chr_mtrx.txt
#Matrix with first column containing sample IDs named snvIDs and the second column containing genomic position in bp. The columns are tab-separated.
conv_pos=FullPath/pos_mtrx.txt
#Full path to the results directory.
outDir=/FullPath/ToResults/Name
#Name of the disease condition in input Labels.
PatCond=MS
#Fold from the outer loop of the nested cross-validation used in the job (0 to 4).
manualFold=0

#Run the code for CNN
python ./Python_scripts/CNN/1_CNN_With_nestedCV.py -m "$inMtrx" -l "$inLab" -a "$conv_chr" -p "$conv_pos" -o "$outDir" -c "$PatCond" -f "$manualFold"
```

The previous code will generate files named _'Name_MS_fold0_LR_EvMetrics_CV.txt'_ and _'Name_MS_fold0_CNN_EvMetrics_CV.txt'_ for LR and CNN, respectively. Each row in the file corresponds to a different hyperparameter configuration, and each column represents the mean and standard deviation of different evaluation metrics obtained from the training and validation subsets. The column containing the hyperparameter configurations lists the hyperparameters separated by vertical bars in the same order as in the [__*parameters.txt*__](Hyperparameters/parameters.txt) file.

The column _'rank_score'_ corresponds to the value of balanced accuracy minus the absolute difference between specificity and sensitivity. This metric is used to rank hyperparamaters from the highest (better performance) to the lowest scores (worse performance). However, performance can be interpreted using other evaluation metrics, and consequently, the best hyperparameter configuration can be selected based on different criteria. 

--------------------------------------------------------

#### Final models from the outer loop of the nested CV

For each fold in the outer loop of the nested CV, the hyperparameter configuration selected in the inner loop is applied in the outer loop. This process is repeated 5 times, once for each fold (specified by the $manualFold parameter in the code from 0 to 4), resulting in 5 different final models.

##### Call for LR

```bash
#Input numeric matrix in .txt format where columns are predictors and rows are samples.
inMtrx=/FullPath/Mtrx.txt
#Labels corresponding to the sample IDs in the input matrix in .txt format. The first column named eid contains sample IDs and the second column named cond contains binary categories: control or disease condition.
inLab=/FullPath/labels.txt 
#Full path to the results directory.
outDir=/FullPath/ToResults/Name
#Name of the disease condition in input Labels.
PatCond=MS
#Fold from the outer loop of the nested cross-validation used in the job (0 to 4).
manualFold=0
#Hyperparameter: Solver or algorithm to use in the optimization problem.
H_so=newton-cg
#Hyperparameter: Inverse of regularization strength.
H_re=1
#Hyperparameter: Balancing rate.
balance=50
#Hyperparameter: Sampling strategy.
sampl_strategy=SMOTE_random

#Run the code for LR
python ./Python_scripts/LR/2_LR_FinalModel.py -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold" -v "$H_so" -r "$H_re" -b "$balance" -s "$sampl_strategy"

```

##### Call for GB

```bash
#Input numeric matrix in .txt format where columns are predictors and rows are samples.
inMtrx=/FullPath/Mtrx.txt
#Labels corresponding to the sample IDs in the input matrix in .txt format. The first column named eid contains sample IDs and the second column named cond contains binary categories: control or disease condition.
inLab=/FullPath/labels.txt 
#Full path to the results directory.
outDir=/FullPath/ToResults/Name
#Name of the disease condition in input Labels.
PatCond=MS
#Fold from the outer loop of the nested cross-validation used in the job (0 to 4).
manualFold=0
#HyperParameter: number of estimators.
H_ne=50
#HyperParameter: learning rate.
H_lr=0.01
#HyperParameter: subsample.
H_sb=0.5
#HyperParameter: max depth.
H_md=None
#HyperParameter: loss function.
H_lo=exponential
#Hyperparameter: Balancing rate.
balance=50
#Hyperparameter: Sampling strategy
sampl_strategy=SMOTE_random

#Run the code for GB
python ./Python_scripts/GB/2_GB_FinalModel.py -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold" -e "$H_ne" -r "$H_lr" -a "$H_sb" -d "$H_md" -u "$H_lo" -b "$balance" -s "$sampl_strategy"

```

##### Call for RF and ET

```bash
#Input numeric matrix in .txt format where columns are predictors and rows are samples.
inMtrx=/FullPath/Mtrx.txt
#Labels corresponding to the sample IDs in the input matrix in .txt format. The first column named eid contains sample IDs and the second column named cond contains binary categories: control or disease condition.
inLab=/FullPath/labels.txt 
#Full path to the results directory.
outDir=/FullPath/ToResults/Name
#Name of the disease condition in input Labels.
PatCond=MS
#Fold from the outer loop of the nested cross-validation used in the job (0 to 4).
manualFold=0
#HyperParameter: number of estimators.
H_ne=50
#HyperParameter: min_samples_split.
H_ss=2
#HyperParameter: min_samples_leaf.
H_sl=5
#HyperParameter: max depth.
H_md=None
#Hyperparameter: Balancing rate.
balance=50
#Hyperparameter: Sampling strategy.
sampl_strategy=SMOTE_random

#Run the code for ET
python ./Python_scripts/ET/2_ET_FinalModel.py -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold" -e "$H_ne" -a "$H_ss" -i "$H_sl" -d "$H_md" -b "$balance" -s "$sampl_strategy"

#Run the code for RF
python ./Python_scripts/RF/2_RF_FinalModel.py -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold" -e "$H_ne" -a "$H_ss" -i "$H_sl" -d "$H_md" -b "$balance" -s "$sampl_strategy"

```

##### Call for FFN

```bash
#Input numeric matrix in .txt format where columns are predictors and rows are samples.
inMtrx=/FullPath/Mtrx.txt
#Labels corresponding to the sample IDs in the input matrix in .txt format. The first column named eid contains sample IDs and the second column named cond contains binary categories: control or disease condition.
inLab=/FullPath/labels.txt 
#Full path to the results directory.
outDir=/FullPath/ToResults/Name
#Name of the disease condition in input Labels.
PatCond=MS
#Fold from the outer loop of the nested cross-validation used in the job (0 to 4).
manualFold=0
#HyperParameter: Number of epochs.
numEp=200
#HyperParameter: learning_rate.
learningRate=0.01
#HyperParameter: dropout rate.
selfdr=0.1
#HyperParameter: number of units (width).
nUnits=100
#HyperParameter: number of layers (depth).
nLayers=2
#Hyperparameter: Balancing rate.
balance=50
#Hyperparameter: Sampling strategy.
sampl_strategy=random

#Run the code for FFN
python ./Python_scripts/FFN/2_FFN_FinalModel.py -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold" -e "$numEp" -r "$learningRate" -d "$selfdr" -u "$nUnits" -y "$nLayers" -b "$balance" -s "$sampl_strategy"

```

##### Call for CNN

```bash
#Input numeric matrix in .txt format where columns are predictors and rows are samples.
inMtrx=/FullPath/Mtrx.txt
#Labels corresponding to the sample IDs in the input matrix in .txt format. The first column named eid contains sample IDs and the second column named cond contains binary categories: control or disease condition.
inLab=/FullPath/labels.txt 
#Matrix with first column containing predictor IDs named snvIDs and the second column containing chromosome numbers. The columns are tab-separated.
conv_chr=FullPath/chr_mtrx.txt
#Matrix with first column containing sample IDs named snvIDs and the second column containing genomic position in bp. The columns are tab-separated.
conv_pos=FullPath/pos_mtrx.txt
#Full path to the results directory.
outDir=/FullPath/ToResults/Name
#Name of the disease condition in input Labels.
PatCond=MS
#Fold from the outer loop of the nested cross-validation used in the job (0 to 4).
manualFold=0
#HyperParameter: Number of epochs.
numEp=200
#HyperParameter: learning_rate.
learningRate=0.01
#HyperParameter: dropout rate.
selfdr=0.1
#HyperParameter: number of units (width).
nUnits=100
#HyperParameter: number of layers (depth).
nLayers=2
#Hyperparameter: Balancing rate.
balance=50
#Hyperparameter: Sampling strategy.
sampl_strategy=random

#Run the code for CNN
python ./Python_scripts/CNN/2_CNN_FinalModel.py -m "$inMtrx" -l "$inLab" -a "$conv_chr" -p "$conv_pos" -o "$outDir" -c "$PatCond" -f "$manualFold" -e "$numEp" -r "$learningRate" -d "$selfdr" -u "$nUnits" -y "$nLayers" -b "$balance" -s "$sampl_strategy"

```

The call of the final models generates several output files (X=fold and XX=method in the file names):

  * __TestFinal_TestFinalResults.txt__: The results of the different final models are summarized in this file. With the results appended row by row.
  * __TestFinal_Trained_XX_FoldX.joblib__: This file contains the trained final model for the ML models (LR, GB, RF and ET).
  * __TestSampl_Trained_XX_X_net.pt__: This file contains the trained final model for the DL models (FFN and CNN).
  * __TestSampl_FoldX_XX_Samples.txt__: Classified samples in the test set with the predicted label and corresponding probabilities.
  * __TestFinal_FoldX_XX_TrainingIDs.txt__: List of samples used for training in the DL models.
  * __TestSampl_FoldX_XX_Predictors.txt__: List of predictors with their corresponding feature importance, as assigned by the ML models. In the case of LR the feature importance corresponds to the coefficient of the features in the decision function.
  * __TestFinal_Attributes_XX_X_lig.txt__: Importance score for each input feature obtained with layer integrated gradients for DL models.
  * __TestFinal_Attributes_XX_X_dl.txt__: Importance score for each input feature obtained with DeepLIFT algorithm for DL models.
  * __TestFinal_Attributes_XX_X_sl.txt__: Importance score for each input feature obtained with saliency maps for DL models.
  * __TestFinal_Attributes_XX_X_gbp.txt__: Importance score for each input feature obtained with guided backpropagation for DL models.

