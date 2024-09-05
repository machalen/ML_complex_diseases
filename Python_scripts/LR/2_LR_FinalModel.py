#!/usr/bin/env python3

"""
#Magda Arnal
#14/02/2023
#Run the model with Logistic Regression (LR) and the final set of hyperparameters.
"""

### import libraries
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import joblib #to save the models

# for number-crunching
import numpy as np

# for dataset management
import pandas as pd

#For system
import os
from optparse import OptionParser

#Disable warnings: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

######################################################################################
######################################################################################

parser = OptionParser()

parser.add_option('-m', '--inputMtrx', help='Input numeric matrix in .txt format where columns are predictors and rows are samples.', 
                  metavar='FullPath/Mtrx.txt')
parser.add_option('-l', '--inputLabels',
                  help='Labels corresponding to the sample IDs in the input matrix in .txt format. The first column named eid contains sample IDs and the second column named cond contains binary categories: control or disease condition.',
                  metavar='FullPath/labels.txt')
parser.add_option('-o', '--outputDir',
                  help='Full path to the results directory.',
                  metavar='FullPath')
parser.add_option('-c', '--condition',
                  help='Name of the disease condition in inputLabels.',
                  metavar='MS')
parser.add_option('-f', '--fold',
                  help='Fold from the outer loop of the nested cross-validation used in the job (0 to 4).',
                  metavar='4')
parser.add_option('-v', '--Solver',
                  help='Hyperparameter: Algorithm to use in the optimization problem.',
                  metavar='newton-cg')
parser.add_option('-r', '--creg',
                  help='Hyperparameter: Inverse of regularization strength.',
                  metavar=0.01)
parser.add_option('-b', '--Balance',
                  help='Balancing rate',
                  metavar='50')
parser.add_option('-s', '--Undersampling',
                  help='Sampling strategy',
                  metavar='SMOTE_ENN')

(options, args) = parser.parse_args()

#Assess the input variables
#Convert <class 'optparse.Values'> to dictionary
option_dict = vars(options)
#Save elements in the dictionary to variables
inMtrx = option_dict['inputMtrx']
inLab = option_dict['inputLabels']
outDir = option_dict['outputDir']
PatCond = option_dict['condition']
manualFold = int(option_dict['fold'])
H_so = option_dict['Solver']
H_re = float(option_dict['creg'])
balance=int(option_dict['Balance'])
sampl_strategy=option_dict['Undersampling']

#########################################################################
#############################Load data###################################
#Load matrix with variants
df = pd.read_csv(inMtrx, sep="\t")
PredNum = df.shape[1]

#Load labels
labels = pd.read_csv(inLab, sep="\t")

#Get the name of the files to output them in results:
inMtrxName=os.path.basename(inMtrx) + '//' + os.path.basename(inLab)

################################################################################
################################################################################
#Select rows for case and controls
rows = labels.cond.values
casei = np.where(rows==PatCond)[0]
cntrli  = np.where(rows=='control')[0]
CaseNumber = len(casei)
CntrlNumber = len(cntrli)

print('There are ' + str(CaseNumber)+ ' cases in total')
print('There are ' + str(CntrlNumber)+ ' controls in total')
###############################################################################
###############################################################################
#Append labels to the numeric matrix
Xtab=pd.concat([labels, df], axis=1)

#Convert to numeric matrices and vectors with numpy
con1=labels.iloc[:,1]
y=np.where(con1==PatCond, 1, con1)
y=np.where(y=='control', 0, y)
y=y.astype('int')
#Shuffle samples to avoid possible bias when sampling
Xtab, y = shuffle(Xtab, y, random_state=1)
#Check that the cases and controls are properly assigned after shuffle
# con2=Xtab.iloc[:,1]
# yt=np.where(con2==PatCond, 1, con2)
# yt=np.where(yt=='control', 0, yt)
# yt=yt.astype('int')
# print(np.array_equal(y, yt))#TRUE!

################################################################################
##############################Make nested CV####################################
np.random.seed(0)
#Outer CV, select the fold in the outer layer of cross-validation
outer_strat = StratifiedKFold(shuffle=False, n_splits = 5)#From 0 to 4

for fold, (a,b) in enumerate(outer_strat.split(Xtab, y)):
    #print(fold)
    if fold == manualFold :
        train_dev_index=a
        test_index=b

X_trtab=Xtab.iloc[train_dev_index,:]
Y_train=y[train_dev_index]
X_tetab=Xtab.iloc[test_index,:]
Y_test=y[test_index]

################################################################################
##########################Check the split#######################################

print ('Train set:', X_trtab.shape,  Y_train.shape)
unique_values, counts = np.unique(Y_train, return_counts=True)
print ('Train/Dev set has: ' + str(counts[0]) +' '+ str(unique_values[0])+ ' and ' + str(counts[1]) + ' '+ str(unique_values[1]))

print ('Test set:', X_tetab.shape,  Y_test.shape)
unique_values, counts = np.unique(Y_test, return_counts=True)
print ('Test set has: ' + str(counts[0]) +' '+ str(unique_values[0])+ ' and ' + str(counts[1]) + ' '+ str(unique_values[1]))

#################################################################################
#################################################################################
#Check that the cases and controls are properly assigned after cross fold split
# con2=X_trtab.iloc[:,1]
# ytr=np.where(con2==PatCond, 1, con2)
# ytr=np.where(ytr=='control', 0, ytr)
# ytr=ytr.astype('int')
# print(np.array_equal(Y_train, ytr))#TRUE!

# con2=X_tetab.iloc[:,1]
# yte=np.where(con2==PatCond, 1, con2)
# yte=np.where(yte=='control', 0, yte)
# yte=yte.astype('int')
# print(np.array_equal(Y_test, yte))#TRUE!

#################################################################################
#################################################################################
#Define and apply the sampling functions

balance_rate=(100-balance)/balance
rus = RandomUnderSampler(random_state=0, sampling_strategy= balance_rate)

if sampl_strategy == 'random' :
    #Apply random undersampling
    x_trtab, y_train = rus.fit_resample(X_trtab, Y_train)
elif sampl_strategy == 'ENN' :
    #Define ENN
    enn = EditedNearestNeighbours(sampling_strategy='majority')
    #Convert the X_trtab to numeric for enn method:
    x_num=X_trtab.iloc[:,labels.shape[1]:len(X_trtab.columns)]
    x_num=x_num.to_numpy()
    #Apply ENN:
    _, y_resampled = enn.fit_resample(x_num, Y_train)
    subsampled_indices = enn.sample_indices_
    X_resampled=X_trtab.iloc[subsampled_indices]
    #Apply random undersampling for balancing:
    x_trtab, y_train = rus.fit_resample(X_resampled, y_resampled)
elif sampl_strategy == 'SMOTE_random' :
    # Calculate the target number of samples for the minority class
    minority_class_size = int(0.20 * len(Y_train[Y_train == 1]))
    # Set the sampling strategy to oversample the minority class
    sampl=(minority_class_size + len(Y_train[Y_train == 1]))/len(Y_train[Y_train == 0])
    #Define SMOTE
    sm = SMOTE(sampling_strategy=sampl, random_state=0)
    #Convert the X_trtab to numeric for smote method:
    x_num=X_trtab.iloc[:,labels.shape[1]:len(X_trtab.columns)]
    x_num=x_num.to_numpy()
    #Apply SMOTE:
    X_oversampled, y_oversampled = sm.fit_resample(x_num, Y_train)
    #Apply random undersampling for balancing:
    x_trtab, y_train = rus.fit_resample(X_oversampled, y_oversampled)
elif sampl_strategy == 'SMOTE_ENN' :
    # Calculate the number of samples for the minority class
    minority_class_size = int(0.20 * len(Y_train[Y_train == 1]))
    # Set the sampling strategy to oversample the minority class
    sampl=(minority_class_size + len(Y_train[Y_train == 1]))/len(Y_train[Y_train == 0])
    #define SMOTE
    sm = SMOTE(sampling_strategy=sampl, random_state=0)
    #define ENN
    enn = EditedNearestNeighbours(sampling_strategy='majority')
    #Convert the X_trtab to numeric for smote and enn method:
    x_num=X_trtab.iloc[:,labels.shape[1]:len(X_trtab.columns)]
    x_num=x_num.to_numpy()
    #Apply SMOTE:
    X_oversampled, y_oversampled = sm.fit_resample(x_num, Y_train)
    #Apply ENN:
    X_resampled, y_resampled = enn.fit_resample(X_oversampled, y_oversampled)
    #Apply random undersampling for balancing:
    x_trtab, y_train = rus.fit_resample(X_resampled, y_resampled)

##################################################################################
##################################################################################
if sampl_strategy in ['random','ENN'] :
    #Check that the cases and controls are properly assigned after undersampling
    # con2=x_trtab.iloc[:,1]
    # ytr=np.where(con2==PatCond, 1, con2)
    # ytr=np.where(ytr=='control', 0, ytr)
    # ytr=ytr.astype('int')
    # print(np.array_equal(y_train, ytr))#TRUE!
    #Save the labels
    x_trlab=x_trtab.iloc[:,0:labels.shape[1]]
    x_trlab.columns=labels.columns
    f1_output=outDir+'Fold'+str(manualFold)+'_LR_TrainingIDs.txt'
    x_trlab.to_csv(f1_output, index=None, sep='\t')
    #Split and create the numeric np matrix
    x_train=x_trtab.iloc[:,labels.shape[1]:len(x_trtab.columns)]
    x_train=x_train.to_numpy()
else:
    x_train=x_trtab

#I don't do undersampling to the test set!
#Save the labels
x_telab=X_tetab.iloc[:,0:labels.shape[1]]
x_telab.columns=labels.columns
#Split and create the numeric np matrix
x_test=X_tetab.iloc[:,labels.shape[1]:len(X_tetab.columns)]
x_test=x_test.to_numpy()

#Look at the number of cases and controls in train and test
unique, counts = np.unique(y_train, return_counts=True)
print('Counts in training:',np.asarray((unique, counts)).T)

unique, counts = np.unique(Y_test, return_counts=True)
print('Counts in testing:',np.asarray((unique, counts)).T)
######################################################################################
######################################################################################

#Define the model with parameters provided in the input
modLR = LogisticRegression(solver=H_so,
                             C=H_re,
                             random_state=0,
                             max_iter=1000)
modLR.fit(x_train, y_train)

#Save the dataframe with the feature importances
PredictVars_t = pd.DataFrame(modLR.coef_)
PredictVars=PredictVars_t.T
PredictVars['SNV']=df.columns
PredictVars.rename({0: 'FI'}, axis=1, inplace=True)#Change the name to match the feature importance column obtained with tree-based methods.
print(PredictVars[PredictVars['FI'] > 0.01].sort_values(by=['FI'], ascending=False))
f2_output=outDir+'Fold'+str(manualFold)+'_LR_Predictors.txt'
PredictVars.to_csv(f2_output, index=None, sep='\t')

#Calculate accuracy
yhat=modLR.predict(x_test)
m_acc=metrics.accuracy_score(Y_test, yhat)
print('Accuracy is:',m_acc)
#Calculate specificity
m_acc0 = metrics.accuracy_score(Y_test[Y_test == 0], yhat[Y_test == 0])
#Calculate sensitivity
m_acc1 = metrics.accuracy_score(Y_test[Y_test == 1], yhat[Y_test == 1])
#Calculate ROC AUC
mpred = modLR.predict_proba(x_test)
pred=mpred[:,1]
fpr, tpr, thresholds = metrics.roc_curve(Y_test, pred, pos_label=1)
m_roc=metrics.auc(fpr, tpr)
print('ROC AUC is:',m_roc)
#Calculate the fbeta score
m_fscore=metrics.fbeta_score(Y_test, yhat, average='binary', beta=1)
print('f-beta is:', m_fscore)

#Confusion matrix
c_m=metrics.confusion_matrix(Y_test, yhat)
TN=c_m[0,0]
TP=c_m[1,1]
FN=c_m[1,0]
FP=c_m[0,1]

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

# print('Accuracy is:',ACC)
print('Sensitivity is:',TPR)
print('Specificity is:',TNR)
print('Positive Predicted Value is:',PPV)
print('Negative Predicted Value is:',NPV)

#Save the dataframe with the predictions
x_telab['y_pred']=yhat
x_telab['Prob0']=mpred[:,0]
x_telab['Prob1']=mpred[:,1]
f3_output=outDir+'Fold'+str(manualFold)+'_LR_Samples.txt'
x_telab.to_csv(f3_output, index=None, sep='\t')

#save the model
joblib.dump(modLR, outDir + 'Trained_LR_'+'Fold'+str(manualFold)+'.joblib', compress=3)

#############################################################################################
#############################################################################################
#Confirm that the metrics are the same after loading the model
# loaded_modLR=joblib.load(outDir + 'Trained_LR_'+'Fold'+str(manualFold)+'.joblib')
# yhat=loaded_modLR.predict(x_test)

# #Confusion matrix
# c_m=metrics.confusion_matrix(Y_test, yhat)
# TN=c_m[0,0]
# TP=c_m[1,1]
# FN=c_m[1,0]
# FP=c_m[0,1]

# # Sensitivity, hit rate, recall, or true positive rate
# TPR_load = TP/(TP+FN)
# # Specificity or true negative rate
# TNR_load = TN/(TN+FP)
# # Precision or positive predictive value
# PPV_load = TP/(TP+FP)
# # Negative predictive value
# NPV_load = TN/(TN+FN)
# # Overall accuracy
# ACC_load = (TP+TN)/(TP+FP+FN+TN)

# print('Accuracy is:',ACC_load)
# print('Sensitivity is:',TPR_load)
# print('Specificity is:',TNR_load)
# print('Positive Predicted Value is:',PPV_load)
# print('Negative Predicted Value is:',NPV_load)

###################################################################################################
###############################Output results in the general table#################################
#Path to save the final results of the evaluation metrics for the test set.
GlobalResults=outDir+'TestFinalResults.txt'

#Convert the set of parameters to strings
c=(H_so,H_re,balance,sampl_strategy)
Hiper='|'.join(str(p) for p in c)

# Check if the file exists
if not os.path.exists(GlobalResults):
    header = "condition\tmodel\tinput_matrix\tn_predictors\tn_controls\tn_cases\thyperparameters\tfold\taccuracy\tspecificity\tsensitivity\trocauc\tfscore\tPPV\tNPV\n"
    # If the file doesn't exist, create it and write the header
    with open(GlobalResults, 'w') as f:
        f.write(header)

with open(GlobalResults, 'a+') as f:
    f.write(PatCond +'\t'+'LR'+'\t'+ inMtrxName +
    '\t' + str(PredNum) +
    '\t' + str(CntrlNumber) +
    '\t' + str(CaseNumber) +
    '\t' + 'Hiper:' + Hiper +
    '\t' + 'Fold' + str(manualFold) +
    '\t' + str(round(m_acc,4)) +
    '\t' + str(round(m_acc0,4)) +
    '\t' + str(round(m_acc1,4)) +
    '\t' + str(round(m_roc,4)) +
    '\t' + str(round(m_fscore,4)) +
    '\t' + str(round(PPV,4)) +
    '\t' + str(round(NPV,4)) + '\n')
    f.close()