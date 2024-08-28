#!/usr/bin/env python3

"""
#Magda Arnal
#07/07/2023
#Run the Extremely Randomized Trees (ET) with nested CV 
"""

import sys
import os

# Add the directory containing the Imports folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

### import libraries
from sklearn.utils import shuffle
from lib.ML_Functions import Cross_Val_Groups
from lib.ML_Functions import MakeEvalSum
from lib.ML_Functions import read_parameters
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

# for number-crunching
import numpy as np

# for dataset management
import pandas as pd

#For system
from optparse import OptionParser
from datetime import datetime

#Disable warnings: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

###############################################################################
###############################################################################
#Read the data
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
                  metavar='1')

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

#########################################################################
#########################Load data#######################################
#Load matrix with variants
df = pd.read_csv(inMtrx, sep="\t")

#Load labels
labels = pd.read_csv(inLab, sep="\t")

#########################################################################
#########################################################################
#Select rows for cases and controls
rows = labels.cond.values
casei = np.where(rows==PatCond)[0]
cntrli  = np.where(rows=='control')[0]
CaseNumber = len(casei)
CntrlNumber = len(cntrli)

print('There are ' + str(CaseNumber)+ ' cases in total')
print('There are ' + str(CntrlNumber)+ ' controls in total')
###############################################################################
###############################################################################
#Convert to numeric matrices and vectors with numpy and shuffle
X_all=df.to_numpy()
PredNum = X_all.shape[1]
con1=labels.iloc[:,1]
y_all=np.where(con1==PatCond, 1, con1)
y_all=np.where(y_all=='control', 0, y_all)
y_all=y_all.astype('int')
X_all, y_all = shuffle(X_all, y_all, random_state=1)

################################################################################
##############################Make nested CV####################################
#Outer CV, select the fold in the outer layer of the nested cross-validation
outer_strat = StratifiedKFold(shuffle=False, n_splits = 5)#From 0 to 4

for fold, (a,b) in enumerate(outer_strat.split(X_all, y_all)):
    #print(fold)
    if fold == manualFold :
        train_dev_index=a
        test_index=b

X_train_dev=X_all[train_dev_index]
y_train_dev=y_all[train_dev_index]
X_test=X_all[test_index]
y_test=y_all[test_index]

################################################################################
################################################################################
#Check the split
unique_values, counts = np.unique(y_train_dev, return_counts=True)
print('Train/Dev set has: ' + str(counts[0]) +' '+ str(unique_values[0])+ ' and ' + str(counts[1]) + ' '+ str(unique_values[1]))

unique_values, counts = np.unique(y_test, return_counts=True)
print('Test set has: ' + str(counts[0]) +' '+ str(unique_values[0])+ ' and ' + str(counts[1]) + ' '+ str(unique_values[1]))

##################################################################################################
##################################################################################################
# define the model
model=ExtraTreesClassifier()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the parameters.txt file two directories above the script location
file_path = os.path.join(script_dir, '..', '..', 'Hyperparameters', 'parameters.txt')

#Read the hyperparameters for the grid search
grid = read_parameters(file_path, 'ET')
keys_list = list(grid)

#Initialize lists to store the evaluation metrics
metrics_prec = {}
metrics_rec = {}
metrics_fscore = {}
metrics_acc = {}
metrics_acc0 = {}
metrics_acc1 = {}
metrics_roc = {}
metrics_pr = {}


now1 = datetime.now()#Start timer
for e in range(len(grid[keys_list[0]])):
    ne=grid[keys_list[0]][e]
    for l in range(len(grid[keys_list[1]])):
        ms=grid[keys_list[1]][l]
        for s in range(len(grid[keys_list[2]])):
            ml=grid[keys_list[2]][s]
            for m in range(len(grid[keys_list[3]])):
                md=grid[keys_list[3]][m]
                for b in range(len(grid[keys_list[4]])):
                    ba=grid[keys_list[4]][b]
                    for n in range(len(grid[keys_list[5]])):
                        sa=grid[keys_list[5]][n]
                        #print the combinations
                        #print(ne,ms,ml,md,ba, sa)
                        c=(ne,ms,ml,md,ba,sa)
                        combination = {'n_estimators':ne,
                                       'min_samples_split':ms,
                                       'min_samples_leaf':ml,
                                       'max_depth': md,
                                       'Balance':ba,
                                       'Sampling':sa}
                        metrics_prec[c], metrics_rec[c], metrics_fscore[c],metrics_acc[c], metrics_acc0[c], metrics_acc1[c], metrics_roc[c], metrics_pr[c] = Cross_Val_Groups(model, X=X_train_dev, y=y_train_dev, combination=combination, n_splits = 10)


now2 = datetime.now()#End timer
print(now2-now1)

##################################################################################################
##################################################################################################
#Merge the dict and convert to pandas data frame
eval_df = MakeEvalSum(mtrx_acc=metrics_acc, mtrx_acc0=metrics_acc0, mtrx_acc1=metrics_acc1, mtrx_prec=metrics_prec, mtrx_recall=metrics_rec, mtrx_fscore=metrics_fscore, mtrx_roc=metrics_roc, mtrx_pr=metrics_pr)
#Save the pandas data.frame
f_output=outDir+PatCond+'_fold'+str(manualFold)+'_ET_EvMetrics_CV.txt'
eval_df.to_csv(f_output, index=None, sep='\t')
