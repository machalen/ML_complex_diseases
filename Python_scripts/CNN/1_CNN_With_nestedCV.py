#!/usr/bin/env python3

"""
#Magda Arnal
#30/01/2023
#Run the CNN with nested CV 
"""

### import libraries
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import fbeta_score
#For system
import sys
import os

# Add the directory containing the lib folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#Import the class object and functions
from lib.CNN_classes import CNN_WithCV
from lib.CNN_classes import PlotPerformance
from lib.ML_Functions import read_parameters

# for number-crunching
import numpy as np

# for dataset management
import pandas as pd

#For input
from optparse import OptionParser
#For time
from datetime import datetime

#Important to set a random seed!
torch.manual_seed(0)
np.random.seed(0)

###########################################################################################
###########################################################################################
#Read the data
parser = OptionParser()

parser.add_option('-m', '--inputMtrx', help='Input numeric Matrix in .txt format where columns are predictors and rows are samples.', 
                  metavar='FullPath/Mtrx.txt')
parser.add_option('-l', '--inputLabels',
                  help='Labels corresponding to the sample IDs in the input matrix in .txt format. The first column named eid contains sample IDs and the second column named cond contains binary categories: control or disease condition.',
                  metavar='FullPath/labels.txt')
parser.add_option('-a', '--convChr',
                  help='Matrix with first column containing predictor IDs named snvIDs and the second column containing chromosome numbers. The columns are tab-separated.',
                  metavar='FullPath/Conv1D_chr_mtrx.txt')
parser.add_option('-p', '--convPos',
                  help='Matrix with first column containing sample IDs named snvIDs and the second column containing genomic position in bp. The columns are tab-separated.',
                  metavar='FullPath/Conv1D_pos_mtrx.txt')
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
conv_chr=option_dict['convChr']
conv_pos=option_dict['convPos']
outDir = option_dict['outputDir']
PatCond = option_dict['condition']
manualFold = int(option_dict['fold'])

# use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#########################################################################
##########################Load data######################################
#Load matrix with variants
df = pd.read_csv(inMtrx, sep="\t")

#Load labels
labels = pd.read_csv(inLab, sep="\t")

#Load the information about chromosomes
df_chr = pd.read_csv(conv_chr, sep="\t")

#Load the information about the position
df_pos = pd.read_csv(conv_pos, sep="\t")

###############################################################################
###############################################################################
#Select rows for cases and controls
rows = labels.cond.values
casei = np.where(rows==PatCond)[0]
cntrli  = np.where(rows=='control')[0]
CaseNumber = len(casei)
CntrlNumber = len(cntrli)

print('There are ' + str(CaseNumber)+ ' cases in total')
print('There are ' + str(CntrlNumber)+ ' controls in total')

################################################################################
################################################################################
#Convert to numeric matrices and vectors with numpy
X=df.to_numpy()
# l=X.sum(axis=0)
PredNum = X.shape[1]
con1=labels.iloc[:,1]
y=np.where(con1==PatCond, 1, con1)
y=np.where(y=='control', 0, y)
y=y.astype('int')
X, y = shuffle(X, y, random_state=1)
# m=X.sum(axis=0)
# np.array_equal(l,m)#TRUE I confirm that the order of the variants (rows) is maintained

#Convert the numeric values of the chromosome and position matrices to numpy
chr_col=df_chr.iloc[:,1].to_numpy()
pos_col=df_pos.iloc[:,1].to_numpy()

#Check if the order of the variants is the same in the three matrices
# mtrx1_snvs=df.columns.tolist()
# mtrx2_snvs=df_chr["snvIDs"].tolist()
# mtrx3_snvs=df_pos["snvIDs"].tolist()
# print(mtrx1_snvs== mtrx2_snvs)#TRUE!
# print(mtrx1_snvs== mtrx3_snvs)#True!

################################################################################
################Normalize the values to the range of -1 to 1####################
#Normalize X
X = X - np.min(X)
X = 2 * X / (np.max(X) - np.min(X)) - 1
# np.min(X)#-1
# np.max(X)#1

#Normalize chromosome
chr_col = chr_col - np.min(chr_col)
chr_col = 2 * chr_col / (np.max(chr_col) - np.min(chr_col)) - 1
# np.min(chr_col)#-1
# np.max(chr_col)#1

#Normalize position
pos_col = pos_col - np.min(pos_col)
pos_col = 2 * pos_col / (np.max(pos_col) - np.min(pos_col)) - 1
# np.min(pos_col)#-1
# np.max(pos_col)#1

################################################################################
################################################################################
#Outer CV, select the fold in the outer layer of cross-validation
outer_strat = StratifiedKFold(n_splits = 5)#From 0 to 4

for fold, (a,b) in enumerate(outer_strat.split(X, y)):
    #print(fold)
    if fold == manualFold :
        train_dev_index=a
        test_index=b

X_train_dev_np=X[train_dev_index,:]
y_train_dev=y[train_dev_index]
X_test_np=X[test_index,:]
y_test=y[test_index]

################################################################################
################################################################################
#Check the split

unique_values, counts = np.unique(y_train_dev, return_counts=True)
print ('Train/Dev set has: ' + str(counts[0]) +' '+ str(unique_values[0])+ ' and ' + str(counts[1]) + ' '+ str(unique_values[1]))

unique_values, counts = np.unique(y_test, return_counts=True)
print ('Test set has: ' + str(counts[0]) +' '+ str(unique_values[0])+ ' and ' + str(counts[1]) + ' '+ str(unique_values[1]))

###################################################################################
###########################DEFINE THE FUNCTION#####################################
def trainTheModel(numEp,learningRate, selfdr,nUnits,nLayers,Sampling, Balance):
    
    #Apply cross validation
    n_splits=10
    strat = StratifiedKFold(n_splits = n_splits)
    
    ###############################################################
    #Define the function to apply the sampling
    balance_rate=(100-Balance)/Balance
    rus = RandomUnderSampler(random_state=0, sampling_strategy= balance_rate)
    
    if Sampling == 'ENN':
        enn = EditedNearestNeighbours(sampling_strategy='majority', n_jobs=5)
    elif Sampling == 'SMOTE_random' :
        # Calculate the number of samples for the minority class
        minority_class_size = int(0.20 * len(y_train_dev[y_train_dev == 1]))
        # Set the sampling strategy to oversample the minority class
        sampl=(minority_class_size + len(y_train_dev[y_train_dev == 1]))/len(y_train_dev[y_train_dev == 0])
        sm = SMOTE(sampling_strategy=sampl, random_state=0)
    elif Sampling == 'SMOTE_ENN' :
        # Calculate the number of samples for the minority class
        minority_class_size = int(0.20 * len(y_train_dev[y_train_dev == 1]))
        # Set the sampling strategy to oversample the minority class
        sampl=(minority_class_size + len(y_train_dev[y_train_dev == 1]))/len(y_train_dev[y_train_dev == 0])
        sm = SMOTE(sampling_strategy=sampl, random_state=0)
        #ENN
        enn = EditedNearestNeighbours(sampling_strategy='majority', n_jobs=5)
    
    ###############################################################
    # initialize matrices to store the evaluation metrics
    losses   = np.zeros((numEp,n_splits))
    trainAcc = np.zeros((numEp,n_splits))
    devAcc  = np.zeros((numEp,n_splits))
    devAcc0  = np.zeros((numEp,n_splits))
    devAcc1  = np.zeros((numEp,n_splits))
    devFscore = np.zeros((numEp,n_splits))
    
    ###############################################################
    #Loop over inner CV
    for fold, (train_index, dev_index) in enumerate(strat.split(X_train_dev_np, y_train_dev)):

        x_train_np = X_train_dev_np[train_index]
        y_train = y_train_dev[train_index]
        x_dev_np = X_train_dev_np[dev_index]
        y_dev = y_train_dev[dev_index]
        
        #############################################################################
        #apply the sampling 
        if Sampling == 'random' :
            #Apply random undersampling
            X_train_np, Y_train = rus.fit_resample(x_train_np, y_train)
        elif Sampling == 'ENN' :
            #Apply ENN
            X_resampled, y_resampled = enn.fit_resample(x_train_np, y_train)
            #Apply random undersampling
            X_train_np, Y_train = rus.fit_resample(X_resampled, y_resampled)
        elif Sampling == 'SMOTE_random' :
            #Apply SMOTE
            X_oversampled, y_oversampled = sm.fit_resample(x_train_np, y_train)
            #Apply random undersampling
            X_train_np, Y_train = rus.fit_resample(X_oversampled, y_oversampled)
        elif Sampling == 'SMOTE_ENN' :
            #Apply SMOTE
            X_oversampled, y_oversampled = sm.fit_resample(x_train_np, y_train)
            #Apply ENN
            X_resampled, y_resampled = enn.fit_resample(X_oversampled, y_oversampled)
            #Apply random undersampling
            X_train_np, Y_train = rus.fit_resample(X_resampled, y_resampled)
        
        #################################
        #Convert to tensors to input to the FFN
        Y_train = torch.tensor(Y_train).float()
        y_dev = torch.tensor(y_dev).float()
        
        # we'll actually need the labels to be a "matrix"
        Y_train = Y_train[:,None]
        y_dev = y_dev[:,None]

        ###############################################
        #Build the 3D tensors for the train set
        shape_0=X_train_np.shape[0] #number of samples
        shape_1=3 #Number of channels (variant, chromosome and position)
        shape_2=X_train_np.shape[1] #number of variants
        X_train = torch.empty(size=(shape_0, shape_1, shape_2))
        
        for i in range(shape_0):
            #Add the variant channel
            X_train[i, 0, :]=torch.tensor(X_train_np[i,:])
            #Add the chromosome channel
            X_train[i, 1, :]=torch.tensor(chr_col)
            #Add the position channel
            X_train[i, 2, :]=torch.tensor(pos_col)
        
        ###############################################
        #Build the 3D tensors for the dev set
        shape_0=x_dev_np.shape[0] #number of samples
        shape_1=3 #Number of channels (variant, chromosome and position)
        shape_2=x_dev_np.shape[1]#number of variants
        x_dev = torch.empty(size=(shape_0, shape_1, shape_2))
        
        for i in range(shape_0):
            #Add the variant channel
            x_dev[i, 0, :]=torch.tensor(x_dev_np[i,:])
            #Add the chromosome channel
            x_dev[i, 1, :]=torch.tensor(chr_col)
            #Add the position channel
            x_dev[i, 2, :]=torch.tensor(pos_col)
        
        ##################################
        #Send the data to the GPU
        X_train=X_train.to(device)
        x_dev=x_dev.to(device)
        
        # create a new model from CNN_classes.py
        net,lossfun,optimizer = CNN_WithCV(learningRate, selfdr,nUnits,nLayers,inUnits=PredNum)
        #Send the data to the GPU
        net.to(device)

        # loop over epochs
        for epochi in range(numEp):
            # loop over training data batches
            net.train() # switch to train mode
            
            #Send the data to the GPU at the beginning of the epoch
            Y_train=Y_train.to(device)

            # forward pass and loss
            yHat = net(X_train)
            loss = lossfun(yHat,Y_train)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Send Y variable back to CPU for calculations. The real and predicted values.
            Y_train=Y_train.cpu()
            yHat=yHat.cpu()

            # now that we've trained through the batches, get their average training accuracy
            yHat_pre=(yHat>0).float()#Translate continuous to 0/1
            trainAcc[epochi,fold] = torch.mean((yHat_pre == Y_train).float()).item()
            
            # and get average losses across the batches
            losses[epochi,fold] = loss.item()
            
            # test accuracy
            net.eval() # switch to test mode
            with torch.no_grad(): # deactivates autograd
                yHat = net(x_dev)
            
            yHat=yHat.cpu()
            yHat_pre=(yHat>0).float()#Translate continuous to 0/1
            #Save accuracy on evaluation/validation set
            devAcc[epochi,fold]=torch.mean((yHat_pre == y_dev).float()).item()
            #Save specificity on evaluation/validation set
            devAcc0[epochi,fold]= torch.mean((yHat_pre[y_dev == 0] == y_dev[y_dev == 0]).float()).item()
            #Save sensitivity on evaluation/validation set
            devAcc1[epochi,fold] = torch.mean((yHat_pre[y_dev == 1] == y_dev[y_dev == 1]).float()).item()
            #Save F-score on evaluation/validation set
            devFscore[epochi,fold]=fbeta_score(y_dev, yHat_pre, average='binary', beta=1)
    
    # function output
    return trainAcc, devAcc, devAcc0, devAcc1, devFscore, losses, net

##############################################################################################
###################################EXPERIMENT#################################################
#Test several parmeters in a loop

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the parameters.txt file two directories above the script location
file_path = os.path.join(script_dir, '..', '..', 'Hyperparameters', 'parameters.txt')

#Read the hyperparameters for the grid search
params = read_parameters(file_path, 'CNN')

numEpV=params.get('numep', [])
learningRateV=params.get('learning_rate', [])
selfdrV=params.get('selfdr', [])
nUnitsV=params.get('nunits', [])
nLayersV=params.get('nlayers', [])
BalanceV=params.get('balance', [])
SamplingV=params.get('sampl_strategy', [])

#Hyperparameters tested
hyperparam = []
#F-score in the validation set (mean and sd across the folds in the inner loop)
fscore_dev_mean = []
fscore_dev_std = []
#Accuracy in the training set (mean and sd across the folds in the inner loop)
acc_train_mean = []
acc_train_std=[]
#Accuracy in the validation set (mean and sd across the folds in the inner loop)
acc_dev_mean = []
acc_dev_std = []
#Sensitivity in the validation set (mean and sd across the folds in the inner loop)
acc1_dev_mean = []
acc1_dev_std = []
#Specificity in the validation set (mean and sd across the folds in the inner loop)
acc0_dev_mean = []
acc0_dev_std = []
#Other scores to evaluate the models
loss_score1 = []
loss_score2 = []
TotalRank = []

# Start the timer!
now1 = datetime.now()#Start timer

#Run the expirement
for numEp in numEpV:
    for learningRate in learningRateV:
        for selfdr in selfdrV:
            for nUnits in nUnitsV:
                for nLayers in nLayersV:
                    for Balance in BalanceV:
                        for Sampling in SamplingV:
                            trainAcc,devAcc, devAcc0, devAcc1, devFscore, losses, net = trainTheModel(numEp=numEp,learningRate=learningRate, selfdr=selfdr,nUnits=nUnits,nLayers=nLayers,Sampling=Sampling, Balance=Balance)

                            #Save the parameter names
                            #Append the hyperParameter configuration
                            grid_name=[str(numEp),str(learningRate),str(selfdr),str(nUnits),str(nLayers),str(Balance),str(Sampling)]
                            hyperparam.append('|'.join(grid_name))
                            
                            #Select last rows for all matrices, corresponding to the last epoch (some methods use the mean of the last three rows)
                            trainAcc_last = trainAcc[-1,:]
                            devAcc_last = devAcc[-1,:]
                            devAcc0_last=devAcc0[-1,:]
                            devAcc1_last=devAcc1[-1,:]
                            devFscore_last = devFscore[-1,:]

                            #Save evaluation metrics of the train/dev set, mean and sd across the folds in the inner loop
                            fscore_dev_mean.append(np.mean(devFscore_last))
                            fscore_dev_std.append(np.std(devFscore_last))

                            acc_train_mean.append(np.mean(trainAcc_last))
                            acc_train_std.append(np.std(trainAcc_last))

                            acc_dev_mean.append(np.mean(devAcc_last))
                            acc_dev_std.append(np.std(devAcc_last))

                            a0=np.mean(devAcc0_last)
                            acc0_dev_mean.append(a0)
                            acc0_dev_std.append(np.std(devAcc0_last))

                            a1=np.mean(devAcc1_last)
                            acc1_dev_mean.append(a1)
                            acc1_dev_std.append(np.std(devAcc1_last))

                            #Append scores in order to evaluate the model
                            s1=(a0+a1)/2
                            s2=abs(a0 - a1)
                            s3=s1-s2
                            loss_score1.append(s1)
                            loss_score2.append(s2)
                            TotalRank.append(s3)

                            #Plot only the best configurations
                            if (a0 > 0.61 and a1 > 0.61):
                                outpath=outDir+'CNN_SeveralParams'+'_fold' + str(manualFold)+'_numEp'+ str(numEp) +'_learningRate'+ str(learningRate)+'_selfdr'+str(selfdr)+'_nUnits'+str(nUnits)+'_nLayers' +str(nLayers)+ '_balancing' + str(Balance)+'_sampling' + str(Sampling) +'.pdf'
                                PlotPerformance(trainAcc,devAcc, devAcc0, devAcc1, devFscore, losses, outpath)

#Print the time required for the calculations.
now2 = datetime.now()#End timer
print(now2-now1)

#Save the full results in a table
nested_CV_results=pd.DataFrame()
nested_CV_results['HyperParam']  = hyperparam
nested_CV_results['acc_train_mean']  = acc_train_mean
nested_CV_results['acc_train_std']  = acc_train_std
nested_CV_results['acc_dev_mean']  = acc_dev_mean
nested_CV_results['acc_dev_std']  = acc_dev_std
nested_CV_results['acc0_dev_mean']  = acc0_dev_mean
nested_CV_results['acc0_dev_std']  = acc0_dev_std
nested_CV_results['acc1_dev_mean']  = acc1_dev_mean
nested_CV_results['acc1_dev_std']  = acc1_dev_std
nested_CV_results['fscore_dev_mean']  = fscore_dev_mean
nested_CV_results['fscore_dev_std']  = fscore_dev_std
nested_CV_results['loss_score1']  = loss_score1
nested_CV_results['loss_score2']  = loss_score2
nested_CV_results['TotalRank']  = TotalRank

nested_CV_results = nested_CV_results.sort_values(by='TotalRank', ascending=False)

f_output=outDir+ PatCond + '_fold' + str(manualFold)+'_CNN_EvMetrics_CV.txt'
nested_CV_results.to_csv(f_output, index=None, sep='\t')





