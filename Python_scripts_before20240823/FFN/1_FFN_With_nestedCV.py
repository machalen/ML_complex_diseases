#!/usr/bin/env python3

"""
#Magda Arnal
#30/01/2023
#Run the FFN with nested CV 
"""

### import libraries
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import fbeta_score

#Import the class object
from FFN_classes import FFN_4hidden
from FFN_classes import PlotPerformance

# for number-crunching
import numpy as np

# for dataset management
import pandas as pd

#For system
import os
from optparse import OptionParser
import time

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

# use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#########################################################################
##########################Load data######################################
#Load matrix with variants
df = pd.read_csv(inMtrx, sep="\t")
#df.head()
print(df.shape)

#Load labels
labels = pd.read_csv(inLab, sep="\t")
#labels.head()
print(labels.shape)

###############################################################################
###############################################################################
#Select rows for case and controls
rows = labels.cond.values
casei = np.where(rows==PatCond)[0]
cntrli  = np.where(rows=='control')[0]
CaseNumber = len(casei)
CntrlNumber = len(cntrli)

print('There are ' + str(CaseNumber)+ ' cases in total')
print('There are ' + str(CntrlNumber)+ ' controls in total')

################################################################################
################################################################################
#Convert to numeric matrices and vectors with numpy and shuffle
X=df.to_numpy()
print(X.shape)
PredNum = X.shape[1]
con1=labels.iloc[:,1]
y=np.where(con1==PatCond, 1, con1)
y=np.where(y=='control', 0, y)
y=y.astype('int')
X, y = shuffle(X, y, random_state=1)

################################################################################
################################################################################
#Outer CV, select the fold in the outer layer of cross-validation
outer_strat = StratifiedKFold(n_splits = 5)#From 0 to 4

for fold, (a,b) in enumerate(outer_strat.split(X, y)):
    #print(fold)
    if fold == manualFold :
        train_dev_index=a
        test_index=b

X_train_dev=X[train_dev_index]
y_train_dev=y[train_dev_index]
X_test=X[test_index]
y_test=y[test_index]

################################################################################
################################################################################
#Check the split

print ('Train/Dev set:', X_train_dev.shape,  y_train_dev.shape)
unique_values, counts = np.unique(y_train_dev, return_counts=True)
print ('Train/Dev set has: ' + str(counts[0]) +' '+ str(unique_values[0])+ ' and ' + str(counts[1]) + ' '+ str(unique_values[1]))

print ('Test set:', X_test.shape,  y_test.shape)
unique_values, counts = np.unique(y_test, return_counts=True)
print ('Test set has: ' + str(counts[0]) +' '+ str(unique_values[0])+ ' and ' + str(counts[1]) + ' '+ str(unique_values[1]))


#Convert the test set to tensor
X_test  = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()
# we'll actually need the labels to be a "matrix"
y_test = y_test[:,None]
#Send to the GPU the X set because it will be always there
X_test=X_test.to(device)

###################################################################################
###########################DEFINE THE FUNCTION#####################################
def trainTheModel(numEp,learningRate, selfdr,nUnits,nLayers, Sampling, Balance):
    
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
        # Calculate the target number of samples for the minority class
        minority_class_size = int(0.20 * len(y_train_dev[y_train_dev == 1]))
        # Set the sampling strategy to oversample the minority class
        sampl=(minority_class_size + len(y_train_dev[y_train_dev == 1]))/len(y_train_dev[y_train_dev == 0])
        sm = SMOTE(sampling_strategy=sampl, random_state=0)
    elif Sampling == 'SMOTE_ENN' :
        # Calculate the target number of samples for the minority class
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
    for fold, (train_index, dev_index) in enumerate(strat.split(X_train_dev, y_train_dev)):

        x_train = X_train_dev[train_index]
        y_train = y_train_dev[train_index]
        x_dev = X_train_dev[dev_index]
        y_dev = y_train_dev[dev_index]

        #############################################################################
        if Sampling == 'random' :
            X_train, Y_train = rus.fit_resample(x_train, y_train)
            # unique, counts = np.unique(Y_train, return_counts=True)
            # print('Counts after undersampling:',np.asarray((unique, counts)).T)
        elif Sampling == 'ENN' :
            X_resampled, y_resampled = enn.fit_resample(x_train, y_train)
            # unique, counts = np.unique(y_resampled, return_counts=True)
            # print('Counts after undersampling:',np.asarray((unique, counts)).T)
            X_train, Y_train = rus.fit_resample(X_resampled, y_resampled)#Set the proportion of the balancing
            # unique, counts = np.unique(Y_train, return_counts=True)
            # print('Counts after undersampling:',np.asarray((unique, counts)).T)
        elif Sampling == 'SMOTE_random' :
            X_oversampled, y_oversampled = sm.fit_resample(x_train, y_train)
            # unique, counts = np.unique(y_oversampled, return_counts=True)
            # print('Counts after oversampling:',np.asarray((unique, counts)).T)
            X_train, Y_train = rus.fit_resample(X_oversampled, y_oversampled)#Set the proportion of the balancing
            # unique, counts = np.unique(Y_train, return_counts=True)
            # print('Counts after undersampling:',np.asarray((unique, counts)).T)
        elif Sampling == 'SMOTE_ENN' :
            X_oversampled, y_oversampled = sm.fit_resample(x_train, y_train)
            # unique, counts = np.unique(y_oversampled, return_counts=True)
            # print('Counts after oversampling:',np.asarray((unique, counts)).T)
            X_resampled, y_resampled = enn.fit_resample(X_oversampled, y_oversampled)
            # unique, counts = np.unique(y_resampled, return_counts=True)
            # print('Counts after undersampling:',np.asarray((unique, counts)).T)
            X_train, Y_train = rus.fit_resample(X_resampled, y_resampled)#Set the proportion of the balancing
            # unique, counts = np.unique(Y_train, return_counts=True)
            # print('Counts after undersampling:',np.asarray((unique, counts)).T)
        
        #################################
        #Convert to tensors to input to the FFN
        X_train  = torch.tensor(X_train).float()
        Y_train = torch.tensor(Y_train).float()
        x_dev  = torch.tensor(x_dev).float()
        y_dev = torch.tensor(y_dev).float()
        
        # we'll actually need the labels to be a "matrix"
        Y_train = Y_train[:,None]
        y_dev = y_dev[:,None]
        
        ##################################
        #Send the data to the GPU
        X_train=X_train.to(device)
        x_dev=x_dev.to(device)
        
        # create a new model from FFN_classes.py
        net,lossfun,optimizer = FFN_4hidden(learningRate, selfdr,nUnits,nLayers,inUnits=PredNum)
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
# numEpV=[200,300,500]#Number of epochs
# learningRateV=[0.0001,0.001, 0.01]#learning rate
# selfdrV=[0.1,0.2,0.4]#dropout
# nUnitsV=[100,200]#number of units in the inner layers of the FFN
# nLayersV=[1,2,3]#Number of inner layers in the FFN
# BalanceV=[50,70]#Balancing ratio (% of controls over the total)
# SamplingV=['random','ENN', 'SMOTE_random','SMOTE_ENN']
#random: Random undersampling of controls.
#ENN: Smart undersampling of controls. Undersample based on the edited nearest neighbour method.
#SMOTE_random: Oversampling with SMOTE creating 20% of cases and undersampling controls with random undersampling.
#SMOTE_ENN: Oversampling with SMOTE creating 20% of cases and undersampling controls with the edited nearest neighbour method.

numEpV=[200]
learningRateV=[0.01]
selfdrV=[0.1,0.2]
nUnitsV=[100]
nLayersV=[2]
BalanceV=[50]
SamplingV=['random']

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
timerInFunction = time.process_time()

#Run the expirement
for numEp in numEpV:
    for learningRate in learningRateV:
        for selfdr in selfdrV:
            for nUnits in nUnitsV:
                for nLayers in nLayersV:
                    for Balance in BalanceV:
                        for Sampling in SamplingV:
                            trainAcc,devAcc, devAcc0, devAcc1, devFscore, losses, net = trainTheModel(numEp=numEp,learningRate=learningRate, selfdr=selfdr,nUnits=nUnits,nLayers=nLayers, Sampling=Sampling, Balance=Balance)
                            
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
                            
                            ##Append scores in order to evaluate the model
                            s1=(a0+a1)/2
                            s2=abs(a0 - a1)
                            s3=s1-s2
                            loss_score1.append(s1)
                            loss_score2.append(s2)
                            TotalRank.append(s3)
                            
                            #Plot only the best configurations
                            if (a0 > 0.61 and a1 > 0.61):
                                outpath=outDir+'FFN_SeveralParams'+'_fold' + str(manualFold)+'_numEp'+ str(numEp) +'_learningRate'+ str(learningRate)+'_selfdr'+str(selfdr)+'_nUnits'+str(nUnits)+'_nLayers' +str(nLayers)+ '_balancing' + str(Balance)+'_sampling' + str(Sampling) +'.pdf'
                                PlotPerformance(trainAcc,devAcc, devAcc0, devAcc1, devFscore, losses, outpath)

#Print the time required for the calculations.
print((time.process_time() - timerInFunction)/60)

#Save the full results
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

f_output=outDir+ PatCond + '_fold' + str(manualFold)+'_FFN_EvMetrics_CV.txt'
nested_CV_results.to_csv(f_output, index=None, sep='\t')




