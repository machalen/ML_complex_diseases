#!/usr/bin/env python3

"""
#Magda Arnal
#30/01/2023
#Run the CNN final model and apply ExAI methods. 
"""

### import libraries
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import fbeta_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
#For system
import sys
import os

# Add the directory containing the lib folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#Import the class object and other related functions
from lib.FFN_classes import FFN_4hidden
from lib.FFN_classes import PlotPerformance
from lib.FFN_classes import ANNclassWithSigmoid

#For explainability
from captum.attr import LayerIntegratedGradients
from captum.attr import LayerDeepLift
from captum.attr import Saliency
from captum.attr import GuidedBackprop

# for number-crunching
import numpy as np

# for dataset management
import pandas as pd

#For input
from optparse import OptionParser

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
parser.add_option('-e', '--numEp',
                  help='HyperParameter: Number of epochs.',
                  metavar=500)
parser.add_option('-r', '--learningRate',
                  help='HyperParameter: learning_rate.',
                  metavar=0.001)
parser.add_option('-d', '--selfdr',
                  help='HyperParameter: dropout rate.',
                  metavar=0.2)
parser.add_option('-u', '--nUnits',
                  help='HyperParameter: number of units (width).',
                  metavar=100)
parser.add_option('-y', '--nLayers',
                  help='HyperParameter: number of layers (depth).',
                  metavar=2)
parser.add_option('-b', '--Balance',
                  help='Balancing rate.',
                  metavar=50)
parser.add_option('-s', '--Undersampling',
                  help='Sampling strategy.',
                  metavar='random')

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
numEp = int(option_dict['numEp'])
learningRate = float(option_dict['learningRate'])
selfdr = float(option_dict['selfdr'])
nUnits = int(option_dict['nUnits'])
nLayers = int(option_dict['nLayers'])
balance=int(option_dict['Balance'])
sampl_strategy=option_dict['Undersampling']

# use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#########################################################################
##########################Load data######################################
#Load matrix with variants
df = pd.read_csv(inMtrx, sep="\t")
PredNum = df.shape[1]

#Load labels
labels = pd.read_csv(inLab, sep="\t")

#Get the name of the files to output them in results:
inMtrxName=os.path.basename(inMtrx) + '//' + os.path.basename(inLab)

#########################################################################
#########################################################################
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

################################################################################
##############################Make nested CV####################################
#Outer CV, select the fold in the outer layer of cross-validation
outer_strat = StratifiedKFold(shuffle=False, n_splits = 5)#From 0 to 4

for fold, (a,b) in enumerate(outer_strat.split(Xtab, y)):
    #print(fold)
    if fold == manualFold :
        train_dev_index=a
        test_index=b

X_train_dev=Xtab.iloc[train_dev_index,:]
y_train_dev=y[train_dev_index]
X_tetab=Xtab.iloc[test_index,:]
y_test=y[test_index]

unique_values, counts = np.unique(y_train_dev, return_counts=True)
print ('Train/Dev set has: ' + str(counts[0]) +' '+ str(unique_values[0])+ ' and ' + str(counts[1]) + ' '+ str(unique_values[1]))

unique_values, counts = np.unique(y_test, return_counts=True)
print ('Test set has: ' + str(counts[0]) +' '+ str(unique_values[0])+ ' and ' + str(counts[1]) + ' '+ str(unique_values[1]))

#################################################################################
#################################################################################
# #Check that the cases and controls are properly assigned after cross fold split
# con2=X_train_dev.iloc[:,1]
# ytr=np.where(con2==PatCond, 1, con2)
# ytr=np.where(ytr=='control', 0, ytr)
# ytr=ytr.astype('int')
# print(np.array_equal(y_train_dev, ytr))#TRUE!

# con2=X_tetab.iloc[:,1]
# yte=np.where(con2==PatCond, 1, con2)
# yte=np.where(yte=='control', 0, yte)
# yte=yte.astype('int')
# print(np.array_equal(y_test, yte))#TRUE!

###################################################################################
###################################################################################
def trainTheModel(numEp,learningRate, selfdr,nUnits,nLayers,sampl_strategy,balance):
    
    #Apply cross validation (inner loop)
    n_splits=10
    strat = StratifiedKFold(n_splits = n_splits)

    #################################################################################
    #Define the function to apply the sampling
    balance_rate=(100-balance)/balance
    rus = RandomUnderSampler(random_state=0, sampling_strategy= balance_rate)
    
    if sampl_strategy == 'ENN' :
        enn = EditedNearestNeighbours(sampling_strategy='majority')
    elif sampl_strategy == 'SMOTE_random' :
        # Calculate the number of samples for the minority class
        minority_class_size = int(0.20 * len(y_train_dev[y_train_dev == 1]))
        # Set the sampling strategy to oversample the minority class
        sampl=(minority_class_size + len(y_train_dev[y_train_dev == 1]))/len(y_train_dev[y_train_dev == 0])
        sm = SMOTE(sampling_strategy=sampl, random_state=0)
    elif sampl_strategy == 'SMOTE_ENN' :
        # Calculate the number of samples for the minority class
        minority_class_size = int(0.20 * len(y_train_dev[y_train_dev == 1]))
        # Set the sampling strategy to oversample the minority class
        sampl=(minority_class_size + len(y_train_dev[y_train_dev == 1]))/len(y_train_dev[y_train_dev == 0])
        sm = SMOTE(sampling_strategy=sampl, random_state=0)
        #ENN
        enn = EditedNearestNeighbours(sampling_strategy='majority')
    
    #Initialize the pandas data.frame with train samples
    train_samples = pd.DataFrame(columns=labels.columns)
    
    # initialize evaluation metrics
    losses   = np.zeros((numEp,n_splits))
    trainAcc = np.zeros((numEp,n_splits))
    devAcc  = np.zeros((numEp,n_splits))
    devAcc0  = np.zeros((numEp,n_splits))
    devAcc1  = np.zeros((numEp,n_splits))
    devFscore = np.zeros((numEp,n_splits))

    ###############################################################
    #Loop over CV
    for fold, (train_index, dev_index) in enumerate(strat.split(X_train_dev, y_train_dev)):

        X_train = X_train_dev.iloc[train_index,:]
        Y_train = y_train_dev[train_index]
        x_devtab = X_train_dev.iloc[dev_index,:]
        y_dev = y_train_dev[dev_index]
        
        ###################################
        if sampl_strategy == 'random' :
            #Apply random undersampling
            x_trtab, y_train = rus.fit_resample(X_train, Y_train)
        elif sampl_strategy == 'ENN' :
            #Convert the X_trtab to numeric for ENN method:
            x_num=X_train.iloc[:,labels.shape[1]:len(X_train.columns)]
            x_num=x_num.to_numpy()
            #Apply ENN
            _, y_resampled = enn.fit_resample(x_num, Y_train)
            subsampled_indices = enn.sample_indices_
            X_resampled=X_train.iloc[subsampled_indices]
            #Apply random undersampling
            x_trtab, y_train = rus.fit_resample(X_resampled, y_resampled)
        elif sampl_strategy == 'SMOTE_random' :
            #Convert the X_trtab to numeric for SMOTE method:
            x_num=X_train.iloc[:,labels.shape[1]:len(X_train.columns)]
            x_num=x_num.to_numpy()
            #Apply SMOTE
            X_oversampled, y_oversampled = sm.fit_resample(x_num, Y_train)
            #Apply random undersampling
            x_trtab, y_train = rus.fit_resample(X_oversampled, y_oversampled)
        elif sampl_strategy == 'SMOTE_ENN' :
            #Convert the X_trtab to numeric for SMOTE and ENN method:
            x_num=X_train.iloc[:,labels.shape[1]:len(X_train.columns)]
            x_num=x_num.to_numpy()
            #Apply SMOTE
            X_oversampled, y_oversampled = sm.fit_resample(x_num, Y_train)
            #Apply ENN
            X_resampled, y_resampled = enn.fit_resample(X_oversampled, y_oversampled)
            #Apply random undersampling
            x_trtab, y_train = rus.fit_resample(X_resampled, y_resampled)
        ############################################################################
        if sampl_strategy in ['random','ENN'] :
            #Save the labels of training samples
            x_trlab=x_trtab.iloc[:,0:labels.shape[1]]
            x_trlab.columns=labels.columns
            train_samples = pd.concat([train_samples, x_trlab], ignore_index=True)
            #Split and create the numeric np matrix
            x_train=x_trtab.iloc[:,labels.shape[1]:len(x_trtab.columns)]
            x_train=x_train.to_numpy()
        else:
            x_train=x_trtab
        
        #I don't do undersampling to the test set!
        #Save the labels
        x_devlab=x_devtab.iloc[:,0:labels.shape[1]]
        x_devlab.columns=labels.columns
        #Split and create the numeric np matrix
        x_dev=x_devtab.iloc[:,labels.shape[1]:len(x_devtab.columns)]
        x_dev=x_dev.to_numpy()
        
        ###################################
        #Convert to tensors to input to the FFN
        x_train=x_train.astype(int)
        x_train  = torch.tensor(x_train).float()
        y_train = torch.tensor(y_train).float()
        x_dev=x_dev.astype(int)
        x_dev  = torch.tensor(x_dev).float()
        y_dev = torch.tensor(y_dev).float()
        
        # we'll actually need the labels to be a "matrix"
        y_train = y_train[:,None]
        y_dev = y_dev[:,None]

        ##################################
        #Send the data to the GPU
        x_train=x_train.to(device)
        x_dev=x_dev.to(device)
        
        # create a new model from FFN_classes.py
        net,lossfun,optimizer = FFN_4hidden(learningRate, selfdr,nUnits,nLayers,inUnits=PredNum)
        #Send the model to the GPU
        net.to(device)

        # loop over epochs
        for epochi in range(numEp):
            # loop over training data batches
            net.train() # switch to train mode
            
            #Send the data to the GPU at the beginning of the epoch
            y_train=y_train.to(device)

            # forward pass and loss
            yHat = net(x_train)
            loss = lossfun(yHat,y_train)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_train=y_train.cpu()
            yHat=yHat.cpu()

            # now that we've trained through the batches, get their average training accuracy
            yHat_pre=(yHat>0).float()#Translate continuous to 0/1
            trainAcc[epochi,fold] = torch.mean((yHat_pre == y_train).float()).item()
            
            # and get average losses across the batches
            losses[epochi,fold] = loss.item()
            
            # test accuracy
            net.eval() # switch to test mode
            with torch.no_grad(): # deactivates autograd
                yHat = net(x_dev)
            
            yHat=yHat.cpu()
            yHat_pre=(yHat>0).float()#Translate continuous to 0/1
            devAcc[epochi,fold]=torch.mean((yHat_pre == y_dev).float()).item()
            devAcc0[epochi,fold]= torch.mean((yHat_pre[y_dev == 0] == y_dev[y_dev == 0]).float()).item()
            devAcc1[epochi,fold] = torch.mean((yHat_pre[y_dev == 1] == y_dev[y_dev == 1]).float()).item()
            devFscore[epochi,fold]=fbeta_score(y_dev, yHat_pre, average='binary', beta=1)
    
    #Save train samples
    if sampl_strategy in ['random','ENN'] :
        f1_output=outDir+'Fold'+str(manualFold)+'_FFN_TrainingIDs.txt'
        train_samples.to_csv(f1_output, index=None, sep='\t')
    # function output
    return trainAcc, devAcc, devAcc0, devAcc1, devFscore, losses, net

##############################################################################################
###################################EXPERIMENT#################################################
#Train the net
trainAcc,devAcc, devAcc0, devAcc1, devFscore, losses, net = trainTheModel(numEp=numEp,learningRate=learningRate, selfdr=selfdr,nUnits=nUnits,nLayers=nLayers,sampl_strategy=sampl_strategy, balance=balance)

#save the model
net.cpu()
torch.save(net.state_dict(), outDir + 'Trained_FFN_'+str(manualFold)+'_net.pt')

#Save the hyperParameter configuration
grid_name='numEp'+ str(numEp) +'_learningRate'+ str(learningRate)+'_selfdr'+str(selfdr)+'_nUnits'+str(nUnits)+'_nLayers' +str(nLayers)+ '_balancing' + str(balance) + '_sampling' + str(sampl_strategy)

outpath=outDir+'FFN_SeveralParams'+'_fold' + str(manualFold)+grid_name+'.pdf'
PlotPerformance(trainAcc,devAcc, devAcc0, devAcc1, devFscore, losses, outpath)

########################################################################
#############################TEST#######################################
#Save the labels
x_telab=X_tetab.iloc[:,0:labels.shape[1]]
x_telab.columns=labels.columns
#Split and create the numeric np matrix
x_test=X_tetab.iloc[:,labels.shape[1]:len(X_tetab.columns)]
x_test=x_test.to_numpy()

#Convert the test set to tensor
x_test=x_test.astype(int)
x_test  = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()
# we'll actually need the labels to be a "matrix"
y_test = y_test[:,None]
#Send to the GPU the X set because it will be always there
x_test=x_test.to(device)

###################################################################################
# #Test with the original trained model
# #Run the model
# net.to(device)
# net.eval()
# torch.manual_seed(1)
# yHat = net(x_test)
# #Bring back
# yHat=yHat.cpu()
                    
# #Calculate the evaluation metrics
# yHat_pre=(yHat>0).float()#Translate continuous to 0/1
# m_fscore = fbeta_score(y_test, yHat_pre, average='binary', beta=1)

# yHat_prob=torch.sigmoid(yHat).detach()#probabilities for the roc_auc
# m_roc = roc_auc_score(y_test, yHat_prob)

# #Calculate PPV and NPV
# c_m=metrics.confusion_matrix(y_test, yHat_pre)
# TN=c_m[0,0]
# TP=c_m[1,1]
# FN=c_m[1,0]
# FP=c_m[0,1]

# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP)
# # Precision or positive predictive value
# PPV = TP/(TP+FP)
# # Negative predictive value
# NPV = TN/(TN+FN)
# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)

# print('Accuracy is :',ACC)
# print('Sensitivity is:',TPR)
# print('Specificity is:',TNR)
# print('Positive Predicted Value is:',PPV)
# print('Negative Predicted Value is:',NPV)

# #Build the dataframe with the predictions
# x_telab['y_pred']=yHat_pre.numpy()
# x_telab['Prob']=yHat_prob.numpy()
###################################################################################
#Test with the saved model

#Load the model
net_a, _, _  = FFN_4hidden(learningRate, selfdr,nUnits,nLayers,inUnits=PredNum)
net_a.load_state_dict(torch.load(outDir + 'Trained_FFN_'+str(manualFold)+'_net.pt'))

#Run the model
net_a.to(device)
net_a.eval()
torch.manual_seed(1)
yHat = net_a(x_test)
#Bring back
yHat=yHat.cpu()
                    
#Calculate the evaluation metrics
yHat_pre=(yHat>0).float()#Translate continuous to 0/1
m_fscore = fbeta_score(y_test, yHat_pre, average='binary', beta=1)

yHat_prob=torch.sigmoid(yHat).detach()#probabilities for the roc_auc
m_roc = roc_auc_score(y_test, yHat_prob)

c_m=metrics.confusion_matrix(y_test, yHat_pre)
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

print('Accuracy is :',ACC)
print('Sensitivity is:',TPR)
print('Specificity is:',TNR)
print('Positive Predicted Value is:',PPV)
print('Negative Predicted Value is:',NPV)

#Build the dataframe with the predictions
x_telab['y_pred']=yHat_pre.numpy()
x_telab['Prob1']=yHat_prob.numpy()

############################Output results##################################

f3_output=outDir+'Fold'+str(manualFold)+'_FFN_Samples.txt'
x_telab.to_csv(f3_output, index=None, sep='\t')

#Path to save the final results of the evaluation metrics for the test set.
GlobalResults=outDir+'TestFinalResults.txt'

# Check if the file exists
if not os.path.exists(GlobalResults):
    header = "condition\tmodel\tinput_matrix\tn_predictors\tn_controls\tn_cases\thyperparameters\tfold\taccuracy\tspecificity\tsensitivity\trocauc\tfscore\tPPV\tNPV\n"
    # If the file doesn't exist, create it and write the header
    with open(GlobalResults, 'w') as f:
        f.write(header)

#Convert parameters to strings
with open(GlobalResults, 'a+') as f:
    f.write(PatCond +'\t'+'FFN'+'\t'+ inMtrxName +
    '\t' + str(PredNum) +
    '\t' + str(CntrlNumber) +
    '\t' + str(CaseNumber) +
    '\t' + 'Hiper:' + grid_name +
    '\t' + 'Fold' + str(manualFold) +
    '\t' + str(round(ACC,4)) +
    '\t' + str(round(TNR,4)) +
    '\t' + str(round(TPR,4)) +
    '\t' + str(round(m_roc,4)) +
    '\t' + str(round(m_fscore,4)) +
    '\t' + str(round(PPV,4)) +
    '\t' + str(round(NPV,4)) + '\n')
    f.close()

###########################################################################
########################Apply Explainability methods#######################
#I have to change the output of the FFN to produce a probability value with the sigmoid function
net_a.cpu()
net_b = ANNclassWithSigmoid(net_a.dr, nUnits, net_a.nLayers, net_a.layers['input'].in_features)
net_b.load_state_dict(net_a.state_dict())

################################################################################################
#Layer Integrated Gradients
net_b.cpu()
x_test=x_test.cpu()

print('LayerIntegratedGradients')
lig = LayerIntegratedGradients(net_b, net_b.layers['input'])

# Compute the attribution scores
attr_lig, delta_lig = lig.attribute(x_test, baselines=x_test * 0, attribute_to_layer_input=True, return_convergence_delta=True)

#Convert attributes to numpy and save
attr_lig_np = attr_lig.detach().numpy()
np.savetxt(outDir + 'Attributes_FFN_'+str(manualFold)+'_lig.txt', attr_lig_np, delimiter='\t')

#Convert delta to numpy and save
delta_lig_np=delta_lig.numpy()
np.savetxt(outDir + 'Delta_FFN_'+str(manualFold)+'_lig.txt', delta_lig_np, delimiter='\t')

###############################################################################################
###############################################################################################
#Layer DeepLift

print('LayerDeepLift')
dl = LayerDeepLift(net_b, net_b.layers['input'])

# Compute the attribution scores
attr_dl, delta_dl = dl.attribute(x_test, baselines=x_test * 0, attribute_to_layer_input=True, return_convergence_delta=True)

#Convert attributes to numpy and save
attr_dl_np = attr_dl.detach().numpy()
np.savetxt(outDir + 'Attributes_FFN_'+str(manualFold)+'_dl.txt', attr_dl_np, delimiter='\t')

#Convert delta to numpy and save
delta_dl_np=delta_dl.numpy()
np.savetxt(outDir + 'Delta_FFN_'+str(manualFold)+'_dl.txt', delta_dl_np, delimiter='\t')

################################################################################################
################################################################################################
#Saliency Maps

print('Saliency')
sl = Saliency(net_b)

# Compute the attribution scores
attr_sl = sl.attribute(x_test)

#Convert attributes to numpy
attr_sl_np = attr_sl.detach().numpy()
#Save attributes
np.savetxt(outDir + 'Attributes_FFN_'+str(manualFold)+'_sl.txt', attr_sl_np, delimiter='\t')

##############################################################################################
##############################################################################################
#Guided backpropagation

# Warning: Ensure that all ReLU operations in the forward function of the given model are performed 
# using a module (nn.module.ReLU). If nn.functional.ReLU is used, gradients are not overridden appropriately.

print('GuideBackprop')
gbp = GuidedBackprop(net_b)

# Compute the attribution scores
attr_gbp = gbp.attribute(x_test)

#Convert attributes to numpy
attr_gbp_np = attr_gbp.detach().numpy()
#Save attributes
np.savetxt(outDir + 'Attributes_FFN_'+str(manualFold)+'_gbp.txt', attr_gbp_np, delimiter='\t')

