#!/usr/bin/env python3

"""
#Magda Arnal
#24/05/2021
#Functions to make cross validation on training and validation sets, and save the results 
"""

### import libraries
#ML and evaluation metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

#For sampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RandomUnderSampler

#For data management
import numpy as np
import pandas as pd
#Read configuration files
import configparser

############################################################################
############################################################################
def Cross_Val_Groups(model, X, y, combination, n_splits = 10):

    #Initialize lists to store the evaluation metrics
    f_score_train = []
    f_score_val = []

    prec_train = []
    prec_val = []
    
    acc_train = []
    acc_val = []

    acc0_train = []
    acc0_val = []

    acc1_train = []
    acc1_val = []
    
    recall_train = []
    recall_val = []
    
    roc_train = []
    roc_val = []

    pr_train = []
    pr_val = []
    
    #Define the functions for sampling
    balance_rate=(100-combination['Balance'])/combination['Balance']
    #Random undersampling
    rus = RandomUnderSampler(random_state=0, sampling_strategy= balance_rate)

    if combination['Sampling'] == 'ENN' :
        #Undersampling with ENN
        enn = EditedNearestNeighbours(sampling_strategy='majority')
    elif combination['Sampling'] == 'SMOTE_random' :
        # Calculate the number of samples for the minority class
        minority_class_size = int(0.20 * len(y[y == 1]))
        sampl=(minority_class_size + len(y[y == 1]))/len(y[y == 0])
        #Oversampling with SMOTE
        sm = SMOTE(sampling_strategy=sampl, random_state=0)
    elif combination['Sampling'] == 'SMOTE_ENN' :
        # Calculate the number of samples for the minority class
        minority_class_size = int(0.20 * len(y[y == 1]))
        sampl=(minority_class_size + len(y[y == 1]))/len(y[y == 0])
        #Oversampling with SMOTE
        sm = SMOTE(sampling_strategy=sampl, random_state=0)
        #Undersampling with ENN
        enn = EditedNearestNeighbours(sampling_strategy='majority')
    
    np.random.seed(0)
    strat = StratifiedKFold(n_splits = n_splits, shuffle=False)
    
    for train_index, val_index in strat.split(X, y):
        
        x_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        Y_val = y[val_index]
        
        #Sampling
        if combination['Sampling'] == 'random' :
            X_train, Y_train = rus.fit_resample(x_train, y_train)
        elif combination['Sampling'] == 'ENN' :
            X_resampled, y_resampled = enn.fit_resample(x_train, y_train)
            X_train, Y_train = rus.fit_resample(X_resampled, y_resampled)
        elif combination['Sampling'] == 'SMOTE_random' :
            X_oversampled, y_oversampled = sm.fit_resample(x_train, y_train)
            X_train, Y_train = rus.fit_resample(X_oversampled, y_oversampled)
        elif combination['Sampling'] == 'SMOTE_ENN' :
            X_oversampled, y_oversampled = sm.fit_resample(x_train, y_train)
            X_resampled, y_resampled = enn.fit_resample(X_oversampled, y_oversampled)
            X_train, Y_train = rus.fit_resample(X_resampled, y_resampled)
        
        #Set the model
        if type(model).__name__ == 'LogisticRegression':
            modSet = model.set_params(solver=combination['solver'],
                                   C=combination['Creg'],
                                   random_state=0)
        if type(model).__name__ == 'ExtraTreesClassifier' or type(model).__name__ == 'RandomForestClassifier':
            modSet = model.set_params(n_estimators=combination['n_estimators'],
                                   min_samples_split=combination['min_samples_split'],
                                   min_samples_leaf= combination['min_samples_leaf'], 
                                   max_depth=combination['max_depth'],
                                   random_state=0)
        if type(model).__name__ == 'GradientBoostingClassifier':
            modSet = model.set_params(n_estimators=combination['n_estimators'],
                                   learning_rate=combination['learning_rate'],
                                   subsample= combination['subsample'], 
                                   max_depth=combination['max_depth'],
                                   loss=combination['loss'],
                                   random_state=0)
        
        #Fit the model
        modSet.fit(X_train, Y_train)
        
        #Prediction and evaluation on the training set
        pred_train = modSet.predict(X_train)
        score = precision_recall_fscore_support(Y_train, pred_train, average='binary')
        prec_train.append(score[0])
        recall_train.append(score[1])
        f_score_train.append(f1_score(Y_train, pred_train, average='binary'))
        acc = accuracy_score(Y_train, pred_train)
        acc_train.append(acc)
        acc0 = accuracy_score(Y_train[Y_train == 0], pred_train[Y_train == 0])
        acc0_train.append(acc0)
        acc1 = accuracy_score(Y_train[Y_train == 1], pred_train[Y_train == 1])
        acc1_train.append(acc1)
        #Calculate the roc curve
        mpred = modSet.predict_proba(X_train)
        pred=mpred[:,1]
        fpr, tpr, thresholds = roc_curve(Y_train, pred, pos_label=1)
        roc_train.append(auc(fpr, tpr))
        #Calculate the precision recall curve
        precision, recall, thresholds = precision_recall_curve(Y_train, pred)
        pr_train.append(auc(recall, precision))
        
        #Prediction and evaluation on the validation set
        pred_val = modSet.predict(X_val)
        score = precision_recall_fscore_support(Y_val, pred_val, average='binary')
        prec_val.append(score[0])
        recall_val.append(score[1])
        f_score_val.append(f1_score(Y_val, pred_val, average='binary'))     
        acc = accuracy_score(Y_val, pred_val)
        acc_val.append(acc)
        acc0 = accuracy_score(Y_val[Y_val == 0], pred_val[Y_val == 0])
        acc0_val.append(acc0)
        acc1 = accuracy_score(Y_val[Y_val == 1], pred_val[Y_val == 1])
        acc1_val.append(acc1)
        #Calculate the roc curve
        mpred = modSet.predict_proba(X_val)
        pred=mpred[:,1]
        fpr, tpr, thresholds = roc_curve(Y_val, pred, pos_label=1)
        roc_val.append(auc(fpr, tpr))
        #Calculate the precision recall curve
        precision, recall, thresholds = precision_recall_curve(Y_val, pred)
        pr_val.append(auc(recall, precision))
    
    #Save the mean and sd of the evaluation metrics across the folds in the inner loop
    mean_prec_train = np.mean(np.array(prec_train))
    std_prec_train = np.std(np.array(prec_train))
    mean_prec_val = np.mean(np.array(prec_val))
    std_prec_val = np.std(np.array(prec_val))
    prec_metrics={'mean_prec_train': mean_prec_train, 
                  'std_prec_train': std_prec_train,
                  'mean_prec_val': mean_prec_val, 
                  'std_prec_val': std_prec_val}
    
    mean_recall_train = np.mean(np.array(recall_train))
    std_recall_train = np.std(np.array(recall_train))
    mean_recall_val = np.mean(np.array(recall_val))
    std_recall_val = np.std(np.array(recall_val))
    recall_metrics={'mean_recall_train': mean_recall_train, 
                  'std_recall_train': std_recall_train,
                  'mean_recall_val': mean_recall_val, 
                  'std_recall_val': std_recall_val}
    
    mean_f_score_train = np.mean(np.array(f_score_train))
    std_f_score_train = np.std(np.array(f_score_train))
    mean_f_score_val = np.mean(np.array(f_score_val))
    std_f_score_val = np.std(np.array(f_score_val))
    fscore_metrics={'mean_fscore_train': mean_f_score_train, 
                  'std_fscore_train': std_f_score_train,
                  'mean_fscore_val': mean_f_score_val, 
                  'std_fscore_val': std_f_score_val}
    
    mean_acc_train = np.mean(np.array(acc_train))
    std_acc_train = np.std(np.array(acc_train))
    mean_acc_val = np.mean(np.array(acc_val))
    std_acc_val = np.std(np.array(acc_val))
    acc_metrics={'mean_acc_train': mean_acc_train, 
                  'std_acc_train': std_acc_train,
                  'mean_acc_val': mean_acc_val, 
                  'std_acc_val': std_acc_val}
    
    mean_acc0_train = np.mean(np.array(acc0_train))
    std_acc0_train = np.std(np.array(acc0_train))
    mean_acc0_val = np.mean(np.array(acc0_val))
    std_acc0_val = np.std(np.array(acc0_val))
    acc0_metrics={'mean_acc0_train': mean_acc0_train, 
                  'std_acc0_train': std_acc0_train,
                  'mean_acc0_val': mean_acc0_val, 
                  'std_acc0_val': std_acc0_val}
    
    mean_acc1_train = np.mean(np.array(acc1_train))
    std_acc1_train = np.std(np.array(acc1_train))
    mean_acc1_val = np.mean(np.array(acc1_val))
    std_acc1_val = np.std(np.array(acc1_val))
    acc1_metrics={'mean_acc1_train': mean_acc1_train, 
                  'std_acc1_train': std_acc1_train,
                  'mean_acc1_val': mean_acc1_val, 
                  'std_acc1_val': std_acc1_val}
    
    mean_roc_train = np.mean(np.array(roc_train))
    std_roc_train = np.std(np.array(roc_train))
    mean_roc_val = np.mean(np.array(roc_val))
    std_roc_val = np.std(np.array(roc_val))
    roc_metrics={'mean_roc_train': mean_roc_train, 
                  'std_roc_train': std_roc_train,
                  'mean_roc_val': mean_roc_val, 
                  'std_roc_val': std_roc_val}
    
    mean_pr_train = np.mean(np.array(pr_train))
    std_pr_train = np.std(np.array(pr_train))
    mean_pr_val = np.mean(np.array(pr_val))
    std_pr_val = np.std(np.array(pr_val))
    pr_metrics={'mean_pr_train': mean_pr_train, 
                  'std_pr_train': std_pr_train,
                  'mean_pr_val': mean_pr_val, 
                  'std_pr_val': std_pr_val}
    
    return prec_metrics, recall_metrics, fscore_metrics, acc_metrics, acc0_metrics, acc1_metrics, roc_metrics, pr_metrics

#######################################################################################################
#######################################################################################################

def MakeEvalSum(mtrx_acc, mtrx_acc0, mtrx_acc1, mtrx_prec, mtrx_recall, mtrx_fscore, mtrx_roc, mtrx_pr):
    #First check if the three dictionaries have the same keys
    keys_list = list(mtrx_acc)
    # keys_list_acc0 = list(mtrx_acc0)
    # keys_list_acc1 = list(mtrx_acc1)
    # keys_list_prec = list(mtrx_prec)
    # keys_list_recall = list(mtrx_recall)
    # keys_list_fscore = list(mtrx_fscore)
    # keys_list_roc = list(mtrx_roc)
    # keys_list_pr = list(mtrx_pr)
    # print(keys_list == keys_list_acc0 == keys_list_acc1 == keys_list_prec == keys_list_recall == keys_list_fscore == keys_list_roc == keys_list_pr)#True
    
    #Define the lists that will be columns in the matrix
    hyperparam = []
    #For accuracy
    train_mean = []
    train_std = []
    val_mean = []
    val_std = []
    #For specificity
    train_mean_0 = []
    train_std_0 = []
    val_mean_0 = []
    val_std_0 = []
    #For sensitivity
    train_mean_1 = []
    train_std_1 = []
    val_mean_1 = []
    val_std_1 = []
    #For precision
    val_mean_prec = []
    val_std_prec = []
    #For recall
    val_mean_recall = []
    val_std_recall = []
    #For fscore
    val_mean_fscore = []
    val_std_fscore = []
    #For ROC-AUC
    val_mean_roc = []
    val_std_roc = []
    #For precision-recall
    val_mean_pr = []
    val_std_pr = []
    
    for i in range(len(keys_list)):
        #Save the hyperparameter combination
        hyperparam.append('|'.join(str(p) for p in keys_list[i]))
        #For accuracy
        val_mean.append(mtrx_acc[keys_list[i]]['mean_acc_val'])
        val_std.append(mtrx_acc[keys_list[i]]['std_acc_val'])
        train_mean.append(mtrx_acc[keys_list[i]]['mean_acc_train'])
        train_std.append(mtrx_acc[keys_list[i]]['std_acc_train'])
        #For specificity
        val_mean_0.append(mtrx_acc0[keys_list[i]]['mean_acc0_val'])
        val_std_0.append(mtrx_acc0[keys_list[i]]['std_acc0_val'])
        train_mean_0.append(mtrx_acc0[keys_list[i]]['mean_acc0_train'])
        train_std_0.append(mtrx_acc0[keys_list[i]]['std_acc0_train'])
        #For sensitivity
        val_mean_1.append(mtrx_acc1[keys_list[i]]['mean_acc1_val'])
        val_std_1.append(mtrx_acc1[keys_list[i]]['std_acc1_val'])
        train_mean_1.append(mtrx_acc1[keys_list[i]]['mean_acc1_train'])
        train_std_1.append(mtrx_acc1[keys_list[i]]['std_acc1_train'])
        #for precision
        val_mean_prec.append(mtrx_prec[keys_list[i]]['mean_prec_val'])
        val_std_prec.append(mtrx_prec[keys_list[i]]['std_prec_val'])
        #for recall
        val_mean_recall.append(mtrx_recall[keys_list[i]]['mean_recall_val'])
        val_std_recall.append(mtrx_recall[keys_list[i]]['std_recall_val'])
        #for fscore
        val_mean_fscore.append(mtrx_fscore[keys_list[i]]['mean_fscore_val'])
        val_std_fscore.append(mtrx_fscore[keys_list[i]]['std_fscore_val'])
        #for roc
        val_mean_roc.append(mtrx_roc[keys_list[i]]['mean_roc_val'])
        val_std_roc.append(mtrx_roc[keys_list[i]]['std_roc_val'])
        #for precision-recall
        val_mean_pr.append(mtrx_pr[keys_list[i]]['mean_pr_val'])
        val_std_pr.append(mtrx_pr[keys_list[i]]['std_pr_val'])
    
    #Save the table with the results of the evaluation metrics
    mtrxEv_df = pd.DataFrame()
    mtrxEv_df['hyperparameters']  = hyperparam
    mtrxEv_df['accuracy_validation_mean']  = val_mean
    mtrxEv_df['accuracy_validation_std']  = val_std
    mtrxEv_df['accuracy_train_mean']  = train_mean
    mtrxEv_df['accuracy_train_std']  = train_std
    mtrxEv_df['specificity_validation_mean']  = val_mean_0
    mtrxEv_df['specificity_validation_std']  = val_std_0
    mtrxEv_df['specificity_train_mean']  = train_mean_0
    mtrxEv_df['specificity_train_std']  = train_std_0
    mtrxEv_df['sensitivity_validation_mean']  = val_mean_1
    mtrxEv_df['sensitivity_validation_std']  = val_std_1
    mtrxEv_df['sensitivity_train_mean']  = train_mean_1
    mtrxEv_df['sensitivity_train_std']  = train_std_1
    mtrxEv_df['precision_validation_mean']  = val_mean_prec
    mtrxEv_df['precision_validation_std']  = val_std_prec
    mtrxEv_df['recall_validation_mean']  = val_mean_recall
    mtrxEv_df['recall_validation_std']  = val_std_recall
    mtrxEv_df['fscore_validation_mean']  = val_mean_fscore
    mtrxEv_df['fscore_validation_std']  = val_std_fscore
    mtrxEv_df['rocauc_validation_mean']  = val_mean_roc
    mtrxEv_df['rocauc_validation_std']  = val_std_roc
    mtrxEv_df['precrecall_validation_mean']  = val_mean_pr
    mtrxEv_df['precrecall_validation_std']  = val_std_pr
    #Save custom made metrics
    mtrxEv_df['score1'] =[(a + b)/2 for a, b in zip(val_mean_0, val_mean_1)]
    mtrxEv_df['score2'] =[abs(a - b) for a, b in zip(val_mean_0, val_mean_1)]
    mtrxEv_df['rank_score'] = mtrxEv_df['score1'] - mtrxEv_df['score2']
    #Sort the table
    mtrxEv_df = mtrxEv_df.sort_values(by='rank_score', ascending=False)

    return(mtrxEv_df)

###################################################################################################
###################################################################################################
#Read the hyperparameters from the parameters.txt file and save the parameters into a dictionary

def read_parameters(file_path, section_name):
    config = configparser.ConfigParser()
    config.read(file_path)
    
    grid = {}

    if section_name in config:
        for key in config[section_name]:
            # Remove all blank spaces from the value
            value = config[section_name][key].replace(' ', '')

            # Convert the value to a list, splitting by comma
            if ',' in value:
                value = value.split(',')
                # Try to convert each item to float or int if possible
                value = [try_convert(v) for v in value]
            elif value == 'None':
                value = [None]  # Handle 'None' as a single-element list containing None
            else:
                # Wrap single value in a list and convert
                value = [try_convert(value)]
                
            # Add the key-value pair to the grid dictionary
            grid[key] = value
    else:
        raise ValueError(f"Section '{section_name}' not found in the file.")

    return grid

def try_convert(value):
    """ Helper function to convert strings to int or float if applicable """
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # Return as string if it cannot be converted