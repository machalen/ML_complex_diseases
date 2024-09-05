#!/bin/bash


# mamba activate dataScience
# sys.path.append('/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/')
# file_path = '/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Hyperparameters/parameters.txt'

#######################################################################################################################################
#1_CNN_With_nestedCV.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/CNN/1_CNN_With_nestedCV.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD_Imp.NumMtrx.sorted.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
conv_chr='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/Conv1D_chr_AD.Red.txt'
conv_pos='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/Conv1D_pos_AD.Red.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSamplnCV_'
PatCond=AD
manualFold=0

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -a "$conv_chr" -p "$conv_pos" -o "$outDir" -c "$PatCond" -f "$manualFold"

#######################################################################################################################################
#1_FFN_With_nestedCV.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/FFN/1_FFN_With_nestedCV.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.Imp.NumMtrx.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSamplnCV_'
PatCond=AD
manualFold=0

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold"

#######################################################################################################################################
#1_GB_With_nestedCV.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/GB/1_GB_With_nestedCV.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.Imp.NumMtrx.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSamplnCV_'
PatCond=AD
manualFold=0

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold"

#######################################################################################################################################
#1_LR_With_nestedCV.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/LR/1_LR_With_nestedCV.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.Imp.NumMtrx.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSamplnCV_'
PatCond=AD
manualFold=0

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold"

#######################################################################################################################################
#1_ET_With_nestedCV.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/ET/1_ET_With_nestedCV.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.Imp.NumMtrx.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSamplnCV_'
PatCond=AD
manualFold=0

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold"

#######################################################################################################################################
#1_RF_With_nestedCV.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/RF/1_RF_With_nestedCV.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.Imp.NumMtrx.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSamplnCV_'
PatCond=AD
manualFold=0

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold"
########################################################################################################################################
########################################################################################################################################
#2_CNN_FinalModel.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/CNN/2_CNN_FinalModel.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD_Imp.NumMtrx.sorted.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
conv_chr='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/Conv1D_chr_AD.Red.txt'
conv_pos='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/Conv1D_pos_AD.Red.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSampl_'
PatCond=AD
manualFold=0
numEp=200
learningRate=0.01
selfdr=0.1
nUnits=100
nLayers=2
balance=50
sampl_strategy=random

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -a "$conv_chr" -p "$conv_pos" -o "$outDir" -c "$PatCond" -f "$manualFold" -e "$numEp" -r "$learningRate" -d "$selfdr" -u "$nUnits" -y "$nLayers" -b "$balance" -s "$sampl_strategy"

########################################################################################################################################
#2_FFN_FinalModel.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/FFN/2_FFN_FinalModel.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.Imp.NumMtrx.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSampl_'
PatCond=AD
manualFold=0
numEp=200
learningRate=0.01
selfdr=0.1
nUnits=100
nLayers=2
balance=50
sampl_strategy=random

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold" -e "$numEp" -r "$learningRate" -d "$selfdr" -u "$nUnits" -y "$nLayers" -b "$balance" -s "$sampl_strategy"

########################################################################################################################################
#2_GB_FinalModel.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/GB/2_GB_FinalModel.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.Imp.NumMtrx.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSampl_'
PatCond=AD
manualFold=0
H_ne=50
H_lr=0.01
H_sb=0.5
H_md=None
H_lo=exponential
balance=50
sampl_strategy=SMOTE_random

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold" -e "$H_ne" -r "$H_lr" -a "$H_sb" -d "$H_md" -u "$H_lo" -b "$balance" -s "$sampl_strategy"

########################################################################################################################################
#2_LR_FinalModel.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/LR/2_LR_FinalModel.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.Imp.NumMtrx.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSampl_'
PatCond=AD
manualFold=0
H_so=newton-cg
H_re=1
balance=50
sampl_strategy=SMOTE_random

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold" -v "$H_so" -r "$H_re" -b "$balance" -s "$sampl_strategy"

########################################################################################################################################
#2_ET_FinalModel.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/ET/2_ET_FinalModel.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.Imp.NumMtrx.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSampl_'
PatCond=AD
manualFold=0
H_ne=50
H_ss=2
H_sl=5
H_md=None
balance=50
sampl_strategy=SMOTE_random

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold" -e "$H_ne" -a $H_ss -i $H_sl -d $H_md -b "$balance" -s "$sampl_strategy"

########################################################################################################################################
#2_RF_FinalModel.py

SCRIPT='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/ML_complex_diseases/Python_scripts/RF/2_RF_FinalModel.py'
inMtrx='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.Imp.NumMtrx.Red.txt'
inLab='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/DiseaseGenotypeTables/AD/AD.samples.txt'
outDir='/Users/magdalenaarnal/OneDrive - peptomyc.com/PhD_Italy/UKB_Project_79450/Python_tables/AD/TestSampl_'
PatCond=AD
manualFold=0
H_ne=50
H_ss=2
H_sl=5
H_md=None
balance=50
sampl_strategy=SMOTE_random

python "$SCRIPT" -m "$inMtrx" -l "$inLab" -o "$outDir" -c "$PatCond" -f "$manualFold" -e "$H_ne" -a $H_ss -i $H_sl -d $H_md -b "$balance" -s "$sampl_strategy"