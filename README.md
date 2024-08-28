Machine learning methods applied to classify complex diseases using genomic data
==========
Python scripts for the manuscript
© Magdalena Arnal Segura, 2024
https://doi.org/10.1101/2024.03.18.585541

AUTHOR/SUPPORT
==============
https://github.com/machalen/ML_complex_diseases/issues </br>

MANUAL
======
https://github.com/alexdobin/STAR/blob/master/doc/STARmanual.pdf

DIRECTORY CONTENTS
==================
  * Hyperparameters: All parameters used in the hyperparameter selection process.
  * Python_scripts: Python scripts divided in different subdirectories corresponding to the different ML and DL models.
  * requirements.txt: Required packages for the installation.

RUNNING THE CODE
=====================

All the Python scripts are located in the 'Python_scripts' folder and are organized into subdirectories named according to the ML and DL methods used:
  * Logistic Regression (LR)
  * Gradient-Boosted Decision Trees (GB)
  * Random Forest (RF)
  * Extremely Randomized Trees (ET)
  * Feedforward neural network (FFN)
  * Convolutional neural networks (CNN)

The strategy employed is nested cross-validation (nested CV), which is an adaptation of the K-fold CV that consists in setting one outer loop and one inner loop of CV. In this approach, the CV in the inner loop is performed on the training set of the outer loop and is used to select the optimum hyperparameter configuration. This step is implemented in the scripts indexed with 1_:

  * ./Python_scripts/LR/1_LR_With_nestedCV.py
  * ./Python_scripts/GB/1_GB_With_nestedCV.py
  * ./Python_scripts/RF/1_RF_With_nestedCV.py
  * ./Python_scripts/ET/1_ET_With_nestedCV.py
  * ./Python_scripts/FFN/1_FFN_With_nestedCV.py
  * ./Python_scripts/CNN/1_CNN_With_nestedCV.py

Conversely, the CV in the outer loop is used to train the final model with the selected hyperparameter configuration obtained from the inner loop, and to test the model with the remaining test set that has not been used for hyperparameter selection or training the model. This step is implemented in the scripts indexed with 2_:

  * ./Python_scripts/LR/2_LR_FinalModel.py
  * ./Python_scripts/GB/2_GB_FinalModel.py
  * ./Python_scripts/RF/2_RF_FinalModel.py
  * ./Python_scripts/ET/2_ET_FinalModel.py
  * ./Python_scripts/FFN/2_FFN_FinalModel.py
  * ./Python_scripts/CNN/2_CNN_FinalModel.py

Iterating over different folds in the inner and outer loop allows for the use of different samples in training, validation, and testing in each iteration, optimizing the use of all the available samples. At the end, nested CV generate as many final models as number of folds in the outer loop.

![nested CV](./images/Figure_NestedCV.png)

--------------------------------------------------------


```bash
# Get latest STAR source from releases
wget https://github.com/alexdobin/STAR/archive/2.7.11b.tar.gz
tar -xzf 2.7.11b.tar.gz
cd STAR-2.7.11b

# Alternatively, get STAR source using git
git clone https://github.com/alexdobin/STAR.git
```


LIMITATIONS
===========
This release was tested with the default parameters for human and mouse genomes.
Mammal genomes require at least 16GB of RAM, ideally 32GB.
Please contact the author for a list of recommended parameters for much larger or much smaller genomes.


FUNDING
=======
The development of STAR is supported by the National Human Genome Research Institute of
the National Institutes of Health under Award Number R01HG009318.
The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.
