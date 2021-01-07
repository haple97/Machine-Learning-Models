Python version: Python 3
Packages: pandas, numpy, matplotlib

The folder consists of 3 files: 

1) preprocess-assg3.py: For data preprocessing, including one-hot encoding. This file takes dating-full.csv as input and return trainingSet.csv and testSet.csv

2) lr_svm.py: For training and testing Logistic Regression and Support Vector Machine. This file takes trainingSet.csv and testSet.csv as inputs and prints accuracy on training set and test set, based on the model chosen

3) cv.py: For 10-fold cross-validation, with plot of performance on difference t_frac for NBC, LR, and SVM. Additionally, the file also prints a dataframe of statistics used for hypothesis testing. This file takes trainingSet.csv (for LR and SVM) and dating-binned.csv (from HW2, for NBC model).  

These codes are written in Jupyter Notebook originally; therefore, there are comments indicating the box numbers in the notebook. The codes are then modified to allow for command line arguments. 
