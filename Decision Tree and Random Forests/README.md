Python version: Python 3
Packages: pandas, numpy, matplotlib

The folder consists of 5 files: 

1) preprocess-assg4.py: For data preprocessing. This file takes the first 6500 lines of dating-full.csv as input and return trainingSet.csv and testSet.csv

2) trees.py: For training and testing Decision Tree, Bagged Trees (or Bagging) and Random Forests. 
- This file takes trainingSet.csv and testSet.csv as inputs
- This file prints accuracy on training set and test set, based on the model chosen 
- To run file, enter command line 'python trees.py trainingSet.csv testSet.csv [modelIdx]' with modelIdx being 1, 2, or 3 corresponding to the above models, respectively. 

3) cv_depth.py: For 10-fold cross-validation of 3 models for each value of depth limit in list [3, 5, 7, 9]. 
- This file takes trainingSet.csv 
- This file returns a .csv file containing all means and standard errors of 3 models for various values of d (depth limit), and a .png image as visualization of those results
- To run this file, enter command line 'python cv_depth.py'

4) cv_frac.py: For 10-fold cross-validation of 3 models for each value of training fraction in list [0.05, 0.075, 0.1, 0.15, 0.2]. 
- This file takes trainingSet.csv 
- This file returns a .csv file containing all means and standard errors of 3 models for various values of t_frac (training fraction), and a .png image as visualization of those results
- To run this file, enter command line 'python cv_frac.py'

5) cv_numtrees.py: For 10-fold cross-validation of 3 models for each value of number of trees in list [10, 20, 40, 50]. 
- This file takes trainingSet.csv 
- This file returns a .csv file containing all means and standard errors of 3 models for various values of t (number of trees), and a .png image as visualization of those results
- To run this file, enter command line 'python cv_numtrees.py'

These codes are written in Jupyter Notebook originally; therefore, there are comments indicating the box numbers in the notebook. The codes are then modified to allow for command line arguments. 
