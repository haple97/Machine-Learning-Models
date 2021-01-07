Naive Bayes Classifier (NBC) model is written from scratch, using Python 3 and the following packages: pandas, numpy, matplotlib. The model is trained and tested on a dataset called dating_full.csv, which consists of ratings given by participants in speed dating events on their partners and themselves.

The following files must be run in sequence, and dating-full.csv as input data in the same folder: 
1)	preprocess.py (for standardizing, label encoding, and normalizing data)
2)	2_1.py (for analyzing the difference in preference between male and female participants via visualizations)
3)	2_2.py (for analyzing the relationship between ratings and probability of second date via visualizations)
4)	discretize.py (for discretizing continuous variables)
5)	split.py (for Train-Test set split)
6)	5_1.py (NBC model) OR 5_3.py (for evaluation of varied fractions of training data on NBC model)

The file 5_2.py (for evaluation of varied bin size on NBC model) can be run by itself, with dating-full.csv as input data in the same folder. 

These codes are written in Jupyter Notebook originally; therefore, there are comments indicating the box numbers in the notebook.
