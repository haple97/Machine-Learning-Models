Python version: Python 3
Packages: pandas, numpy, matplotlib, scipy

The folder consists of 4 files: 

1) exploration.py: This file is for data exploration. This file takes digits.csv and digits-embedding.csv as inputs and prints a grid-plot of 10 random images and a scatter plot of the points based on true class labels. 

2) kmeans.py: This file contains the basic kmeans clustering model and assessment using WC-SSD, SC, and NMI. It takes 2 command line arguments, DataFileName and K, and can be run on bash.

3) kmeans_analysis.py: This file contains all code for analysis required in Section 2. It takes dataset digits-embedding.csv as input. 
- For section 2.2.3: To run the analysis 10 times, replace seed value with one of the values in the list shown. Each iteration would output a corresponding csv file. Then, all of these csv file would be read and concatenated to perform further analysis - calculate and visualize means and standards error for each case across 10 runs. 

4) hierarchical.py: This file contains hierarchical clustering model using scipy agglomerative clustering. It takes digits-embedding.csv as input. 


These codes are written in Jupyter Notebook originally; therefore, there are comments indicating the box numbers in the notebook. The codes are then modified to allow for command line arguments. 
