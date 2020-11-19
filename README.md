# Biochar

# Phase 1: Predict CO2 uptake from textural properties/ chemical composition. 

How to use: \
from Phase1 import CO2_uptake_predictor \

Path Name represents the file path to the csv file \
cup = CO2_uptake_predictor(path_name) \
'''
Fuctions: 
1. Display the scores of the train and validation 
2. Display feature importances of the model
3. Perform Bayesian Optimization 
4. Display the jointplot of the predictions 
5. Predict CO2 uptake of new point.
'''
-----------------------------------------------------------------------------------
# Phase 2: Classify the Surface Area Range from process conditions provided.

How to use: \
from Phase2 import RangeClassifier \

* Path Name represents the file path to the csv file \
* Counter could be specified to vary the range for each class \
* mn and mx represents the lower and upper bound of the classes to analyze \

rc = RangeClassifier(path_name)

'''
Fuctions: 
1. Display the scores of the train and validation 
2. Display feature importances of the model
3. Predict Surface Area Range of new point.
'''
-----------------------------------------------------------------------------------
# Phase 3: Bayesian Optimization to obtain new sampling point

How to use: \
from Phase3 import BayesianSampler \

* Path Name represents the file path to the csv file \
* n_iter to determine rounds of iteration (preferably just 1) \

bs = BayesianSampler(path_name, n_iter = 1)

'''
Fuctions: 
1. Display the feature values of the new sampling point.
'''
