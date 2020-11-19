# Biochar

# Phase 1: Predict CO2 uptake from textural properties/ chemical composition. 

How to use:<br>
from Phase1 import CO2_uptake_predictor<br>

Path Name represents the file path to the csv file<br>
cup = CO2_uptake_predictor(path_name)<br>

'''<br>
Fuctions: 
1. Display the scores of the train and validation 
2. Display feature importances of the model
3. Perform Bayesian Optimization 
4. Display the jointplot of the predictions 
5. Predict CO2 uptake of new point.
'''<br>
-----------------------------------------------------------------------------------
# Phase 2: Classify the Surface Area Range from process conditions provided.

How to use:<br>
from Phase2 import RangeClassifier<br>

* Path Name represents the file path to the csv file <br>
* Counter could be specified to vary the range for each class <br>
* mn and mx represents the lower and upper bound of the classes to analyze <br>

rc = RangeClassifier(path_name)

'''<br>
Fuctions: 
1. Display the scores of the train and validation 
2. Display feature importances of the model
3. Predict Surface Area Range of new point.
'''<br>
-----------------------------------------------------------------------------------
# Phase 3: Bayesian Optimization to obtain new sampling point

How to use:<br>
from Phase3 import BayesianSampler<br>

* Path Name represents the file path to the csv file <br>
* n_iter to determine rounds of iteration (preferably just 1)<br>

bs = BayesianSampler(path_name, n_iter = 1)

'''<br>
Fuctions: 
1. Display the feature values of the new sampling point.
'''<br>
