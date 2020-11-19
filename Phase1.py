import pandas as pd 
import numpy as np
import seaborn as sns
from seaborn import JointGrid
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Integer

SEED = 42 

class CO2_uptake_predictor():

    def __init__(self, path_name, fit = True):
        '''
        Args:
            path_name: path to the csv file where data was stored (str)
            fit: fit a model if True, else reduces the need to fit model. (bool)
        '''

        print("Retrieving data...\n")
        self.data = pd.read_csv(path_name)
        print("Retrieving data completed.\n")

        self.x = self.data.drop(["CO2 uptake"], axis = 1)
        self.y = self.data["CO2 uptake"]

        # Previously optimized model
        self.model = XGBRegressor(
            colsample_bytree=0.3564295055372352, 
            eta=0.11551117649849924,
            max_depth=5, 
            n_estimators=566,
            reg_lambda=0.7201545414475794,
            ubsample=0.3968211938032326)

        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.x, self.y, random_state = SEED)
        
        if fit:
            print("Fitting Data into Model...\n")
            self.model.fit(self.xtrain, self.ytrain)
            print("Fitting Data into Model completed.\n")

        # pp : predictions 
        # pp_tr : predictions on training set. (Used for data vizualization)
        self.pp = self.model.predict(self.xtest)
        self.pp_tr = self.model.predict(self.xtrain)

        # trainr2 : Training R2 of the model 
        # testr2 : Test R2 of the model 
        self.trainr2 = self.model.score(self.xtrain, self.ytrain)
        self.testr2 = r2_score(self.ytest, self.pp)

    def display_scores(self):
        # Render the scores of the model when called. 
        # Format is 3dp. 

        print("Train R2: %.3f\n" %self.trainr2)
        print("Test R2: %.3f\n" %self.testr2)
        return    
   
    def show_importance(self):
        # Display the feature importances in ascending order.
        feature_importance = self.model.get_booster().get_score(importance_type='weight')

        importance = pd.DataFrame(data= list(feature_importance.values()), index = list(feature_importance.keys()),
                                  columns=['score']).sort_values(by='score',ascending =False)
        plt.figure(figsize=(20,10))

        sns.barplot(x=importance.score,y=importance.index,orient='h')
        plt.xticks(fontsize=20)
        plt.show()
        return 
    
    def bayes_me_up(self, n_iter = 50):
        # Perform Bayesian Optimization 
        
        # ss : Search space of the features.
        ss ={'n_estimators':Integer(100,1000),
             'learning_rate':Real(0.01,1.0),
             'colsample_bytree':Real(0.01,1),
             'subsample':Real(0.01,1),
             'max_depth':Integer(2,10),
             'reg_lambda':Real(0.5,1)}

        def on_step(optim_result):
            """
            Callback meant to view scores after each iteration while performing Bayesian Optimization in Skopt"""
            score = bayes_search.best_score_
            print("best score: %s \n" % score)
            if score >= 0.98:
                print('Interrupting!')
                return True

        print("Starting Bayesian Optimization on hyperparameters...")
        bayes_search = BayesSearchCV(self.model, ss, n_iter=n_iter, scoring="r2", cv=5)

        bayes_search.fit(self.xtrain, self.ytrain, callback=on_step)
        print("Completed.")

        bayesian_output = bayes_search.best_estimator_

        print("New Train R2: %.3f\n" %bayesian_output.score(self.xtrain, self.ytrain))
        print("New Test R2: %.3f\n" %r2_score(self.ytest,(bayesian_output.predict(self.xtest))))
        
        # After comparing the metrics of the new bayesian optimizaeed model, user can either keep the existing
        # model or update it to the new model
        update = int(input("If you would like to update the model's parameters, press 1, else 0\n"))
        if update:
            self.model = bayesian_output
            print("Fitting Data into Model...\n")
            self.model.fit(self.xtrain, self.ytrain)
            print("Fitting Data into Model completed.\n")

            self.pp = self.model.predict(self.xtest)
            self.pp_tr = self.model.predict(self.xtrain)
            print("Model updated.\n")

        return
    
    def plot_everything(self):
        # Plot a complete graph containing Predicted and True Values on a JointGrid
        # Coupled with KDE plots on the marginal axis.

        self.g = JointGrid(x=self.ytest, y=self.pp)
        sns.scatterplot(x=self.ytest, y=self.pp, s=30, color='blue', ax=self.g.ax_joint)
        sns.scatterplot(x=self.ytrain, y=self.pp_tr, s=30, color='orange', ax=self.g.ax_joint)
        self.g.ax_joint.legend(["Test", "Train"])
        self.g.set_axis_labels("Experimental Values", "Predicted Values", fontsize =12)
        
        sns.kdeplot(self.ytest, ax=self.g.ax_marg_x, color ='blue')
        sns.kdeplot(self.pp, ax=self.g.ax_marg_x, color ='orange')
        self.g.ax_marg_x.legend(["Test", "Train"])
        self.g.ax_marg_x.set_title("KDE plots of Train & Test data")
        
        r2_txt = "Test R2: " + str(round(r2_score(self.ytest,self.pp),4))
        r2_tr_txt = "Train R2: " + str(round(self.model.score(self.xtrain, self.ytrain),4))

        plt.text(0.2,1.4,r2_tr_txt, fontsize=10)
        plt.text(0.2,0.6,r2_txt, fontsize=10)
        
        plt.show()
        return

    def predict_value(self):
        # Used to predict unseen data or new data from the scientists. 
        # Output will be in an array
        
        counter = int(input("Please enter number of datapoints to predict:\n"))
        to_test = pd.DataFrame(columns = self.data.columns.tolist()[:-1])
        for c in range(counter):
            row = [] 
            for col in self.data.columns.tolist()[:-1]:
                row.append(float(input("Please key in a value for %s:\n" %col)))
            to_test.loc[c] = row

        print("Predicting...")
        print(self.model.predict(to_test))
        print("Predicting completed.")
        return

    
            


    

    