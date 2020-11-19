import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier

class RangeClassifier():
    def __init__(self, path_name, fit = True, counter = 200, mn = 500, mx = 2000):
        self.data = pd.read_csv(path_name)

        def convert_to_range(x, counter):
            '''
            Args
                x: Surface Area value (int/float)
                counter: Range to classify Surface Area (default = 200)
            Return
                out: Surface Area Range (str) '''
            upper = 0
            while x > upper:
                upper += counter
            out = str(upper-counter) + " - " + str(upper)
            return out

        def set_min_max(data, mn, mx):
            '''
            Args
                data: DataFrame containing target values 
                mn: Minimum Surface Area 
                mx: Maximum Surface Area 
            Return
                sorted_data: DataFrame only containing target values within the range [mn,mx]'''
            sorted_data = data[data["Surface Area (m2/g)"] >= mn]
            sorted_data = sorted_data[sorted_data["Surface Area (m2/g)"] <= mx]
            return sorted_data
        
        # Set Min and Max of Surface Area to consider
        self.data = set_min_max(self.data, mn, mx)

        # Create Surface Area Ranges 
        self.data["Surface Area Range"] = self.data["Surface Area (m2/g)"].apply(lambda x: convert_to_range(x, counter = counter))

        # Prepare x y for resampling
        self.x_df = self.data.drop(["Surface Area Range", "Surface Area (m2/g)", "Feedstock"], axis = 1)
        self.y_df = self.data["Surface Area Range"]
        ros = RandomOverSampler(random_state=42)
        x_resampled, y_resampled = ros.fit_resample(self.x_df, self.y_df)

        # Prepare data for input 
        self.x = x_resampled.values
        self.y = y_resampled.values.reshape(-1,1)
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.x, self.y, test_size = 0.25, random_state = 42)

        self.model = XGBClassifier()
        if fit:
            self.model.fit(self.xtrain, self.ytrain)
        
    
    def display_scores(self):
        print("Training Accuracy: %.3f" %self.model.score(self.xtrain, self.ytrain))
        print("Validation Accuracy: %.3f" %accuracy_score(self.ytest, self.model.predict(self.xtest)))
        return

    # def add_data(self, d, p):
    #     self.xtrain = np.concatenate((self.xtrain, d))
    #     self.ytrain = np.concatenate((self.ytrain, p.reshape(-1,1)))
    #     self.model.fit(self.xtrain, self.ytrain)

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
    
    
    def predict_range(self, num):
        master_list = []
        for _ in range(num):
            row = []
            for col in self.x_df.columns.tolist():
                usr_input = float(input("Enter a value for " + col + ' :'))
                row.append(usr_input)
            master_list.append(row)
        master_list = np.array(master_list)

        predictions = self.model.predict(master_list)
        print(predictions)

        # add_data_yesno = input("Add data to training? 1 (YES), 0 (NO)\n")
        # if (add_data_yesno):
        #     add_data(master_list, predictions)
        return

            


