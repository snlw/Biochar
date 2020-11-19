import pandas as pd 
import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class BayesianSampler():
    def __init__(self, path_name, n_iter):
        self.data = pd.read_csv(path_name)

        #Obtain the bounds of the search space of each feature using the minmax 
        self.bounds = np.array([[min(self.data[col]),max(self.data[col])] for col in self.data.columns.tolist()[:-1]])

        #Prepare individual sets where x contains the features and y contains the target(Surface Area)
        self.x = self.data.drop("Surface Area", axis =1)
        self.y = self.data["Surface Area"]

        #Prepare individual sets where X_init contains only the values of the features in the form of arrays and Y_init contains the target values
        self.X_init = self.x.values
        self.Y_init = self.y.values.reshape(-1,1)

        # def create_sample(desired_length = 60):
        #     '''
        #     Creates X which contains arrays of feature values by assuming a normal distribution within the bounds of features
        #     Creates Y which contains the values of the corresponding feature values in X by applying the black box function
        #     Args:
        #         desired_length: Number of sample points to evalute EI 
            
        #     Returns:
        #         Arrays X (desired_length x n) and Y (desired_length x 1).
        #     '''
        #     X = []
        #     for _ in range(desired_length):
        #         X1 = np.random.uniform(low=self.bounds[:,0], high= self.bounds[:,1])
        #         X.append(X1)
        #     X = np.array(X)

        #     #Y = np.array([f(x, gpr) for x in X]).reshape(-1,1)
            
        #     return X #Y

        def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
            '''
            Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model.
            
            Args:
                X: Points at which EI shall be computed (m x d).
                X_sample: Sample locations (n x d).
                Y_sample: Sample values (n x 1).
                gpr: A GaussianProcessRegressor fitted to samples.
                xi: Exploitation-exploration trade-off parameter.
            
            Returns:
                Expected improvements at points X.
            '''
            mu, sigma = gpr.predict(X, return_std=True)
            mu_sample = gpr.predict(X_sample)

            sigma = sigma.reshape(-1, 1)
            
            # Needed for noise-based model,
            # otherwise use np.max(Y_sample).
            # See also section 2.4 in [...]
            mu_sample_opt = np.max(mu_sample)

            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0

            return ei

        def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
            '''
            Proposes the next sampling point by optimizing the acquisition function.
            *Optimization is restarted n_restarts times to avoid local optima.
            Args:
                acquisition: Acquisition function.
                X_sample: Sample locations (n x d).
                Y_sample: Sample values (n x 1).
                gpr: A GaussianProcessRegressor fitted to samples.
                bounds: bounds of X_sample
                n_restarts: Number of iterations for minimize function. Larger value can escape local optima trap.

            Returns:
                Location of the acquisition function maximum.
            '''
            dim = X_sample.shape[1]

            def min_obj(X):
                # Minimization objective is the negative acquisition function
                return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
            
            val=[]
            idx=[]
            
        #     for x0 in np.random.uniform(bounds[:,0], bounds[:,1], size=(n_restarts, dim)):
        #         res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        #         val.append(res.fun[0][0])
        #         idx.append(res.x[0])

            for _ in range(n_restarts):
                X1 = np.random.uniform(low=bounds[0][0], high= bounds[0][1])
                X2 = np.random.uniform(low=bounds[1][0], high= bounds[1][1])
                X3 = np.random.uniform(low=bounds[2][0], high= bounds[2][1])
                X4 = np.random.uniform(low=bounds[3][0], high= bounds[3][1])
                X5 = np.random.uniform(low=bounds[4][0], high= bounds[4][1])
                X6 = np.random.uniform(low=bounds[5][0], high= bounds[5][1])
                x0 = [X1,X2,X3,X4,X5,X6]
                
                res = minimize(min_obj, x0=x0, bounds = bounds, method='L-BFGS-B')
                val.append(res.fun[0][0])
                idx.append(res.x)

            helper={k:v for k,v in zip(val,idx)}
            helper_sorted= sorted(helper.items(),key=lambda x:x[0],reverse=False)
        #     best = [helper_sorted[0][1]]
            topfive = [i[1] for i in helper_sorted[:5]]

            return np.array(topfive)

        # Prepare surrogate model for Bayesian Optimization
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.model = GaussianProcessRegressor(kernel=m52, alpha= 0.2*0.2)

        self.X_sample = self.X_init
        self.Y_sample = self.Y_init

        self.n_iter = n_iter

        # Update Gaussian process with existing samples
        self.model.fit(self.X_sample, self.Y_sample)
        
        # Obtain next sampling point from the acquisition function (expected_improvement)
        self.X_next = propose_location(expected_improvement, self.X_sample, self.Y_sample, self.model, self.bounds)

    def display_next_sampling_point(self):
        l = len(self.X_next)
        print("Note: Temperature (oC), Time (hr), Flow (ml/min)\n")
        for j in range(l):
            i = 0
            for col in self.data.columns.tolist()[:-1]:
                print(col.title() + ':' + str(self.X_next[j][i]) +'\n')
                i += 1
        return
    
