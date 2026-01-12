import numpy as np
import scipy
from scipy.special import comb
import scipy.special
import pandas as pd

import sklearn.ensemble
import sklearn.linear_model
from .treeprob import tree_prob
from .reg import UniversalRegression

from .base_estimator import BaseEstimator
from .p_generator import get_p
from .est_utils import combination_generator
import xgboost

class NullModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])

def get_fit(X_flat, y_flat, weighting, regression_adj):
    n = X_flat.shape[1]
    if regression_adj is False:
        reg_model = NullModel()
        phi_method = lambda reg_model : np.zeros(n)
    elif regression_adj == "linear":
        reg_model = sklearn.linear_model.LinearRegression()
        phi_method = lambda reg_model: reg_model.coef_
    elif regression_adj == "tree":
        #reg_model = sklearn.ensemble.RandomForestRegressor()
        reg_model = xgboost.XGBRegressor()
        phi_method = lambda reg_model: tree_prob(np.zeros((1,n)), np.ones((1,n)), reg_model, weighting)[0].squeeze()
    else:
        raise ValueError("regression_adjustment must be False, 'linear', or 'tree'")

    reg_model.fit(X_flat, y_flat)
    phi = phi_method(reg_model)

    return reg_model, phi

class UniversalMSR(BaseEstimator):
    def __init__(self, model, baseline, weighting, reg_model_class=False, return_direct=False, seed=None):
        super().__init__(model, baseline, weighting)
        self.n = self.baseline.shape[1]
        self.p = get_p(self.n, weighting)
        self.model = model
        self.baseline = baseline
        self.gen = np.random.Generator(np.random.PCG64(seed))
        self.reg_model_class = reg_model_class
        self.sample_prob = self.p * scipy.special.binom(self.n-1, np.arange(self.n))
        self.split_samples = False
        self.split_training = False
        
        if not self.split_samples:
            sample_prob = []
            for size in range(self.n+1):
                prob = 0
                if size > 0:
                    prob += self.p[size-1]**2 * size
                if size < self.n:
                    prob += self.p[size]**2 * (self.n - size)
                sample_prob.append(
                    np.sqrt(prob) * scipy.special.binom(self.n, size)
                )
            self.sample_prob = np.array(sample_prob) / np.sum(sample_prob)

        self.return_direct = return_direct
       
    def sample_with_replacement(self):
        self.X = np.zeros((self.num_samples, self.n), dtype=float)
        self.actual_sample_prob = np.zeros(self.num_samples, dtype=float)

        elements = np.arange(self.n)
        idx = 0
        if self.split_samples:
            for offset in [0,1]:
                sizes = np.arange(self.n) + offset
                sampled_sizes = self.gen.choice(sizes, self.num_samples//2, p=self.sample_prob)
                
                for S_idx, size in enumerate(sampled_sizes):
                    indices = self.gen.choice(elements, size=size, replace=False)

                    self.X[idx, indices] = 1
                    self.actual_sample_prob[idx] = self.sample_prob[size-offset] / scipy.special.binom(self.n-1, size-offset)
                    idx += 1
                    
        else:
            sizes = np.arange(self.n+1)
            sampled_sizes = self.gen.choice(sizes, self.num_samples, p=self.sample_prob)
            
            for idx, size in enumerate(sampled_sizes):
                indices = self.gen.choice(elements, size=size, replace=False)

                self.X[idx, indices] = 1
                self.actual_sample_prob[idx] = self.sample_prob[size] / scipy.special.binom(self.n, size)
 
    def explain(self, explicand, num_samples):
        self.num_samples = num_samples // 2 * 2
        self.pair_sampling = False
        if self.weighting in ['shapley', 'beta_shapley_1_1', 'banzhaf', 'weighted_banzhaf_0.5'] and self.reg_model_class == 'linear':
            return self.explain_special_linear(explicand, num_samples) 

        self.sample_with_replacement()
        
        model_input = self.baseline * (1 - self.X) + explicand * self.X
        y = self.model.predict(model_input)

        # randomly sample m/2 training indices
        train_indices = self.gen.choice(self.num_samples, self.num_samples//2, replace=False)
        if self.split_training:
            X_train, y_train = self.X[train_indices], y[train_indices]
            X_test, y_test = self.X[~train_indices], y[~train_indices]
        else:
            X_train, y_train = X_test, y_test = self.X, y

        feature_names = [f"f{i}" for i in range(model_input.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        reg_model, reg_phi = get_fit(X_train_df, y_train, self.weighting, self.reg_model_class)        
        reg_pred = reg_model.predict(X_test_df)

        if self.return_direct:
            return reg_phi
 
        phi = np.zeros(self.n)
        for i in range(self.n):
            i_contained = (X_test[:, i] == 1)             
            not_contained = ~i_contained
            sizes = X_test.sum(axis=1).astype(int)
            if self.split_training:
                actual_sample_prob_test = self.actual_sample_prob[~train_indices]
            else:
                actual_sample_prob_test = self.actual_sample_prob
            #if self.split_samples:
            #    offset = np.zeros(len(self.X), dtype=int)
            #    offset[self.num_samples//2:] = 1
            #    i_contained = (self.X[:,i]==1) & (offset == 1)
            #    not_contained = (self.X[:,i]==0) & (offset== 0)
            phi[i] = (
                reg_phi[i] + 
                (
                    (y_test[i_contained] - reg_pred[i_contained]) * self.p[sizes[i_contained]-1] / actual_sample_prob_test[i_contained]
                 ).mean() -
                (
                    (y_test[not_contained] - reg_pred[not_contained]) * self.p[sizes[not_contained]] / actual_sample_prob_test[not_contained]
                ).mean()
            )

        return phi
        
    def explain_special_linear(self, explicand, num_samples):

        if self.weighting in ['shapley', 'beta_shapley_1_1']:
            constrain_reg = True
        elif self.weighting in ['banzhaf', 'weighted_banzhaf_0.5']:
            constrain_reg = False
        else:
            raise ValueError("weighting must be 'shapley', 'beta_shapley_1_1', 'banzhaf' or 'weighted_banzhaf_0.5'")
        
        LinearEstimator = UniversalRegression(
            self.model, self.baseline, self.weighting, with_replace=False, constrain_reg=constrain_reg,
        )

        phi_est = LinearEstimator.explain(explicand, num_samples)
        sampled = LinearEstimator.sampled
        y = LinearEstimator.y
        sizes = LinearEstimator.sizes
        prob_sampled = LinearEstimator.prob_sampled
        phi = np.zeros(self.n)

        pred = (sampled @ phi_est)

        for i in range(self.n):
            i_contained = (sampled[:, i] == 1)
            i_contained_weighting = self.p[sizes[i_contained]-1]
            not_contained_weighting = self.p[sizes[~i_contained]]

            phi[i] = (                
                phi_est[i] +
                (
                    (y[i_contained] - pred[i_contained]) * i_contained_weighting / prob_sampled[i_contained]
                 ).mean() -
                (
                    (y[~i_contained] - pred[~i_contained]) * not_contained_weighting / prob_sampled[~i_contained]
                ).mean()
            )
            if constrain_reg:
                phi[i] += (LinearEstimator.v1 - phi_est.sum())/self.n - (LinearEstimator.v0)/self.n

        return phi

class LinearMSR(UniversalMSR):
    def __init__(self, model, baseline, weighting, seed=None):
        super().__init__(model, baseline, weighting, reg_model_class='linear', seed=seed)

class TreeMSR(UniversalMSR):
    def __init__(self, model, baseline, weighting, seed=None):
        super().__init__(model, baseline, weighting, reg_model_class='tree', seed=seed)

class Tree(UniversalMSR):
    def __init__(self, model, baseline, weighting, seed=None):
        super().__init__(model, baseline, weighting, reg_model_class='tree', return_direct=True, seed=seed)

class MSR(UniversalMSR):
    def __init__(self, model, baseline, weighting, seed=None):
        super().__init__(model, baseline, weighting, reg_model_class=False, seed=seed)
