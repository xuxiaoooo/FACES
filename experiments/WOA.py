from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, clone
import numpy as np
from joblib import Parallel, delayed

class WormOptimizationSearch(BaseEstimator):

    def __init__(self, estimator, param_grid, num_worms=5, max_iter=30, scoring=accuracy_score, cv=5, n_jobs=-1, early_stopping_rounds=7, random_state=42):
        self.estimator = estimator
        self.param_grid = param_grid
        self.num_worms = num_worms
        self.max_iter = max_iter
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        
        self.best_score_ = -np.inf
        self.best_params_ = None

    def _evaluate_params(self, X, y, params):
        scores = []
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            estimator = clone(self.estimator).set_params(**params)
            estimator.fit(X_train, y_train)
            
            y_pred = estimator.predict(X_test)
            score = self.scoring(y_test, y_pred)
            scores.append(score)
        
        avg_score = np.mean(scores)
        return avg_score
    
    def fit(self, X, y):
        
        num_params = len(self.param_grid)
        param_names = list(self.param_grid.keys())
        
        param_indices = [list(range(len(self.param_grid[name]))) for name in param_names]
        
        lower_bounds = [0] * num_params
        upper_bounds = [len(self.param_grid[name]) - 1 for name in param_names]
        
        worms = np.random.uniform(lower_bounds, upper_bounds, (self.num_worms, num_params))
        
        w_max = 0.9
        w_min = 0.2
        no_improvement_count = 0

        for iter in range(self.max_iter):
            w = w_max - (w_max - w_min) * iter / self.max_iter

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate_params)(X, y, 
                                               {param_names[j]: self.param_grid[param_names[j]][int(np.round(worms[i, j]))] for j in range(num_params)}
                                               ) for i in range(self.num_worms)
            )
            
            for i in range(self.num_worms):
                score = results[i]
                if score > self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = {param_names[j]: self.param_grid[param_names[j]][int(np.round(worms[i, j]))] for j in range(num_params)}
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            
            if self.early_stopping_rounds is not None and no_improvement_count >= self.early_stopping_rounds:
                print("Early stopping after {} rounds with no improvement.".format(no_improvement_count))
                break

            random_factors = np.random.uniform(-1, 1, (self.num_worms, num_params))
            worms = worms + w * random_factors * (np.array(upper_bounds) - np.array(lower_bounds))
            
            worms = np.clip(worms, lower_bounds, upper_bounds)
            
        self.estimator.set_params(**self.best_params_)

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return self.scoring(y, y_pred)

