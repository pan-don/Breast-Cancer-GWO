import numpy as np
import math
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score
from mealpy import Problem

class SVMJointOptimization(Problem):
    def __init__(self, bounds=None, minmax="min",
                 X_train=None, X_test=None,
                 y_train=None, y_test=None, **kwargs):
        super().__init__(bounds=bounds, minmax=minmax, **kwargs)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_features = X_train.shape[1]
        
    def v4_transfer_function(self, x):
        return abs((2 / math.pi) * math.atan((math.pi / 2) * x))
        
    def decode_solution(self, solution):
        # feature selection
        continuous_features = solution[:self.n_features]
        binary_features = []
        
        for val in continuous_features:
            prob = self.v4_transfer_function(val)
            binary_features.append(1 if np.random.rand() < prob else 0)
        
        if sum(binary_features) == 0:
            binary_features[np.random.randint(0, self.n_features)] = 1
        
        # hyperparameters
        C_log = solution[-4]
        gamma_log = solution[-3]
        C = 2 ** C_log
        gamma = 2 ** gamma_log
        class_weight_raw = solution[-1]
        class_weight = "balanced" if class_weight_raw > 0 else None
        
        return np.array(binary_features), C, gamma, class_weight

    def obj_func(self, solution):
        binary_features, C, gamma, class_weight = self.decode_solution(solution)
        
        selected_indices = np.where(binary_features == 1)[0]
        X_train_sel = self.X_train[:, selected_indices]
        X_test_sel = self.X_test[:, selected_indices]
        
        svm = SVC(
            C=C,
            gamma=gamma,
            kernel='rbf',
            class_weight=class_weight,
            random_state=42
        )
        
        svm.fit(X_train_sel, self.y_train)
        y_pred = svm.predict(X_test_sel)
        
        f1 = f1_score(self.y_test, y_pred, pos_label=1, zero_division=0)
        f1_error = 1.0 - f1
        
        feature_ratio = len(selected_indices) / self.n_features
        fitness = 0.99 * f1_error + 0.01 * feature_ratio
        return fitness