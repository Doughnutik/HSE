from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeRegressor

from typing import Optional
import matplotlib.pyplot as plt


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = None,
        subsample: float = 0.8,
        bagging_temperature: float = 1.0,
        bootstrap_type: Optional[str] = "Bernoulli",
        rsm: float = 1.0,
        quantization_type: Optional[str] = None,
        nbins: int = 255,
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)  # Исправьте формулу на правильную.

        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type

        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins

        self.feature_importances_ = None


    def calculate_feature_importances(self, n_features: int):
        importances = np.zeros(n_features)
        for model in self.models:
            # features all the same
            tree_importances = model.feature_importances_
            importances += tree_importances
            
        total_importance = importances.sum()
        importances /= total_importance

        self.feature_importances_ = importances
        

    def bootstrap(self, X, y):
        if self.bootstrap_type == "Bernoulli":
            mask = np.random.rand(X.shape[0]) < self.subsample
            indices = np.where(mask)[0]
            return X[indices], y[indices], indices
        elif self.bootstrap_type == "Bayesian":
            U = np.random.rand(X.shape[0])
            weights = (-np.log(U)) ** self.bagging_temperature
            return X, y * weights, np.arange(X.shape[0])
        else:
            return X, y, np.arange(X.shape[0])

    def select_features(self, X):
        n_features = X.shape[1]
        n_selected = int(self.rsm * n_features)
        selected_indices = np.random.choice(n_features, n_selected, replace=False)
        selected_indices.sort()
        # print(f"Selected features indices: {selected_indices}")
        return X[:, selected_indices], selected_indices
    
    def quantize_features(self, X):
        X = X.toarray()
        if self.quantization_type == "uniform":
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            bins = [np.linspace(min_val, max_val, self.nbins) for min_val, max_val in zip(min_vals, max_vals)]
            X_quantized = np.array([np.digitize(X[:, i], bins[i], right=True) for i in range(X.shape[1])]).T
            return X_quantized
        elif self.quantization_type == "quantile":
            quantiles = np.linspace(0, 1, self.nbins)
            bins = [np.quantile(X[:, i], quantiles) for i in range(X.shape[1])]
            X_quantized = np.array([np.digitize(X[:, i], bins[i], right=True) for i in range(X.shape[1])]).T
            return X_quantized
        return X

    def partial_fit(self, X, y, preds):
        # X_selected, selected_featres = self.select_features(X)
        # X_quanted = self.quantize_features(X)
        X_sample, y_sample, mask = self.bootstrap(X, y)
        shifts = -self.loss_derivative(y_sample, preds[mask])

        model = self.base_model_class(**self.base_model_params)
        model.fit(X_sample, shifts)
        gamma = self.find_optimal_gamma(y, preds, model.predict(X))

        self.gammas.append(gamma)
        self.models.append(model)
        # raise Exception("partial_fit method not implemented")

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        train_predictions = np.zeros(X_train.shape[0])
        valid_predictions = np.zeros(X_val.shape[0])

        self.history['train_losses'] = []
        self.history['val_losses'] = []

        best_loss = np.inf
        cnt_bad_rounds = 0

        for iter_num in range(self.n_estimators):
            self.partial_fit(X_train, y_train, train_predictions)

            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_train)
            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_val)

            train_loss = self.loss_fn(y_train, train_predictions)
            val_loss = self.loss_fn(y_val, valid_predictions)

            self.history['train_losses'].append(train_loss)
            self.history['valid_losses'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                cnt_bad_rounds = 0
            else:
                cnt_bad_rounds += 1

            if self.early_stopping_rounds is not None and cnt_bad_rounds >= self.early_stopping_rounds:
                print(f"Early stopping at iteration {iter_num}")
                break

        if plot:    
            self.plot_history(X_val, y_val)

    def predict_proba(self, X):
        pred = np.zeros(X.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            pred += gamma * self.learning_rate * model.predict(X)
        probas = np.zeros((X.shape[0], 2))
        probas[:, 1] = self.sigmoid(pred)
        probas[:, 0] = 1 - self.sigmoid(pred)
        return probas
        # raise Exception("predict_proba method not implemented")

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, X, y):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        y_proba = self.predict_proba(X)[:, 1]
        # print(X.shape)
        # print(y.shape)
        # print(y_proba.shape)
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        roc_auc = roc_auc_score(y, y_proba)
        plt.figure(figsize=(20, 16))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc})')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        # raise Exception("plot_history method not implemented")
