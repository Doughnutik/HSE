import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict
from typing import Optional
import matplotlib.pyplot as plt


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
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
        self.base_model_params = base_model_params or {}
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins

        self.models = []
        self.gammas = []
        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

        self.feature_importances_ = None

    def calculate_feature_importances(self, n_features: int):
        importance_scores = np.zeros(n_features)
        for model in self.models:
            importance_scores += model.feature_importances_
        self.feature_importances_ = importance_scores / importance_scores.sum()

    def bootstrap(self, X, y):
        if self.bootstrap_type == "Bernoulli":
            mask = np.random.rand(X.shape[0]) < self.subsample
            indices = np.where(mask)[0]
            return X[indices], y[indices], indices
        elif self.bootstrap_type == "Bayesian":
            weights = (-np.log(np.random.rand(X.shape[0]))) ** self.bagging_temperature
            return X, y * weights, np.arange(X.shape[0])
        else:
            return X, y, np.arange(X.shape[0])

    def select_features(self, X):
        num_features = X.shape[1]
        selected = np.random.choice(num_features, int(self.rsm * num_features), replace=False)
        selected.sort()
        return X[:, selected], selected

    def quantize_features(self, X):
        if self.quantization_type == "uniform":
            bins = [np.linspace(X[:, i].min(), X[:, i].max(), self.nbins) for i in range(X.shape[1])]
            return np.array([np.digitize(X[:, i], bins[i], right=True) for i in range(X.shape[1])]).T
        elif self.quantization_type == "quantile":
            bins = [np.quantile(X[:, i], np.linspace(0, 1, self.nbins)) for i in range(X.shape[1])]
            return np.array([np.digitize(X[:, i], bins[i], right=True) for i in range(X.shape[1])]).T
        return X

    def partial_fit(self, X, y, preds):
        X_sample, y_sample, mask = self.bootstrap(X, y)
        grad = -self.loss_derivative(y_sample, preds[mask])
        model = self.base_model_class(**self.base_model_params)
        model.fit(X_sample, grad)
        gamma = self.find_optimal_gamma(y, preds, model.predict(X))
        self.models.append(model)
        self.gammas.append(gamma)

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        train_predictions = np.zeros(X_train.shape[0])
        val_predictions = np.zeros(X_val.shape[0]) if X_val is not None else None

        best_val_loss = float("inf")
        patience = 0

        for i in range(self.n_estimators):
            self.partial_fit(X_train, y_train, train_predictions)

            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_train)
            if X_val is not None:
                val_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_val)

            train_loss = self.loss_fn(y_train, train_predictions)
            self.history["train_losses"].append(train_loss)

            if X_val is not None:
                val_loss = self.loss_fn(y_val, val_predictions)
                self.history["val_losses"].append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1
                if self.early_stopping_rounds and patience >= self.early_stopping_rounds:
                    print(f"Early stopping at iteration {i}")
                    break

        if plot:
            self.plot_history(X_val, y_val)

    def predict_proba(self, X):
        prediction = np.zeros(X.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            prediction += self.learning_rate * gamma * model.predict(X)
        probs = np.zeros((X.shape[0], 2))
        probs[:, 1] = self.sigmoid(prediction)
        probs[:, 0] = 1 - probs[:, 1]
        return probs

    def find_optimal_gamma(self, y, old_predictions, new_predictions):
        gammas = np.linspace(0, 1, 100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def plot_history(self, X, y):
        probs = self.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, probs)
        auc_score = roc_auc_score(y, probs)
        plt.figure(figsize=(12, 8))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})", color="blue")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.show()