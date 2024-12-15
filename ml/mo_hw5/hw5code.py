import numpy as np
from collections import Counter
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
    $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    cum_sum_left = np.cumsum(sorted_targets)
    cum_sum_right = cum_sum_left[-1] - cum_sum_left

    unique_values, split_points = np.unique(sorted_features, return_index=True)
    split_points = split_points[1:] - 1

    mid_thresholds = (unique_values[1:] + unique_values[:-1]) / 2.0

    total_count = len(target_vector)

    left_counts = split_points + 1
    right_counts = total_count - left_counts

    cum_sum_left = cum_sum_left[split_points]
    cum_sum_right = cum_sum_right[split_points]

    prob_left = cum_sum_left.astype(np.float64) / left_counts
    gini_left = 2 * prob_left * (1 - prob_left)

    prob_right = cum_sum_right.astype(np.float64) / right_counts
    gini_right = 2 * prob_right * (1 - prob_right)

    gini_values = -(left_counts.astype(np.float64) / total_count) * gini_left - (right_counts.astype(np.float64) / total_count) * gini_right
    best_index = np.argmax(gini_values)

    optimal_threshold, optimal_gini = mid_thresholds[best_index], gini_values[best_index]

    return mid_thresholds, gini_values, optimal_threshold, optimal_gini


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=1, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if "depth" not in node:
            node["depth"] = 0

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if (self._max_depth is not None and node["depth"] == self._max_depth) or sub_X.shape[0] < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}
            feature_vector = None
            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if feature_vector is None or len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            cur_split = feature_vector < threshold
            if np.sum(cur_split) < self._min_samples_leaf or np.sum(np.logical_not(cur_split)) < self._min_samples_leaf:
                continue

            if gini_best is None or gini > gini_best:
                 
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {"depth": node["depth"] + 1}, {"depth": node["depth"] + 1} 
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            threshold = node["threshold"]
            to_left = (x[feature] < threshold)
        elif self._feature_types[feature] == "categorical":
            left_categories = node["categories_split"]
            to_left = (x[feature] in left_categories)
        else:
            raise ValueError
        
        if to_left:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    def get_params(self, deep=None):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }


class LinearRegressionTree:
    def __init__(self, 
                 feature_types, 
                 max_depth=None, 
                 min_samples_split=None, 
                 min_samples_leaf=None, 
                 loss="mse", 
                 quantiles=10):
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss
        self.quantiles = quantiles
        
        self.tree = {}

    def _fit_node(self, X, y, node, depth):
        if (self.max_depth is not None and depth == self.max_depth) or \
        (self.min_samples_split is not None and len(y) < self.min_samples_split):
            node["type"] = "leaf"
            node["model"] = LinearRegression().fit(X, y)
            return

        best_feature, best_threshold = None, None
        best_loss, best_split = float("inf"), None

        for feature in range(X.shape[1]):
            if self.feature_types[feature] == "real":
                feature_vector = X[:, feature]
                thresholds = np.quantile(feature_vector, np.linspace(0, 1, self.quantiles + 2)[1:-1])

                for threshold in thresholds:
                    left_indices = feature_vector < threshold
                    right_indices = ~left_indices

                    if (self.min_samples_leaf is not None and np.sum(left_indices) < self.min_samples_leaf) or \
                    (self.min_samples_leaf is not None and np.sum(right_indices) < self.min_samples_leaf):
                        continue

                    left_model = LinearRegression().fit(X[left_indices], y[left_indices])
                    right_model = LinearRegression().fit(X[right_indices], y[right_indices])

                    if self.loss == "mse":
                        left_loss = mse(y[left_indices], left_model.predict(X[left_indices]))
                        right_loss = mse(y[right_indices], right_model.predict(X[right_indices]))
                    elif self.loss == "mae":
                        left_loss = mae(y[left_indices], left_model.predict(X[left_indices]))
                        right_loss = mae(y[right_indices], right_model.predict(X[right_indices]))
                    else:
                        raise ValueError(f"Unsupported loss function: {self.loss}")

                    split_loss = (np.sum(left_indices) / len(y)) * left_loss + \
                                (np.sum(right_indices) / len(y)) * right_loss

                    if split_loss < best_loss:
                        best_loss = split_loss
                        best_feature = feature
                        best_threshold = threshold
                        best_split = left_indices

        if best_feature is None:
            node["type"] = "leaf"
            node["model"] = LinearRegression().fit(X, y)
            return

        node["type"] = "node"
        node["feature"] = best_feature
        node["threshold"] = best_threshold
        node["left"] = {}
        node["right"] = {}

        self._fit_node(X[best_split], y[best_split], node["left"], depth + 1)
        self._fit_node(X[~best_split], y[~best_split], node["right"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "leaf":
            return node["model"].predict(x.reshape(1, -1))[0]
        if x[node["feature"]] < node["threshold"]:
            return self._predict_node(x, node["left"])
        else:
            return self._predict_node(x, node["right"])

    def fit(self, X, y):
        self._fit_node(X, y, self.tree, depth=1)

    def predict(self, X):
        return np.array([self._predict_node(x, self.tree) for x in X])