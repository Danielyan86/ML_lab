import numpy as np
from sklearn.datasets import fetch_california_housing
class SimpleDecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _calculate_mse(self, y):
        """calculate the mean squared error"""
        return np.var(y) * len(y)

    def _split_dataset(self, X, y, feature_index, threshold):
        """split the dataset based on the feature and the threshold"""
        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        """find the best split for the dataset"""
        best_mse = float('inf')
        best_split = None
        n_samples, n_features = X.shape
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split_dataset(X, y, feature_index, threshold)
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue
                mse = (self._calculate_mse(y_left) + self._calculate_mse(y_right)) / n_samples
                if mse < best_mse:
                    best_mse = mse
                    best_split = {"feature_index": feature_index, "threshold": threshold, "left": (X_left, y_left), "right": (X_right, y_right)}
        return best_split

    def _build_tree(self, X, y, depth=0):
        """build the tree recursively"""
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return np.mean(y)

        best_split = self._best_split(X, y)
        # if the best split is None, then return the mean of y, which is the prediction of the leaf node
        if not best_split:
            return np.mean(y)

        left_subtree = self._build_tree(*best_split["left"], depth + 1)
        right_subtree = self._build_tree(*best_split["right"], depth + 1)
        return {"feature_index": best_split["feature_index"], "threshold": best_split["threshold"], "left": left_subtree, "right": right_subtree}

    def fit(self, X, y):
        """训练模型"""
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, x, tree):
        """对单个样本进行预测"""
        if not isinstance(tree, dict):
            return tree
        feature_index = tree["feature_index"]
        threshold = tree["threshold"]
        if x[feature_index] < threshold:
            return self._predict_sample(x, tree["left"])
        else:
            return self._predict_sample(x, tree["right"])

    def predict(self, X):
        """对数据集进行预测"""
        return np.array([self._predict_sample(x, self.tree) for x in X])

# 示例使用
if __name__ == "__main__":
    data = fetch_california_housing()
    X, y = data.data, data.target
    # 训练模型
    tree_reg = SimpleDecisionTreeRegressor(max_depth=10, min_samples_split=2)
    tree_reg.fit(X, y)

    # 预测
    y_pred = tree_reg.predict(X)
    print(y_pred)