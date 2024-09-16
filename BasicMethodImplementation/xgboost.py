import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TreeNode:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeRegressor:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if depth < self.max_depth and len(y) > 1:
            best_split = self._get_best_split(X, y)
            if best_split:
                left_tree = self._build_tree(
                    best_split["left_X"], best_split["left_y"], depth + 1
                )
                right_tree = self._build_tree(
                    best_split["right_X"], best_split["right_y"], depth + 1
                )
                return TreeNode(
                    feature_index=best_split["feature_index"],
                    threshold=best_split["threshold"],
                    left=left_tree,
                    right=right_tree,
                )
        return TreeNode(value=np.mean(y))

    def _get_best_split(self, X, y):
        best_split = {}
        min_loss = float("inf")
        for feature_index in range(X.shape[1]):
            split_points = np.unique(
                X[:, feature_index]
            )  # Get unique values of the feature
            for split_point in split_points:
                left_mask = X[:, feature_index] < split_point
                right_mask = X[:, feature_index] >= split_point
                left_y, right_y = y[left_mask], y[right_mask]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                loss = self._calculate_loss(left_y, right_y)
                if loss < min_loss:
                    min_loss = loss
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": split_point,
                        "left_X": X[left_mask],
                        "right_X": X[right_mask],
                        "left_y": left_y,
                        "right_y": right_y,
                    }
        return best_split if best_split else None

    def _calculate_loss(self, left_y, right_y):
        return np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.root
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


class XGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_prediction = None

    def fit(self, X, y):
        self.base_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.base_prediction)
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.base_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


if __name__ == "__main__":
    # Load the Iris dataset from CSV
    df = pd.read_csv("iris.csv")

    # Map the variety column to numerical values
    df["variety"] = df["variety"].map({"Setosa": 0, "Versicolor": 1, "Virginica": 2})

    # Extract features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize the XGBoost model from scratch
    xgb_scratch = XGBoost(n_estimators=100, learning_rate=0.1, max_depth=1)

    # Train the model
    xgb_scratch.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = xgb_scratch.predict(X_test)

    # Convert predictions to integer class labels
    y_pred = np.round(y_pred).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
