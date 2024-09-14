import numpy as np
from sklearn.datasets import load_iris


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators  # Number of trees
        self.learning_rate = learning_rate  # Learning rate
        self.max_depth = max_depth  # Maximum depth of each tree
        self.trees = []  # List to store trees

    def _mse(self, y_true, y_pred):
        """Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)

    def _fit_tree(self, X, residuals):
        """Fit a simple decision tree (regressor) on the residuals."""
        from collections import defaultdict
        from math import inf

        n_samples, n_features = X.shape
        best_split = {'feature_index': None, 'threshold': None, 'mse': inf, 'left_value': None, 'right_value': None}
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] < threshold
                right_mask = X[:, feature_index] >= threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_value = np.mean(residuals[left_mask])
                right_value = np.mean(residuals[right_mask])
                mse = (np.sum(left_mask) * np.mean((residuals[left_mask] - left_value) ** 2) +
                       np.sum(right_mask) * np.mean((residuals[right_mask] - right_value) ** 2)) / n_samples

                if mse < best_split['mse']:
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'mse': mse,
                        'left_value': left_value,
                        'right_value': right_value
                    }

        def _tree_predict(X):
            predictions = np.zeros(len(X))
            left_mask = X[:, best_split['feature_index']] < best_split['threshold']
            right_mask = ~left_mask
            predictions[left_mask] = best_split['left_value']
            predictions[right_mask] = best_split['right_value']
            return predictions

        return _tree_predict

    def fit(self, X, y):
        """Fit the Gradient Boosting model."""
        n_samples = X.shape[0]
        # Initial prediction: average of y
        self.initial_prediction = np.mean(y)
        y_pred = np.full(n_samples, self.initial_prediction)

        # Train trees sequentially
        for _ in range(self.n_estimators):
            # Compute the residuals (negative gradient)
            residuals = y - y_pred
            # Fit a tree on the residuals
            tree = self._fit_tree(X, residuals)
            # Add the new tree to the list
            self.trees.append(tree)
            # Update the predictions
            y_pred += self.learning_rate * tree(X)

    def predict(self, X):
        """Make predictions."""
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree(X)
        return y_pred


# Example Usage
if __name__ == "__main__":
    # Create a synthetic dataset for demonstration
    # np.random.seed(42)
    # X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    # y = 2 * X.flatten() + np.sin(X.flatten()) + np.random.randn(100)  # Target with some noise

    data = load_iris()
    X = data.data
    y = data.target

    # Split into training and test sets
    train_size = 80
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"Mean Squared Error on test set: {mse:.4f}")