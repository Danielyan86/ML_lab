import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] < threshold
    right_mask = X[:, feature_index] >= threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]


# gini index is used to calculate the impurity of the node
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        counts = np.bincount(group, minlength=len(classes))
        probabilities = counts / size
        score = np.sum(probabilities * probabilities)
        gini += (1.0 - score) * (size / n_instances)
    return gini


def get_best_split(X, y):
    class_values = list(set(y))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_X, right_X, left_y, right_y = split_dataset(
                X, y, feature_index, threshold
            )
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            gini = gini_index([left_y, right_y], class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = (
                    feature_index,
                    threshold,
                    gini,
                    (left_X, right_X, left_y, right_y),
                )
    return {"index": b_index, "value": b_value, "groups": b_groups}


def build_tree(X, y, depth=0, max_depth=10):
    # count the number of samples for each class
    num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
    predicted_class = np.argmax(num_samples_per_class)
    node = TreeNode(
        gini=gini_index([y.tolist()], np.unique(y).tolist()),
        num_samples=len(y),
        num_samples_per_class=num_samples_per_class,
        predicted_class=predicted_class,
    )
    if depth < max_depth:
        split = get_best_split(X, y)
        if split["groups"]:
            left_X, right_X, left_y, right_y = split["groups"]
            if len(left_y) > 0 and len(right_y) > 0:
                node.feature_index = split["index"]
                node.threshold = split["value"]
                node.left = build_tree(left_X, left_y, depth + 1, max_depth)
                node.right = build_tree(right_X, right_y, depth + 1, max_depth)
    return node


def predict(node, X):
    if node.left is None and node.right is None:
        return node.predicted_class
    if X[node.feature_index] < node.threshold:
        return predict(node.left, X)
    else:
        return predict(node.right, X)


if __name__ == "__main__":
    # Load the Iris dataset from CSV
    df = pd.read_csv("iris.csv")

    # Map the variety column to numerical values
    df["variety"] = df["variety"].map({"Setosa": 0, "Versicolor": 1, "Virginica": 2})

    # Extract features and labels
    X = df.iloc[:, :-1].values  # extract all columns except the last one
    y = df.iloc[:, -1].values  # extract the last column

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # Build the CART tree
    trained_tree_model = build_tree(X_train, y_train, max_depth=10)

    # Make predictions on the test set
    y_pred = [predict(trained_tree_model, row) for row in X_test]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
