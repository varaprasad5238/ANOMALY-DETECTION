import random
import numpy as np

class CustomIsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None

    def fit(self, X, current_height):
        if len(X) <= 1 or current_height >= self.height_limit:
            return
        self.split_feature = random.randint(0, X.shape[1] - 1)
        min_val, max_val = np.min(X[:, self.split_feature]), np.max(X[:, self.split_feature])
        if min_val == max_val:
            return
        self.split_value = random.uniform(min_val, max_val)
        left_mask = X[:, self.split_feature] < self.split_value
        right_mask = ~left_mask
        self.left = CustomIsolationTree(self.height_limit)
        self.right = CustomIsolationTree(self.height_limit)
        self.left.fit(X[left_mask], current_height + 1)
        self.right.fit(X[right_mask], current_height + 1)

    def path_length(self, x, current_height):
        if self.split_feature is None or current_height >= self.height_limit:
            return current_height
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, current_height + 1)
        else:
            return self.right.path_length(x, current_height + 1)


class CustomIsolationForest:
    def __init__(self, n_estimators=100, height_limit=None):
        self.n_estimators = n_estimators
        self.trees = []
        self.height_limit = height_limit

    def fit(self, X):
        n_samples = X.shape[0]
        self.height_limit = int(np.ceil(np.log2(n_samples))) if self.height_limit is None else self.height_limit
        for _ in range(self.n_estimators):
            sample_idx = np.random.choice(n_samples, size=n_samples, replace=False)
            sample_data = X[sample_idx]
            tree = CustomIsolationTree(self.height_limit)
            tree.fit(sample_data, 0)
            self.trees.append(tree)

    def anomaly_score(self, X):
        scores = []
        for x in X:
            avg_path_length = np.mean([tree.path_length(x, 0) for tree in self.trees])
            score = 2 ** (-avg_path_length / self.height_limit)
            scores.append(score)
        return np.array(scores)

