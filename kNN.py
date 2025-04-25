from collections import Counter
from vector import *

class kNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, x_test):
        distances = []
        for i in range(len(self.X_train)):
            dist = distance(self.X_train[i], x_test)
            distances.append((dist, self.y_train[i]))

        distances.sort()
        k_nearest_labels = [label for (_, label) in distances[:self.k]]
        return Counter(k_nearest_labels).most_common(1)[0][0]
