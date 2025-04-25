import math
from vector import mean_vector

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def covariance_matrix(X):
    n = len(X)
    m = len(X[0])
    mean = mean_vector(X)
    cov = [[0 for _ in range(m)] for _ in range(m)]

    for i in range(m):
        for j in range(m):
            for k in range(n):
                cov[i][j] += (X[k][i] - mean[i]) * (X[k][j] - mean[j])
            cov[i][j] /= (n - 1)
    return cov

import numpy as np

def power_iteration(A, tol=1e-6):
    A = np.asarray(A)
    v = np.random.rand(A.shape[1])
    v /= np.linalg.norm(v)

    while True:
        w = A.dot(v)
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            return v
        v_next = w / w_norm
        
        if np.linalg.norm(v_next - v) < tol:
            return v_next
        
        v = v_next


class PCA:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.mean = None
        self.components = []

    def fit(self, X):
        self.mean = mean_vector(X)
        X_centered = [[X[i][j] - self.mean[j] for j in range(len(X[0]))] for i in range(len(X))]
        cov = covariance_matrix(X_centered)

        for _ in range(self.n_components):
            eigenvector = power_iteration(cov)
            self.components.append(eigenvector)

    def transform(self, X):
        if len(X[0]) != len(self.mean):
            raise ValueError(f"Expected input with {len(self.mean)} features, but got {len(X[0])}")

        X_centered = [[X[i][j] - self.mean[j] for j in range(len(self.mean))] for i in range(len(X)) ]        
        projections = []
        for x in X_centered:
            proj = [sum(x[i] * comp[i] for i in range(len(x))) for comp in self.components]
            projections.append(proj)
        return projections
