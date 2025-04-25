import math
import random
from vector import mean_vector, vector_subtract, vector_magnitude, vector_dot, scalar_divide

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def covariance_matrix(X):
    n = len(X)
    m = len(X[0])
    cov = [[0.0]*m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            s = 0.0
            for k in range(n):
                s += X[k][i]*X[k][j]
            cov[i][j] = s/(n-1)
    return cov

def power_iteration(matrix, tol=1e-6, max_iter=10000):
    n = len(matrix)
    v = [random.random() for _ in range(n)]
    v = scalar_divide(v, vector_magnitude(v))
    for _ in range(max_iter):
        w = [vector_dot(row, v) for row in matrix]
        norm_w = vector_magnitude(w)
        if norm_w == 0:
            break
        v_next = scalar_divide(w, norm_w)
        if vector_magnitude(vector_subtract(v_next, v)) < tol:
            v = v_next
            break
        v = v_next
    return scalar_divide(v, vector_magnitude(v))

class PCA:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.mean = None
        self.components = []

    def fit(self, X):
        self.mean = mean_vector(X)
        X_centered = [[x_ij - self.mean[j] for j, x_ij in enumerate(x_i)] for x_i in X]
        C = covariance_matrix(X_centered)
        self.components = []
        for _ in range(self.n_components):
            v = power_iteration(C)
            v = scalar_divide(v, vector_magnitude(v))
            self.components.append(v)
            Cv = [vector_dot(row, v) for row in C]
            lam = vector_dot(v, Cv)
            m = len(C)
            for i in range(m):
                for j in range(m):
                    C[i][j] -= lam*v[i]*v[j]

    def transform(self, X):
        if len(X[0]) != len(self.mean):
            raise ValueError(f"Expected {len(self.mean)} features, got {len(X[0])}")
        X_centered = [[x_ij - self.mean[j] for j, x_ij in enumerate(x_i)] for x_i in X]
        projections = []
        for x in X_centered:
            projections.append([vector_dot(x, comp) for comp in self.components])
        return projections
