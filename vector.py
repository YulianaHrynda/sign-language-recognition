import math

def vector_subtract(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]

def vector_magnitude(v):
    return math.sqrt(sum([v[i] ** 2 for i in range(len(v))]))

def vector_dot(v1, v2):
    return sum([v1[i] * v2[i] for i in range(len(v1))])

def scalar_divide(v, scalar):
    return [x / scalar for x in v]

def mean_vector(vectors):
    n = len(vectors[0])
    mean = [0] * n
    for vec in vectors:
        for i in range(n):
            mean[i] += vec[i]
    return [x / len(vectors) for x in mean]
