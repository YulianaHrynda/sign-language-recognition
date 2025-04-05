import math
from vector import vector_subtract, vector_dot, vector_magnitude

def extract_feature_vector(landmarks):
    wrist = landmarks[0]
    tip_ids = [4, 8, 12, 16, 20]
    feature_vector = []

    for idx in tip_ids:
        vec = vector_subtract(landmarks[idx], wrist)
        y_axis = [0, 1, 0]
        dot = vector_dot(vec, y_axis)
        angle = math.acos(dot / (vector_magnitude(vec) * vector_magnitude(y_axis) + 1e-6))
        feature_vector.append(angle)
    return feature_vector
