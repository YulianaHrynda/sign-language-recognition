import math
from vector import vector_subtract, vector_dot, vector_magnitude

def angle_between(v1, v2):
    dot = vector_dot(v1, v2)
    norm_product = vector_magnitude(v1) * vector_magnitude(v2)
    return math.acos(dot / (norm_product + 1e-6))

def distance(p1, p2):
    diff = vector_subtract(p1, p2)
    return vector_magnitude(diff)

def extract_feature_vector(landmarks):
    wrist = landmarks[0]
    
    tip_ids = [4, 8, 12, 16, 20]
    angles = []
    y_axis = [0, 1, 0]
    for idx in tip_ids:
        vec = vector_subtract(landmarks[idx], wrist)
        angles.append(angle_between(vec, y_axis))

    distances = [
        distance(landmarks[8], landmarks[12]),
        distance(landmarks[12], landmarks[16]),
        distance(landmarks[16], landmarks[20]),
        distance(landmarks[4], landmarks[8]),
        distance(landmarks[8], landmarks[20])
    ]

    return angles + distances
