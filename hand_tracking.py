import cv2
import mediapipe as mp
from vector import vector_subtract, vector_magnitude, scalar_divide

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def get_hand_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            return landmarks
    return None

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    shifted = [vector_subtract(lm, wrist) for lm in landmarks]
    max_dist = max([vector_magnitude(v) for v in shifted])
    scaled = [scalar_divide(v, max_dist) for v in shifted]
    return scaled
