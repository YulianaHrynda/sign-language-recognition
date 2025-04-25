import cv2
from hand_tracking import get_hand_landmarks, normalize_landmarks
from features import extract_feature_vector
from pca import PCA
from kNN import kNN
X_train = [
    [2.51, 2.85, 2.94, 2.97, 2.90, 0.10, 0.09, 0.22, 0.43, 0.37],
    [2.48, 2.81, 2.92, 2.95, 2.88, 0.12, 0.08, 0.21, 0.42, 0.36],
    [2.01, 1.90, 1.95, 2.00, 2.02, 0.05, 0.07, 0.12, 0.25, 0.20]
]
y_train = ["peace", "fuck", "fist"]

pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)

knn = kNN(k=2)
knn.fit(X_train_pca, y_train)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

cap = cv2.VideoCapture(0)
last_prediction = "No hand"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    landmarks = get_hand_landmarks(frame)

    if landmarks:
        norm_landmarks = normalize_landmarks(landmarks)
        features = extract_feature_vector(norm_landmarks)
        features_pca = pca.transform([features])[0]
        prediction = knn.predict(features_pca)

        for idx, lm in enumerate(landmarks):
            x = int(lm[0] * w)
            y = int(lm[1] * h)
            cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)

        for i, j in CONNECTIONS:
            x1, y1 = int(landmarks[i][0] * w), int(landmarks[i][1] * h)
            x2, y2 = int(landmarks[j][0] * w), int(landmarks[j][1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    prediction = "no hand" if not landmarks else prediction

    cv2.putText(frame, f"Gesture: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    
    cv2.imshow("Real-Time Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
