import cv2
from hand_tracking import get_hand_landmarks, normalize_landmarks
from features import extract_feature_vector
from pca import PCA
from kNN import kNN

# Simulated training data (replace with real data later)
X_train = [
    [0.1, 0.4, 0.7, 1.0, 0.9],
    [0.2, 0.3, 0.6, 1.1, 0.8],
    [0.8, 0.2, 0.1, 0.0, 0.1]
]
y_train = ["peace", "peace", "fist"]

# Train PCA + kNN
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)

knn = kNN(k=3)
knn.fit(X_train_pca, y_train)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    landmarks = get_hand_landmarks(frame)
    if landmarks:
        norm_landmarks = normalize_landmarks(landmarks)
        features = extract_feature_vector(norm_landmarks)
        features_pca = pca.transform([features])[0]
        gesture = knn.predict(features_pca)

        cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
