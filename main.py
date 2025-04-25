import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image
from hand_tracking import get_hand_landmarks

# Class labels (update if needed)
LABELS = sorted([
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
])

# Define hand landmark connections for drawing
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18()
model.fc = nn.Linear(model.fc.in_features, 29)
model.load_state_dict(torch.load("asl_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Start webcam
cap = cv2.VideoCapture(0)
prediction = "no hand"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    landmarks = get_hand_landmarks(frame)

    if landmarks:
        # Convert landmarks to pixel positions
        xs = [int(pt[0] * w) for pt in landmarks]
        ys = [int(pt[1] * h) for pt in landmarks]
        x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
        y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)
        hand_roi = frame[y_min:y_max, x_min:x_max]

        if hand_roi.size > 0:
            try:
                pil_image = Image.fromarray(cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_class = output.argmax(1).item()
                    prediction = LABELS[pred_class]
            except Exception as e:
                print("Error in prediction:", e)

        # Draw landmarks
        for i, (x, y) in enumerate(zip(xs, ys)):
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        for i, j in CONNECTIONS:
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[j], ys[j]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    else:
        prediction = "no hand"

    # Show prediction
    cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("ASL Real-Time Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
