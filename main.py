import cv2
import torch
import torchvision.transforms as transforms
from model import SimpleCNN
from hand_tracking import get_hand_landmarks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=29)
model.load_state_dict(torch.load("asl_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

LABELS = sorted([
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    landmarks = get_hand_landmarks(frame)

    prediction = "no hand"
    if landmarks:
        xs = [int(pt[0] * w) for pt in landmarks]
        ys = [int(pt[1] * h) for pt in landmarks]
        x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
        y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)
        hand_roi = frame[y_min:y_max, x_min:x_max]

        if hand_roi.size > 0:
            try:
                input_image = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                input_tensor = transform(Image.fromarray(input_image)).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    prediction = LABELS[output.argmax().item()]
            except Exception as e:
                print("Error processing frame:", e)

        for x, y in zip(xs, ys):
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    cv2.putText(frame, f"Gesture: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Real-Time Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
