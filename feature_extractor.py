import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from hand_tracking import get_hand_landmarks, normalize_landmarks
from features import extract_feature_vector

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def equalize_histogram(image):
    if len(image.shape) == 3:  # If the image is in color
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(image)

def preprocess_image(image):
    image = equalize_histogram(image)  # Enhance contrast
    image = adjust_gamma(image, gamma=1.2)  # Adjust brightness
    return image

def extract_features_and_save_to_json(dataset_path, output_filename="features_and_labels.json"):
    features_list = []  # List to store all features
    labels_list = []  # List to store corresponding labels

    for label in tqdm(os.listdir(dataset_path), desc="Processing Directories"):
        label_path = os.path.join(dataset_path, label)

        if not os.path.isdir(label_path):
            continue

        for filename in tqdm(os.listdir(label_path), desc=f"Processing {label}", leave=False):
            img_path = os.path.join(label_path, filename)

            if img_path.endswith(('.jpg', '.jpeg', '.png')):
                frame = cv2.imread(img_path)
                frame = cv2.flip(frame, 1)  # Flip the image horizontally

                frame = preprocess_image(frame)

                landmarks = get_hand_landmarks(frame)

                if landmarks:
                    norm_landmarks = normalize_landmarks(landmarks)
                    features = extract_feature_vector(norm_landmarks)

                    if features:
                        features_list.append(features)
                        labels_list.append(label)
                else:
                    continue

    if len(features_list) == 0:
        raise ValueError("No features extracted, check your dataset and hand landmark extraction.")

    data = {
        "features": features_list,
        "labels": labels_list
    }

    try:
        with open(output_filename, "w") as json_file:
            json.dump(data, json_file)
        print(f"Features and labels saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")


def main():
    dataset_path = "./new_dataset"
    extract_features_and_save_to_json(dataset_path)

if __name__ == "__main__":
    main()
