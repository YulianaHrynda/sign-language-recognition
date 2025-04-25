import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from hand_tracking import get_hand_landmarks, normalize_landmarks
from features import extract_feature_vector

def adjust_gamma(image, gamma=1.0):
    """
    Perform gamma correction to adjust the brightness of the image.
    
    :param image: The input image.
    :param gamma: Gamma value. A value greater than 1 will make the image brighter.
    :return: Gamma corrected image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def equalize_histogram(image):
    """
    Perform histogram equalization to improve image contrast.
    
    :param image: The input image.
    :return: Equalized image.
    """
    if len(image.shape) == 3:  # If the image is in color
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(image)

def preprocess_image(image):
    """
    Preprocess the image to improve lighting and contrast before processing.
    
    :param image: The input image.
    :return: Preprocessed image.
    """
    image = equalize_histogram(image)  # Enhance contrast
    image = adjust_gamma(image, gamma=1.2)  # Adjust brightness
    return image

def extract_features_and_save_to_json(dataset_path, output_filename="features_and_labels.json"):
    """
    Extract features from the dataset and save them to a JSON file.

    :param dataset_path: Path to the dataset directory
    :param output_filename: Output filename for the JSON (default: "features_and_labels.json")
    """
    X_train = {}
    y_train = []

    # Use tqdm for the progress bar over directories
    for label in tqdm(os.listdir(dataset_path), desc="Processing Directories"):
        label_path = os.path.join(dataset_path, label)

        if not os.path.isdir(label_path):
            continue
        
        features_list = []  # To collect features for each class

        # Use tqdm for the progress bar over files in each directory
        for filename in tqdm(os.listdir(label_path), desc=f"Processing {label}", leave=False):
            img_path = os.path.join(label_path, filename)

            if img_path.endswith(('.jpg', '.jpeg', '.png')):
                frame = cv2.imread(img_path)

                # Preprocess the image to adjust lighting and contrast
                frame = preprocess_image(frame)

                landmarks = get_hand_landmarks(frame)

                if landmarks:
                    # Ensure landmarks were successfully detected
                    norm_landmarks = normalize_landmarks(landmarks)
                    features = extract_feature_vector(norm_landmarks)
                    
                    if features:
                        features_list.append(features)
                    else:
                        # Skip if no features are extracted
                        continue
                else:
                    # Skip if no landmarks detected
                    continue
        
        # If features for the current label were collected, compute the mean
        if features_list:
            # Convert the list of features to a numpy array
            features_array = np.array(features_list)
            # Compute the mean of the features for the current class
            mean_features = np.mean(features_array, axis=0)
            X_train[label] = mean_features.tolist()
            y_train.append(label)

    if len(X_train) == 0:
        raise ValueError("No features extracted, check your dataset and hand landmark extraction.")

    data = {
        "features": X_train,
        "labels": y_train
    }

    try:
        with open(output_filename, "w") as json_file:
            json.dump(data, json_file)
        print(f"Features and labels saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")


def main():
    dataset_path = "./dataset"  # Replace with your dataset directory
    extract_features_and_save_to_json(dataset_path)

if __name__ == "__main__":
    main()
