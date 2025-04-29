# Sign Language Recognition
Real-time sign language letter recognition using computer vision, PCA, and k-NN.
The system can build sentences based on detected gestures, including whitespace recognition ("space", "del" gestures).

Authors:\
Pavlosiuk Roman - https://github.com/gllekkoff \
Hrynda Yuliana - https://github.com/YulianaHrynda \
Denysova Iryna - https://github.com/Shnapa

### Installation
To use our project you need to create venv and activate venv:
```sh
python3 -m venv myenv
```

If you are using Linux:
```sh
source myenv/bin/activate
```

If you are using Windows:
```sh
myenv/bin/activate
```

Then install needed libraries for project:
```sh
pip install -r requirements.txt
```

### How to run

In order to run project, after completing steps described above use this:
```sh
python3 main.py
```

### How to stop running

To stop code running simply press "q" button, or press Ctrl+C or Ctrl+D in terminal

### Project Structure

| File/Folder            | Description |
|-------------------------|-------------|
| `main.py`               | Main app for real-time gesture recognition |
| `hand_tracking.py`      | Functions for detecting and normalizing hand landmarks |
| `features.py`           | Feature extraction from hand poses |
| `pca.py`                | Principal Component Analysis implementation |
| `kNN.py`                | k-Nearest Neighbors custom classifier |
| `features_and_labels.json` | Dataset with extracted features and labels |

### Demo

(Insert your GIF or sample screenshots here!)

The system:

    Detects hand landmarks.

    Classifies letters or "space" gesture.

    Builds a live sentence on screen based on recognized gestures!

### Features

    Real-time webcam input processing.

    Accurate gesture detection with MediaPipe.

    Custom feature extraction and dimensionality reduction.
    
    Real-time sentence construction (space and del included).

    Lightweight and easy to extend.

### Dataset

We trained and tested our model using the ASL Alphabet Dataset available on Kaggle.
This dataset contains labeled images of hand signs for each letter of the American Sign Language (ASL) alphabet.

You can download it [here](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
