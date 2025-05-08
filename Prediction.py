# Prediction.py

import os
import numpy as np
import torch
import mediapipe as mp
from model import StationSignLanguageModel
from skeleton import image_process, keypoint_extraction


class StationSignRecognition:
    def __init__(self, model_path: str, data_path: str, cooldown_threshold: int = 20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actions = np.array(os.listdir(data_path))
        self.cooldown_threshold = cooldown_threshold
        self.cooldown_frames = 0
        self.keypoints = []
        self.last_prediction = None

        # Load the trained model
        input_channels = 126  # Number of keypoints (126)
        num_classes = len(self.actions)
        self.model = StationSignLanguageModel(input_dim=input_channels, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def process_frame(self, frame):
        """
        Process a single frame to extract keypoints and make predictions.

        :param frame: A single frame from a video or camera.
        :return: A list of top 3 predictions with their confidence scores.
        """
        results = image_process(frame, self.holistic)

        # Check if a hand is detected
        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks
        if not hand_detected:
            return []

        # Extract keypoints
        self.keypoints.append(keypoint_extraction(results))

        # Predict every 25 frames if cooldown is not active
        if len(self.keypoints) == 25 and self.cooldown_frames == 0:
            # Convert keypoints to PyTorch tensor
            keypoints_tensor = torch.tensor(self.keypoints, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.keypoints = []

            # Make prediction
            with torch.no_grad():
                prediction = self.model(keypoints_tensor)
                prediction = torch.softmax(prediction, dim=1).cpu().numpy()[0]

            # Get top 3 predictions
            top_indices = np.argsort(prediction)[-3:][::-1]
            predictions = [(self.actions[i], float(prediction[i])) for i in top_indices]

            # Print the top predictions
            print(f"\nTop predictions:")
            for i in range (3):
                print(f"{predictions[i][0]} {predictions[i][1] * 100:.2f}%")

            # Set cooldown and return predictions
            self.cooldown_frames = self.cooldown_threshold
            self.last_prediction = predictions[0][0]
            return predictions

        # Decrement the cooldown counter
        self.cooldown_frames = max(0, self.cooldown_frames - 1)
        return []

    def release(self):
        self.holistic.close()