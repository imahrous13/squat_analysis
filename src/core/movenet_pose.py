import cv2
import numpy as np
import mediapipe as mp
import os

class MoveNetLandmark:
    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.z = 0
        self.visibility = visibility

class MoveNetEstimator:
    def __init__(self, model_path=None):
        # Fallback to MediaPipe as TensorFlow is unstable in this environment
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_landmarks = None
        
    def find_pose(self, frame, draw=True):
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks:
            self.last_landmarks = results.pose_landmarks.landmark
            if draw:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
            return frame, results
        
        return frame, None

    def get_landmarks(self):
        return self.last_landmarks
