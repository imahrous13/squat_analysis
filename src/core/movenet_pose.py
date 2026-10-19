import cv2
import numpy as np
import os
from src.core.pose_config import create_mediapipe_pose

class MoveNetLandmark:
    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.z = 0
        self.visibility = visibility

class MoveNetEstimator:
    def __init__(self, model_path=None):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = create_mediapipe_pose(self.mp_pose, mode="default")
        self.last_landmarks = None
        self._active_side_is_left = None  # sticky side for visibility boost
        self._side_switch_margin = 0.15   # hysteresis to prevent side flipping
        
    def find_pose(self, frame, draw=True, enhance_side_view=True):
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks:
            self.last_landmarks = results.pose_landmarks.landmark
            
            # Enhance side-view detection by focusing on the visible side
            if enhance_side_view:
                self._enhance_side_view_landmarks()
            
            if draw:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
            return frame, results
        
        return frame, None

    def _enhance_side_view_landmarks(self):
        """
        Gently boost the more-visible side; sticky hysteresis prevents frame-to-frame flips.
        """
        if not self.last_landmarks:
            return
        
        left_shoulder_vis = self.last_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
        right_shoulder_vis = self.last_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
        left_hip_vis = self.last_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].visibility
        right_hip_vis = self.last_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
        
        left_avg_vis = (left_shoulder_vis + left_hip_vis) / 2
        right_avg_vis = (right_shoulder_vis + right_hip_vis) / 2
        
        if self._active_side_is_left is None:
            self._active_side_is_left = left_avg_vis >= right_avg_vis
        else:
            margin = self._side_switch_margin
            if self._active_side_is_left:
                if right_avg_vis > left_avg_vis + margin:
                    self._active_side_is_left = False
            else:
                if left_avg_vis > right_avg_vis + margin:
                    self._active_side_is_left = True
        
        active_side_is_left = self._active_side_is_left
        
        left_indices = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            self.mp_pose.PoseLandmark.LEFT_HEEL.value,
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
        ]
        
        right_indices = [
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            self.mp_pose.PoseLandmark.RIGHT_HEEL.value,
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
        ]
        
        boost_factor = 1.08
        reduce_factor = 0.92
        
        if active_side_is_left:
            for idx in left_indices:
                self.last_landmarks[idx].visibility = min(1.0, self.last_landmarks[idx].visibility * boost_factor)
            for idx in right_indices:
                self.last_landmarks[idx].visibility *= reduce_factor
        else:
            for idx in right_indices:
                self.last_landmarks[idx].visibility = min(1.0, self.last_landmarks[idx].visibility * boost_factor)
            for idx in left_indices:
                self.last_landmarks[idx].visibility *= reduce_factor

    def get_landmarks(self):
        return self.last_landmarks
