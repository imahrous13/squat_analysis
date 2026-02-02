import cv2
import numpy as np
import os

class MoveNetLandmark:
    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.z = 0
        self.visibility = visibility

class MoveNetEstimator:
    def __init__(self, model_path=None):
        # Optimized MediaPipe configuration for side-view exercises
        from mediapipe.solutions import pose as mp_pose
        from mediapipe.solutions import drawing_utils as mp_drawing
        
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,              # Increased from 0 to 1 for better accuracy
            smooth_landmarks=True,           # Enable smoothing to reduce jitter
            enable_segmentation=False,       # Disable segmentation for performance
            min_detection_confidence=0.6,    # Increased from 0.5 for more reliable detection
            min_tracking_confidence=0.7      # Increased from 0.5 for smoother tracking
        )
        self.last_landmarks = None
        
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
        Enhance side-view detection by boosting the visibility of the active side
        and reducing noise from the far side (side facing away from camera).
        """
        if not self.last_landmarks:
            return
        
        # Determine which side is more visible (higher visibility scores)
        left_shoulder_vis = self.last_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
        right_shoulder_vis = self.last_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
        left_hip_vis = self.last_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].visibility
        right_hip_vis = self.last_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
        
        # Calculate average visibility for each side
        left_avg_vis = (left_shoulder_vis + left_hip_vis) / 2
        right_avg_vis = (right_shoulder_vis + right_hip_vis) / 2
        
        # Determine active side (more visible side)
        active_side_is_left = left_avg_vis > right_avg_vis
        
        # Define landmark indices for each side
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
        
        # Boost active side visibility and reduce far side
        boost_factor = 1.15  # Boost active side by 15%
        reduce_factor = 0.85  # Reduce far side by 15%
        
        if active_side_is_left:
            # Boost left side
            for idx in left_indices:
                self.last_landmarks[idx].visibility = min(1.0, self.last_landmarks[idx].visibility * boost_factor)
            # Reduce right side
            for idx in right_indices:
                self.last_landmarks[idx].visibility *= reduce_factor
        else:
            # Boost right side
            for idx in right_indices:
                self.last_landmarks[idx].visibility = min(1.0, self.last_landmarks[idx].visibility * boost_factor)
            # Reduce left side
            for idx in left_indices:
                self.last_landmarks[idx].visibility *= reduce_factor

    def get_landmarks(self):
        return self.last_landmarks
