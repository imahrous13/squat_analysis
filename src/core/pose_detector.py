import cv2
import mediapipe as mp
from src.core.pose_config import create_mediapipe_pose

class PoseDetector:
    def __init__(self, mode=False, complexity=None, smooth_landmarks=None, 
                 enable_segmentation=None, smooth_segmentation=True, 
                 detection_con=None, track_con=None):
        """
        Initializes the MediaPipe Pose object using centralized config defaults.
        """
        self.mp_pose = mp.solutions.pose
        overrides = {}
        if mode:
            overrides["static_image_mode"] = mode
        if complexity is not None:
            overrides["model_complexity"] = complexity
        if smooth_landmarks is not None:
            overrides["smooth_landmarks"] = smooth_landmarks
        if enable_segmentation is not None:
            overrides["enable_segmentation"] = enable_segmentation
        if detection_con is not None:
            overrides["min_detection_confidence"] = detection_con
        if track_con is not None:
            overrides["min_tracking_confidence"] = track_con
        self.pose = create_mediapipe_pose(self.mp_pose, mode="default", **overrides)
        self.mp_drawing = mp.solutions.drawing_utils
        self.results = None

    def find_pose(self, img, draw=True):
        """
        Processes the image to find pose landmarks.
        
        Args:
            img: Input image (BGR).
            draw (bool): Whether to draw landmarks on the image.
            
        Returns:
            img: Image with landmarks drawn (if draw=True).
            results: MediaPipe pose results.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        
        if self.results.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(
                img, 
                self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
        return img, self.results

    def get_landmarks(self):
        """
        Returns the raw landmarks if found.
        """
        if self.results and self.results.pose_landmarks:
            return self.results.pose_landmarks.landmark
        return None
