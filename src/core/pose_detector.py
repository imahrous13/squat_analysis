import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True, 
                 enable_segmentation=False, smooth_segmentation=True, 
                 detection_con=0.5, track_con=0.5):
        """
        Initializes the MediaPipe Pose object.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=mode,
            model_complexity=complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
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
