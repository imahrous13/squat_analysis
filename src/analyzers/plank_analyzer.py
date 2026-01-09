import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class PlankAnalyzer:
    def __init__(self):
        self.state = "HOLDING"
        self.start_time = None
        self.hold_duration = 0
        self.feedback = ""
        self.advice = ""
        self.mp_pose = mp.solutions.pose
        self.target_muscles = "Core, Shoulders"
        self.correct_reps = 0 # For plank, we can treat 10s as a "rep" or just show time
        self.incorrect_reps = 0

    def analyze(self, landmarks, frame_width, frame_height):
        if not landmarks:
            return self._empty_response("No person detected")

        def get_p(lm): return get_landmark_pixel(landmarks[lm.value], frame_width, frame_height)
        
        l_sh = get_p(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        l_hip = get_p(self.mp_pose.PoseLandmark.LEFT_HIP)
        l_ank = get_p(self.mp_pose.PoseLandmark.LEFT_ANKLE)
        
        # Body line angle (Shoulder-Hip-Ankle should be ~180)
        body_angle = calculate_angle(l_sh, l_hip, l_ank)
        
        is_planking = body_angle > 160 # Relatively straight
        
        if is_planking:
            if self.start_time is None:
                self.start_time = time.time()
            self.hold_duration = int(time.time() - self.start_time)
            self.feedback = f"Holding: {self.hold_duration}s"
            self.advice = "Keep your core tight and back flat."
        else:
            self.start_time = None
            self.feedback = "Straighten your back!"
            self.advice = "Your hips are too high or too low."

        return {
            "state": "PLANK",
            "rep_count": self.hold_duration, # Use duration as rep count for display
            "correct_reps": self.hold_duration,
            "incorrect_reps": 0,
            "feedback": self.feedback,
            "advice": self.advice,
            "target_muscles": self.target_muscles,
            "last_rep_score": 100 if is_planking else 0
        }

    def _empty_response(self, msg):
        return {"state": "PLANK", "rep_count": 0, "correct_reps": 0, "incorrect_reps": 0, "feedback": msg, "advice": "", "target_muscles": self.target_muscles}
