import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class JumpingJacksAnalyzer:
    def __init__(self):
        self.state = "START" # START, UP
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.feedback = ""
        self.advice = ""
        self.current_rep_quality = {}
        self.mp_pose = mp.solutions.pose
        self.target_muscles = "Calves, Shoulders, Core"

    def analyze(self, landmarks, frame_width, frame_height):
        if not landmarks:
            return self._empty_response("No person detected")

        def get_p(lm): return get_landmark_pixel(landmarks[lm.value], frame_width, frame_height)
        
        l_wrist = get_p(self.mp_pose.PoseLandmark.LEFT_WRIST)
        r_wrist = get_p(self.mp_pose.PoseLandmark.RIGHT_WRIST)
        l_sh = get_p(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        r_sh = get_p(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        l_ank = get_p(self.mp_pose.PoseLandmark.LEFT_ANKLE)
        r_ank = get_p(self.mp_pose.PoseLandmark.RIGHT_ANKLE)

        # Logic: Hands above head and feet apart
        hands_up = l_wrist[1] < l_sh[1] and r_wrist[1] < r_sh[1]
        feet_apart = abs(l_ank[0] - r_ank[0]) > (abs(l_sh[0] - r_sh[0]) * 1.5)

        if self.state == "START":
            if hands_up and feet_apart:
                self.state = "UP"
                self.feedback = "Arms up!"
        elif self.state == "UP":
            if not hands_up and not feet_apart:
                self.rep_count += 1
                self.correct_reps += 1 # Jumping jacks are hard to do "wrong" in this basic logic
                self.state = "START"
                self.feedback = "Rep counted!"
                self.advice = "Keep a steady rhythm."

        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "feedback": self.feedback,
            "advice": self.advice,
            "target_muscles": self.target_muscles,
            "last_rep_score": 100
        }

    def _empty_response(self, msg):
        return {"state": "START", "rep_count": self.rep_count, "correct_reps": self.correct_reps, "incorrect_reps": self.incorrect_reps, "feedback": msg, "advice": "", "target_muscles": self.target_muscles}
