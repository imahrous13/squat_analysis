import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class PlankAnalyzer:
    def __init__(self):
        self.state = "SEARCHING" # SEARCHING, HOLDING, BREAK
        self.total_good_time = 0.0
        self.last_update_time = time.time()
        
        self.feedback = ""
        self.advice = ""
        self.mp_pose = mp.solutions.pose
        self.target_muscles = "Core, Shoulders"
        self.correct_reps = 0 # Shows Total Good Seconds
        self.incorrect_reps = 0 # Shows Form Breaks
        
        self.prev_status = "GOOD" # GOOD, BAD
        self.status_debounce = 0
        
    def analyze(self, landmarks, frame_width, frame_height):
        if not landmarks:
            return self._empty_response("No person detected")

        def get_p(lm): return get_landmark_pixel(landmarks[lm.value], frame_width, frame_height)
        
        l_sh = get_p(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        l_hip = get_p(self.mp_pose.PoseLandmark.LEFT_HIP)
        l_ank = get_p(self.mp_pose.PoseLandmark.LEFT_ANKLE)
        
        # 1. Orientation Check (Are they horizontal?)
        dx = abs(l_sh[0] - l_ank[0])
        dy = abs(l_sh[1] - l_ank[1])
        is_horizontal = dx > dy
        
        # Reset if not horizontal (e.g. standing up)
        if not is_horizontal:
             self.state = "SEARCHING"
             self.feedback = "Get into plank position"
             return self._response()

        # 2. Form Check
        # Body line angle (Shoulder-Hip-Ankle should be ~180)
        body_angle = calculate_angle(l_sh, l_hip, l_ank)
        
        # Allow range 160-200 (180 is straight)
        is_good_form = 160 < body_angle < 200
        
        current_time = time.time()
        dt = current_time - self.last_update_time
        # Cap dt to avoid large jumps if frame rate drops or first frame
        if dt > 0.5: dt = 0.0 
        self.last_update_time = current_time
        
        if is_good_form:
            self.state = "HOLDING"
            # Transition from BAD to GOOD
            if self.prev_status == "BAD":
                self.prev_status = "GOOD"
                self.feedback = "Good form!"
            
            # Accumulate time
            self.total_good_time += dt
            self.feedback = f"Holding... {int(self.total_good_time)}s"
            self.advice = "Keep core tight."
            
        else:
            self.state = "BREAK"
            # Form Break
            if self.prev_status == "GOOD":
                 # Debounce could be added here
                 self.incorrect_reps += 1
                 self.prev_status = "BAD"
            
            self.feedback = "Straighten back!"
            if body_angle <= 160: self.advice = "Hips likely too high or sagging."
            else: self.advice = "Check hip alignment."

        self.correct_reps = int(self.total_good_time)

        return self._response()

    def _response(self):
         return {
            "state": self.state,
            "rep_count": self.correct_reps, 
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "feedback": self.feedback,
            "advice": self.advice,
            "target_muscles": self.target_muscles,
            "last_rep_score": 100 if self.prev_status == "GOOD" else 0
        }

    def _empty_response(self, msg):
        return {"state": "SEARCHING", "rep_count": 0, "correct_reps": 0, "incorrect_reps": 0, "feedback": msg, "advice": "", "target_muscles": self.target_muscles}
