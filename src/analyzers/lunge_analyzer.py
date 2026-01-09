import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class LungeAnalyzer:
    def __init__(self):
        # State Machine
        self.state = "STANDING" # STANDING, DESCENDING, BOTTOM, ASCENDING
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_rep_quality = {}
        self.feedback = ""
        self.advice = ""
        self.prev_state = "STANDING"
        self.state_counter = 0 
        self.state_transition_threshold = 8 # Increased from 3 to prevent overcounting
        
        # Thresholds (Balanced for stability)
        self.stand_threshold = 160 # Increased from 150
        self.descend_threshold = 140 # Lowered from 145
        self.bottom_threshold = 115 # Balanced
        
        # Rep Stats
        self.min_knee_angle = 180
        self.lead_leg = None # "LEFT" or "RIGHT"
        self.rep_start_time = 0
        self.descent_duration = 0
        self.ascent_duration = 0
        self.knee_valgus_flags = 0
        self.hip_tilt_flags = 0
        self.knee_over_toes_flags = 0
        self.torso_lean_flags = 0
        self.frame_count = 0
        
        self.mp_pose = mp.solutions.pose
        self._reset_rep_stats()

    def _reset_rep_stats(self):
        self.min_knee_angle = 180
        self.rep_start_time = time.time()
        self.descent_duration = 0
        self.ascent_duration = 0
        self.knee_valgus_flags = 0
        self.hip_tilt_flags = 0
        self.knee_over_toes_flags = 0
        self.torso_lean_flags = 0
        self.frame_count = 0
        self.lead_leg = None
        self.current_rep_quality = {}

    def analyze(self, landmarks, frame_width, frame_height):
        if not landmarks:
            return self._empty_response("No person detected")

        # Extract Key Landmarks
        def get_p(lm): return get_landmark_pixel(landmarks[lm.value], frame_width, frame_height)
        
        l_hip = get_p(self.mp_pose.PoseLandmark.LEFT_HIP)
        r_hip = get_p(self.mp_pose.PoseLandmark.RIGHT_HIP)
        l_knee = get_p(self.mp_pose.PoseLandmark.LEFT_KNEE)
        r_knee = get_p(self.mp_pose.PoseLandmark.RIGHT_KNEE)
        l_ank = get_p(self.mp_pose.PoseLandmark.LEFT_ANKLE)
        r_ank = get_p(self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        l_sh = get_p(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        r_sh = get_p(self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        l_toe = get_p(self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX)
        r_toe = get_p(self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)

        # 1. Detect View & Active Side
        shoulder_width = abs(l_sh[0] - r_sh[0])
        torso_height = abs((l_sh[1] + r_sh[1])/2 - (l_hip[1] + r_hip[1])/2)
        view = "FRONT"
        if torso_height > 0:
            view = "FRONT" if (shoulder_width / torso_height) > 0.5 else "SIDE"

        # 2. Calculate Angles
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ank)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ank)

        # 2.5 Determine facing direction (Side View)
        facing = "RIGHT"
        if view == "SIDE":
            if landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility > landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility:
                facing = "RIGHT" if l_toe[0] > l_ank[0] else "LEFT"
            else:
                facing = "RIGHT" if r_toe[0] > r_ank[0] else "LEFT"

        # 3. Detect Lead Leg (The one that is in front)
        # Heuristic: The front leg is the one further in the facing direction
        if self.state == "STANDING":
            knee_diff = abs(l_knee_angle - r_knee_angle)
            ankle_dist = abs(l_ank[0] - r_ank[0])
            
            # Lunge requires asymmetry or a split stance
            if knee_diff > 20 or ankle_dist > 30: # Lowered from 25 and 40
                if facing == "RIGHT":
                    self.lead_leg = "LEFT" if l_ank[0] > r_ank[0] else "RIGHT"
                else:
                    self.lead_leg = "LEFT" if l_ank[0] < r_ank[0] else "RIGHT"
            else:
                if min(l_knee_angle, r_knee_angle) < self.descend_threshold:
                    self.feedback = "Squat detected? Switch to Squat mode."
        
        active_knee_angle = l_knee_angle if self.lead_leg == "LEFT" else r_knee_angle
        if self.lead_leg is None: active_knee_angle = min(l_knee_angle, r_knee_angle)

        # 4. Form Checks
        # A. Knee Valgus (Front Knee Caving)
        valgus_detected = False
        if self.lead_leg == "LEFT":
            # If knee is significantly to the right of the hip-ankle line (medial)
            if view == "FRONT" and l_knee[0] > (l_hip[0] + l_ank[0]) / 2 + 35: # Relaxed from 20
                valgus_detected = True
        elif self.lead_leg == "RIGHT":
            if view == "FRONT" and r_knee[0] < (r_hip[0] + r_ank[0]) / 2 - 35: # Relaxed from 20
                valgus_detected = True
        
        if valgus_detected: self.knee_valgus_flags += 1

        # B. Hips Level (Pelvic Tilt)
        hip_diff = abs(l_hip[1] - r_hip[1])
        hip_tilt = hip_diff > (torso_height * 0.20) # Relaxed from 0.15
        if hip_tilt: self.hip_tilt_flags += 1

        # C. Knee Over Toes (Side View)
        knee_over_toes = False
        if view == "SIDE":
            target_knee = l_knee if self.lead_leg == "LEFT" else r_knee
            target_toe = l_toe if self.lead_leg == "LEFT" else r_toe
            if facing == "RIGHT":
                if target_knee[0] > target_toe[0] + 40: knee_over_toes = True # Relaxed from 20
            else:
                if target_knee[0] < target_toe[0] - 40: knee_over_toes = True # Relaxed from 20
        
        if knee_over_toes: 
            self.knee_over_toes_flags += 1
            self.feedback = "⚠️ Knee too far forward!"

        # D. Torso Lean (Side View)
        torso_lean = False
        if view == "SIDE":
            # Calculate angle between shoulder, hip, and a vertical point below hip
            v_point = (l_hip[0] if self.lead_leg == "LEFT" else r_hip[0], 
                       (l_hip[1] if self.lead_leg == "LEFT" else r_hip[1]) + 100)
            target_sh = l_sh if self.lead_leg == "LEFT" else r_sh
            target_hip = l_hip if self.lead_leg == "LEFT" else r_hip
            
            torso_angle = calculate_angle(target_sh, target_hip, v_point)
            # If torso_angle is small, it's upright. If large, it's leaning.
            # 0 is vertical. 30+ is significant lean. (Relaxed from 20)
            if torso_angle > 30:
                torso_lean = True
        
        if torso_lean:
            self.torso_lean_flags += 1
            self.feedback = "⚠️ Keep torso upright!"

        # 5. State Machine
        current_time = time.time()
        
        if self.state == "STANDING":
            if active_knee_angle < self.descend_threshold:
                self.state_counter += 1
                if self.state_counter >= self.state_transition_threshold:
                    self.state = "DESCENDING"
                    self.state_counter = 0
                    self.feedback = "Descending..."
            else:
                self.state_counter = 0

        elif self.state == "DESCENDING":
            self.min_knee_angle = min(self.min_knee_angle, active_knee_angle)
            if active_knee_angle < self.bottom_threshold:
                self.state_counter += 1
                if self.state_counter >= self.state_transition_threshold:
                    self.state = "BOTTOM"
                    self.state_counter = 0
                    self.feedback = "Hold it!"
            elif active_knee_angle > self.descend_threshold + 10:
                self.state = "STANDING"
                self.feedback = "Rep aborted - go deeper"
                self._reset_rep_stats()
            else:
                self.state_counter = 0

        elif self.state == "BOTTOM":
            self.min_knee_angle = min(self.min_knee_angle, active_knee_angle)
            if active_knee_angle > self.bottom_threshold + 15:
                self.state_counter += 1
                if self.state_counter >= self.state_transition_threshold:
                    self.state = "ASCENDING"
                    self.state_counter = 0
                    self.feedback = "Push up!"
            else:
                self.state_counter = 0

        elif self.state == "ASCENDING":
            if active_knee_angle > self.stand_threshold:
                self.state_counter += 1
                if self.state_counter >= self.state_transition_threshold:
                    self.rep_count += 1
                    self._score_rep()
                    self.state = "STANDING"
                    self.state_counter = 0
                    self._reset_rep_stats()
            else:
                self.state_counter = 0

        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "l_knee_angle": l_knee_angle,
            "r_knee_angle": r_knee_angle,
            "lead_leg": self.lead_leg,
            "feedback": self.feedback,
            "advice": self.advice,
            "last_rep_score": self.current_rep_quality.get("score", 0),
            "reasons": self.current_rep_quality.get("faults", []),
            "view": view,
            "target_muscles": "Quadriceps, Glutes, Hamstrings"
        }

    def _score_rep(self):
        score = 100
        faults = []
        
        # 1. Depth (Critical) - Balanced
        if self.min_knee_angle > 130: 
            score -= 35 # Balanced from 25/40
            faults.append("Depth: Too Shallow")
        elif self.min_knee_angle > 115:
            score -= 15 # Balanced from 10/20
            faults.append("Depth: Slightly Shallow")
            
        # 2. Knee Stability (Critical) - Balanced tolerance
        if self.knee_valgus_flags > 12: # Balanced from 5/20
            score -= 25 
            faults.append("Form: Knee Caving In")
            
        # 3. Hip Level
        if self.hip_tilt_flags > 15: # Balanced from 8/20
            score -= 10 
            faults.append("Form: Hips Not Level")
            
        # 4. Knee Over Toes (Critical)
        if self.knee_over_toes_flags > 15: # Balanced from 8/20
            score -= 25 
            faults.append("Form: Knee Over Toes")
            
        # 5. Torso Lean
        if self.torso_lean_flags > 15: # Balanced from 8/20
            score -= 15 
            faults.append("Form: Leaning Forward")
            
        # 6. Balance/Wobble
        if self.hip_tilt_flags > 25 or self.knee_valgus_flags > 25:
            score -= 15
            faults.append("Form: Significant Wobble")

        self.current_rep_quality = {
            "score": max(0, score),
            "faults": faults
        }

        # Passing threshold is 75 (Balanced)
        if score >= 75:
            self.correct_reps += 1
            self.advice = "Great form! Keep it up."
            self.feedback = f"Rep {self.rep_count}: Correct!"
        else:
            self.incorrect_reps += 1
            self.advice = self._get_advice(faults)
            self.feedback = f"Rep {self.rep_count}: Incorrect - {', '.join(faults)}"

    def _get_advice(self, faults):
        advice_map = {
            "Depth: Too Shallow": "Lower your hips until your front thigh is parallel to the floor.",
            "Depth: Slightly Shallow": "Almost there! Just a bit more depth for full engagement.",
            "Form: Knee Caving In": "Push your front knee slightly outward to keep it aligned with your foot.",
            "Form: Hips Not Level": "Try to keep your pelvis level. Avoid letting one hip drop lower than the other.",
            "Form: Knee Over Toes": "Don't bring the knee forward over the toes. Keep your weight centered.",
            "Form: Leaning Forward": "Maintain an upright torso. Don't lean your torso forward."
        }
        for fault in faults:
            if fault in advice_map: return advice_map[fault]
        return "Focus on balance and controlled movement."

    def _empty_response(self, msg):
        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "feedback": msg,
            "advice": "",
            "last_rep_score": 0
        }
