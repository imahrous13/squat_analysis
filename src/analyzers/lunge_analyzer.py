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
        self.state_transition_threshold = 3  # Reduced from 4 for faster response 
        
        # Thresholds (Optimized for accuracy)
        self.stand_threshold = 150  # Balanced - not too strict, not too lenient
        self.descend_threshold = 140  # Detect descent early
        self.bottom_threshold = 110  # Require proper depth to prevent false positives 
        
        # Rep Stats
        self.min_knee_angle = 180
        self.lead_leg = None 
        self.rep_start_time = 0
        self.last_state_change_time = 0
        self.knee_valgus_flags = 0
        self.hip_tilt_flags = 0
        self.knee_over_toes_flags = 0
        self.torso_lean_flags = 0
        
        self.mp_pose = mp.solutions.pose
        self._reset_rep_stats()

    def _reset_rep_stats(self):
        self.min_knee_angle = 180
        self.rep_start_time = time.time()
        self.last_state_change_time = time.time()
        self.knee_valgus_flags = 0
        self.hip_tilt_flags = 0
        self.knee_over_toes_flags = 0
        self.torso_lean_flags = 0
        self.lead_leg = None
        self.rep_frame_count = 0 # Track total active frames for ratio-based scoring
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
        
        # KEY FIX: Always track the WORKING knee (the one bending)
        # This prevents measuring the straight back leg and failing the rep
        active_knee_angle = min(l_knee_angle, r_knee_angle)

        # 2.5 Determine facing direction (Side View)
        facing = "RIGHT"
        if view == "SIDE":
            # Better facing detection using nose or toes
            if landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility > landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility:
                facing = "RIGHT" if l_toe[0] > l_ank[0] else "LEFT"
            else:
                facing = "RIGHT" if r_toe[0] > r_ank[0] else "LEFT"

        # 3. Detect Lead Leg (Dynamic)
        # If we haven't found the lead leg yet, or we want to confirm it during descent
        if self.lead_leg is None or self.state == "DESCENDING":
             # In a lunge, the lead leg is usually the one with the more acute knee angle in early descent
             # OR the one with the forward foot in side view
             if view == "SIDE":
                 if facing == "RIGHT":
                     if l_toe[0] > r_toe[0] + 20: self.lead_leg = "LEFT"
                     elif r_toe[0] > l_toe[0] + 20: self.lead_leg = "RIGHT"
                 else:
                     if l_toe[0] < r_toe[0] - 20: self.lead_leg = "LEFT"
                     elif r_toe[0] < l_toe[0] - 20: self.lead_leg = "RIGHT"
             else:
                 # Front view: Identify by which ankle is 'lower' on screen? No, perspective.
                 # Identify by which knee is bending more?
                 if abs(l_knee_angle - r_knee_angle) > 15:
                     self.lead_leg = "LEFT" if l_knee_angle < r_knee_angle else "RIGHT"

        if self.state == "STANDING" and active_knee_angle < self.descend_threshold:
             self.feedback = "Descending..."

        # 4. Form Checks
        # relative_scale helps adapt thresholds to user's distance from camera
        relative_scale = torso_height if torso_height > 0 else 100
        
        # A. Knee Valgus
        valgus_detected = False
        valgus_threshold = relative_scale * 0.15 # ~15% of torso height (approx 30-40px for avg user)
        
        if self.lead_leg == "LEFT":
            if view == "FRONT" and l_knee[0] > (l_hip[0] + l_ank[0]) / 2 + valgus_threshold: valgus_detected = True
        elif self.lead_leg == "RIGHT":
            if view == "FRONT" and r_knee[0] < (r_hip[0] + r_ank[0]) / 2 - valgus_threshold: valgus_detected = True
        if valgus_detected: self.knee_valgus_flags += 1

        # B. Hips Level
        hip_diff = abs(l_hip[1] - r_hip[1])
        if hip_diff > (relative_scale * 0.20): self.hip_tilt_flags += 1

        # C. Knee Over Toes
        knee_over_toes = False
        kot_threshold = relative_scale * 0.15 # Relaxed tolerance (Balanced)
        
        if view == "SIDE":
            target_knee = l_knee if self.lead_leg == "LEFT" else r_knee
            target_toe = l_toe if self.lead_leg == "LEFT" else r_toe
            
            if facing == "RIGHT":
                if target_knee[0] > target_toe[0] + kot_threshold: knee_over_toes = True
            else:
                if target_knee[0] < target_toe[0] - kot_threshold: knee_over_toes = True
        if knee_over_toes: self.knee_over_toes_flags += 1

        # D. Torso Lean
        torso_lean = False
        if view == "SIDE":
            # Point straight up from hip
            v_point = (l_hip[0] if self.lead_leg == "LEFT" else r_hip[0], 
                       (l_hip[1] if self.lead_leg == "LEFT" else r_hip[1]) - 100) # Negative Y is up
            
            target_sh = l_sh if self.lead_leg == "LEFT" else r_sh
            target_hip = l_hip if self.lead_leg == "LEFT" else r_hip
            
            # Angle against vertical
            torso_angle = calculate_angle(target_sh, target_hip, v_point)
            if torso_angle > 20: torso_lean = True # Stricter (was 35)
        if torso_lean: self.torso_lean_flags += 1

        # 5. State Machine
        if self.state != "STANDING":
            self.rep_frame_count += 1

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
                    self.last_state_change_time = time.time()
                    self.state_counter = 0
                    self.feedback = "Hold it!"
            elif active_knee_angle > self.descend_threshold + 10:
                # Check if it was a shallow rep attempt instead of just noise
                if self.min_knee_angle < 155: 
                    self._score_rep()
                
                self.state = "STANDING" # Abort/Reset
                self.last_state_change_time = time.time()
                self._reset_rep_stats()
            else:
                self.state_counter = 0

        elif self.state == "BOTTOM":
            self.min_knee_angle = min(self.min_knee_angle, active_knee_angle)
            if active_knee_angle > self.bottom_threshold + 15:
                self.state_counter += 1
                if self.state_counter >= self.state_transition_threshold:
                    self.state = "ASCENDING"
                    self.last_state_change_time = time.time()
                    self.state_counter = 0
                    self.feedback = "Push up!"
            else:
                self.state_counter = 0

        elif self.state == "ASCENDING":
            # Timeout / Stuck Prevention
            time_in_state = time.time() - self.last_state_change_time
            is_stuck_upright = (time_in_state > 4.0 and active_knee_angle > 140)
            
            if active_knee_angle > self.stand_threshold or is_stuck_upright:
                self.state_counter += 1
                if self.state_counter >= self.state_transition_threshold or is_stuck_upright:
                    self._score_rep() # Finalize and Count
                    self.state = "STANDING"
                    self.last_state_change_time = time.time()
                    self.state_counter = 0
                    self._reset_rep_stats()
            else:
                self.state_counter = 0

        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "lead_leg": self.lead_leg,
            "feedback": self.feedback,
            "advice": self.advice,
            "last_rep_score": self.current_rep_quality.get("score", 0),
            "reasons": self.current_rep_quality.get("faults", []),
            "view": view,
            "target_muscles": "Quadriceps, Glutes",
            "l_knee_angle": l_knee_angle,
            "r_knee_angle": r_knee_angle
        }

    def _score_rep(self):
        score = 100
        deductions = []
        
        total_frames = max(1, self.rep_frame_count)
        
        # Helper to check ratio - OPTIMIZED thresholds
        def check_fault(flags, threshold_ratio=0.35):
            # Must be bad for at least 6 frames AND >35% of total time
            return flags > 6 and (flags / total_frames) > threshold_ratio
        
        # 1. Depth - REASONABLE
        if self.min_knee_angle > 145:  # Reduced from 150 - more forgiving
            score -= 25  # Reduced penalty
            deductions.append("Too Shallow")
        elif self.min_knee_angle > 130:  
            score -= 8  
            deductions.append("Slightly Shallow")
            
        # 2. Knee Stability - MODERATE
        if check_fault(self.knee_valgus_flags, 0.35):  
            score -= 12 
            deductions.append("Knee Caving")
            
        # 3. Hip Level - LENIENT (minor issue)
        if check_fault(self.hip_tilt_flags, 0.45):  
            score -= 5 
            deductions.append("Hips Not Level")
            
        # 4. Knee Over Toes - MODERATE (important but common)
        if check_fault(self.knee_over_toes_flags, 0.35):  
            score -= 12
            deductions.append("Knee Over Toes")
            
        # 5. Torso Lean - MODERATE (important for form)
        if check_fault(self.torso_lean_flags, 0.35):  
            score -= 12
            deductions.append("Leaning Forward")

        self.current_rep_quality = {
            "score": max(0, score),
            "faults": deductions
        }

        # Determine Correct/Incorrect based on CRITICAL FAULTS
        # REASONABLE - catch major issues but allow minor imperfections
        critical_faults = []
        
        # Depth Critical - REASONABLE
        if self.min_knee_angle > 155: critical_faults.append("Shallow")  # Very shallow
        
        # Moderate Critical Thresholds - catch persistent issues
        if check_fault(self.knee_valgus_flags, 0.45): critical_faults.append("Valgus")
        if check_fault(self.knee_over_toes_flags, 0.40): critical_faults.append("Knee Over Toes")
        if check_fault(self.torso_lean_flags, 0.40): critical_faults.append("Back Lean")

        # Result - REASONABLE score threshold
        if not critical_faults and score >= 60:  # Reasonable passing score
            self.correct_reps += 1
            self.rep_count += 1
            self.advice = "Great form! Keep it up."
            self.feedback = f"Rep {self.rep_count}: Correct!"
        else:
            self.incorrect_reps += 1
            self.rep_count += 1 
            self.advice = "Watch your form. " + ", ".join(deductions) if deductions else "Form needs improvement."
            self.feedback = f"Rep {self.rep_count}: Incorrect - {', '.join(critical_faults) if critical_faults else 'Low score'}"

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
