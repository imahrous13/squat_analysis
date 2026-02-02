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
        
        # Locking for stability
        self.locked_view = "FRONT"
        self.locked_facing = "RIGHT"
        self.locked_lead_leg = None
        
        # Thresholds (Optimized for accuracy)
        self.stand_threshold = 150  # Balanced - not too strict, not too lenient
        self.descend_threshold = 145  # Detect descent slightly earlier (was 140)
        self.bottom_threshold = 120  # Tuned to 120 (Reduces false positive reps compared to 125) 
        
        # Rep Stats
        self.min_knee_angle = 180
        self.lead_leg = None 
        self.rep_start_time = 0
        self.last_state_change_time = 0
        self.knee_valgus_flags = 0
        self.hip_tilt_flags = 0
        self.knee_over_toes_flags = 0
        self.head_over_knee_flags = 0  # New strict check
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
        self.head_over_knee_flags = 0
        self.torso_lean_flags = 0
        self.lead_leg = None
        self.rep_frame_count = 0 # Track total active frames for ratio-based scoring
        self.current_rep_quality = {}
        # Do not reset locked vars here, they reset on state transition to STANDING

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

        # 1. Detect View & Active Side (Dynamic in STANDING, Locked otherwise)
        shoulder_width = abs(l_sh[0] - r_sh[0])
        torso_height = abs((l_sh[1] + r_sh[1])/2 - (l_hip[1] + r_hip[1])/2)

        if self.state == "STANDING":
            self.locked_view = "FRONT"
            if torso_height > 0:
                self.locked_view = "FRONT" if (shoulder_width / torso_height) > 0.5 else "SIDE"
                
            # Determine facing
            self.locked_facing = "RIGHT"
            if self.locked_view == "SIDE":
                if landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility > landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility:
                    self.locked_facing = "RIGHT" if l_toe[0] > l_ank[0] else "LEFT"
                else:
                    self.locked_facing = "RIGHT" if r_toe[0] > r_ank[0] else "LEFT"
                    
            # Detect Lead Leg
            # In STANDING, lead leg implies which one is slightly forward or will be
            # We aggressively update this until descent starts
            if self.locked_view == "SIDE":
                 if self.locked_facing == "RIGHT":
                     if l_toe[0] > r_toe[0] + 10: self.locked_lead_leg = "LEFT" # Lower threshold in standing
                     elif r_toe[0] > l_toe[0] + 10: self.locked_lead_leg = "RIGHT"
                 else:
                     if l_toe[0] < r_toe[0] - 10: self.locked_lead_leg = "LEFT"
                     elif r_toe[0] < l_toe[0] - 10: self.locked_lead_leg = "RIGHT"
            else:
                 # Front view: usually the one with the larger visible shin angle or simply deeper bend
                 # In standing, assume none or keep previous
                 pass
                 
        # Use Locked Values for Analysis
        view = self.locked_view
        facing = self.locked_facing
        self.lead_leg = self.locked_lead_leg # Map to instance var for compatibility
        
        # 2. Calculate Angles
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ank)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ank)
        
        # KEY FIX: Always track the WORKING knee (the one bending)
        # This prevents measuring the straight back leg and failing the rep
        active_knee_angle = min(l_knee_angle, r_knee_angle)
        
        # Determine lead leg by angle if locked is None (Fallback)
        if self.lead_leg is None and active_knee_angle < 160:
             if abs(l_knee_angle - r_knee_angle) > 15:
                 self.lead_leg = "LEFT" if l_knee_angle < r_knee_angle else "RIGHT"
                 self.locked_lead_leg = self.lead_leg

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
        hip_tilt_threshold = relative_scale * 0.20
        if abs(l_hip[1] - r_hip[1]) > hip_tilt_threshold: self.hip_tilt_flags += 1

        # C. Knee Over Toes (Strict but with margin)
        knee_over_toes = False
        kot_threshold = relative_scale * 0.12 # Relaxed to reduce false positives (was 0.08)
        
        target_knee = l_knee if self.lead_leg == "LEFT" else r_knee
        target_toe = l_toe if self.lead_leg == "LEFT" else r_toe
        target_nose = get_p(self.mp_pose.PoseLandmark.NOSE) # Head
        
        if view == "SIDE":
            if facing == "RIGHT":
                if target_knee[0] > target_toe[0] + kot_threshold: knee_over_toes = True
            else:
                if target_knee[0] < target_toe[0] - kot_threshold: knee_over_toes = True
        if knee_over_toes: self.knee_over_toes_flags += 1

        # D. Head Over Knee (Excessive Lean)
        head_over_knee = False
        hok_threshold = relative_scale * 0.10 # Relaxed to 0.10 (was 0.05)
        
        if view == "SIDE":
            if facing == "RIGHT":
                 # If Head X > Knee X (Leaning forward past knee)
                 if target_nose[0] > target_knee[0] + hok_threshold: head_over_knee = True
            else:
                 # If Head X < Knee X (Leaning forward past knee)
                 if target_nose[0] < target_knee[0] - hok_threshold: head_over_knee = True
        if head_over_knee: self.head_over_knee_flags += 1

        # E. Torso Lean
        torso_lean = False
        if view == "SIDE":
            # Point straight up from hip
            v_point = (l_hip[0] if self.lead_leg == "LEFT" else r_hip[0], 
                       (l_hip[1] if self.lead_leg == "LEFT" else r_hip[1]) - 100) # Negative Y is up
            
            target_sh = l_sh if self.lead_leg == "LEFT" else r_sh
            target_hip = l_hip if self.lead_leg == "LEFT" else r_hip
            
            # Angle against vertical
            torso_angle = calculate_angle(target_sh, target_hip, v_point)
            if torso_angle > 15: torso_lean = True # Stricter (was 20)
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
            elif active_knee_angle > self.descend_threshold + 5:
                # Abort/Reset if user stands back up without reaching bottom
                # Removed scoring here to prevent double-counting during wobbles
                self.state = "STANDING"
                self.last_state_change_time = time.time()
                self._reset_rep_stats()
            else:
                self.state_counter = 0

        elif self.state == "BOTTOM":
            self.min_knee_angle = min(self.min_knee_angle, active_knee_angle)
            if active_knee_angle > self.bottom_threshold + 10:
                self.state_counter += 1
                if self.state_counter >= self.state_transition_threshold:
                    self.state = "ASCENDING"
                    self.last_state_change_time = time.time()
                    self.state_counter = 0
                    self.feedback = "Push up!"
            else:
                self.state_counter = 0

        elif self.state == "ASCENDING":
            # Timeout / Stuck Prevention (More aggressive)
            time_in_state = time.time() - self.last_state_change_time
            is_stuck_upright = (time_in_state > 2.0 and active_knee_angle > 140)
            
            # Lower stand_threshold to 145 for more reliable completion
            if active_knee_angle > 145 or is_stuck_upright:
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
            
        def fmt_fault(msg, flags):
            return f"{msg} ({flags}/{total_frames})"
        
        # 1. Depth - REASONABLE
        if self.min_knee_angle > 145:  
            score -= 25  
            deductions.append(f"Too Shallow ({int(self.min_knee_angle)} deg)")
        elif self.min_knee_angle > 130:  
            score -= 8  
            deductions.append(f"Slightly Shallow ({int(self.min_knee_angle)} deg)")
            
        # 2. Knee Stability - MODERATE
        if check_fault(self.knee_valgus_flags, 0.25):  
            score -= 15 
            deductions.append(fmt_fault("Knee Caving", self.knee_valgus_flags))
            
        # 3. Hip Level - LENIENT (minor issue)
        if check_fault(self.hip_tilt_flags, 0.35):  
            score -= 5 
            deductions.append(fmt_fault("Hips Not Level", self.hip_tilt_flags))
            
        # 4. Knee Over Toes - CRITICAL (per user request)
        if check_fault(self.knee_over_toes_flags, 0.30):  
            score -= 20
            deductions.append(fmt_fault("Knee Over Toes", self.knee_over_toes_flags))
            
        # 5. Torso Lean - MODERATE
        if check_fault(self.torso_lean_flags, 0.30):  
            score -= 15
            deductions.append(fmt_fault("Leaning Forward", self.torso_lean_flags))

        # 6. Head Over Knee - CRITICAL (per user request)
        if check_fault(self.head_over_knee_flags, 0.30):  
            score -= 25
            deductions.append(fmt_fault("Head Over Knee", self.head_over_knee_flags))

        # 7. Speed Check - CRITICAL (per user request)
        is_too_fast = total_frames < 20 # Roughly < 0.7s
        if is_too_fast:
            score -= 30
            deductions.append(f"Too Fast ({total_frames} frames)")

        self.current_rep_quality = {
            "score": max(0, score),
            "faults": deductions
        }

        # Determine Correct/Incorrect based on CRITICAL FAULTS
        # REASONABLE - catch major issues but allow minor imperfections
        critical_faults = []
        
        # Depth Critical - REASONABLE
        if self.min_knee_angle > 160: critical_faults.append("Shallow")  # Very shallow (was 155)
        
        # Moderate Critical Thresholds - catch persistent issues
        if check_fault(self.knee_valgus_flags, 0.45): critical_faults.append("Valgus")
        if check_fault(self.knee_over_toes_flags, 0.30): critical_faults.append("Knee over toes") # Relaxed to 0.30
        if check_fault(self.head_over_knee_flags, 0.30): critical_faults.append("Head over knee") # Relaxed to 0.30
        if check_fault(self.torso_lean_flags, 0.45): critical_faults.append("Back Lean")
        if is_too_fast: critical_faults.append("Too fast") # CRITICAL

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
