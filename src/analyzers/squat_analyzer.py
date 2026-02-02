import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class SquatAnalyzer:
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
        self.state_counter = 0 # Debounce counter
        self.state_transition_threshold = 1 # Instant response
        
        # Thresholds (Balanced for stability - Relaxed Mode)
        self.stand_threshold = 160 # Easier lockout
        self.descend_threshold = 168 # Start descent very early
        self.deep_threshold = 158     # Shallow depth counts (approx 1/4 squat)
        
        # Quality Metrics Data for current rep
        self.min_knee_angle = 180
        self.rep_start_time = 0
        self.descent_duration = 0
        self.ascent_duration = 0
        self.bottom_start_time = 0
        self.bottom_duration = 0
        self.last_state_change_time = 0 # Track time for timeout logic
        
        # Landmarks indices (MediaPipe Pose)
        self.mp_pose = mp.solutions.pose
        self._reset_rep_stats()
        
    def _reset_rep_stats(self):
        self.min_knee_angle = 180
        self.rep_start_time = time.time()
        self.descent_duration = 0
        self.ascent_duration = 0
        self.bottom_start_time = 0
        self.bottom_duration = 0
        self.last_state_change_time = time.time()
        self.knee_valgus_flags = 0
        self.back_angle_flags = 0
        self.knee_over_toes_flags = 0
        self.heel_lift_flags = 0
        self.frame_count = 0
        self.advice = ""
        self.current_rep_quality = {}
        
    def _reset_rep_stats(self):
        self.min_knee_angle = 180
        self.rep_start_time = time.time()
        self.descent_duration = 0
        self.ascent_duration = 0
        self.bottom_start_time = 0
        self.bottom_duration = 0
        self.knee_valgus_flags = 0
        self.back_angle_flags = 0
        self.knee_over_toes_flags = 0
        self.heel_lift_flags = 0
        self.frame_count = 0
        self.advice = ""
        self.current_rep_quality = {}

    def analyze(self, landmarks, frame_width, frame_height):
        """
        Main analysis loop called every frame.
        """
        if not landmarks:
            return self._empty_result("No person detected")

        # 0. Check Visibility & Detect View
        # Define side-specific key landmarks
        left_side_lm = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_HEEL, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        ]
        right_side_lm = [
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_HEEL, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        ]

        # Check visibility for both sides
        is_left_visible = all(landmarks[lm.value].visibility > 0.5 for lm in left_side_lm)
        is_right_visible = all(landmarks[lm.value].visibility > 0.5 for lm in right_side_lm)

        if not (is_left_visible or is_right_visible):
             return self._empty_result("Show Full Body!")

        # Extract Key Landmarks (Pixels)
        l_hip_px = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value], frame_width, frame_height)
        r_hip_px = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value], frame_width, frame_height)
        l_shoulder_px = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value], frame_width, frame_height)
        r_shoulder_px = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value], frame_width, frame_height)
        
        l_knee_px = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value], frame_width, frame_height)
        r_knee_px = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value], frame_width, frame_height)
        l_ankle_px = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value], frame_width, frame_height)
        r_ankle_px = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value], frame_width, frame_height)
        
        # New Landmarks for Side Check
        l_toe_px = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value], frame_width, frame_height)
        r_toe_px = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value], frame_width, frame_height)

        # 1. Determine View (Front vs Side)
        shoulder_width = abs(l_shoulder_px[0] - r_shoulder_px[0])
        torso_height = abs((l_shoulder_px[1] + r_shoulder_px[1])/2 - (l_hip_px[1] + r_hip_px[1])/2)
        
        view = "FRONT"
        active_side = "BOTH"
        
        if torso_height > 0 and (shoulder_width / torso_height) < 0.4:
            view = "SIDE"
            # Determine Active Side
            l_z = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
            r_z = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
            
            if l_z < r_z: 
                active_side = "LEFT"
            else:
                active_side = "RIGHT"
                
            if not is_right_visible and is_left_visible: active_side = "LEFT"
            elif not is_left_visible and is_right_visible: active_side = "RIGHT"
        
        # 2. Calculate Angles & Metrics based on View
        l_knee_angle = calculate_angle(l_hip_px, l_knee_px, l_ankle_px)
        r_knee_angle = calculate_angle(r_hip_px, r_knee_px, r_ankle_px)
        
        l_torso_angle = calculate_angle((l_hip_px[0], l_hip_px[1]-100), l_hip_px, l_shoulder_px)
        r_torso_angle = calculate_angle((r_hip_px[0], r_hip_px[1]-100), r_hip_px, r_shoulder_px)
        
        current_knee_angle = 0
        current_torso_angle = 0
        symmetry_diff = 0
        knee_over_toes = False
        valgus_detected = False

        if view == "FRONT":
            current_knee_angle = (l_knee_angle + r_knee_angle) / 2.0
            current_torso_angle = (l_torso_angle + r_torso_angle) / 2.0
            symmetry_diff = abs(l_knee_angle - r_knee_angle)
            
            # Check Valgus (Front logic)
            knee_width = abs(l_knee_px[0] - r_knee_px[0])
            ankle_width = abs(l_ankle_px[0] - r_ankle_px[0])
            
            # Relaxed Threshold: Knees < 58% of ankle width (was 63%)
            if knee_width < ankle_width * 0.58:
                valgus_detected = True
                
        else: # SIDE
            if active_side == "LEFT":
                current_knee_angle = l_knee_angle
                current_torso_angle = l_torso_angle
                toe_x = l_toe_px[0]
                ankle_x = l_ankle_px[0]
                knee_x = l_knee_px[0]
            else: # RIGHT
                current_knee_angle = r_knee_angle
                current_torso_angle = r_torso_angle
                toe_x = r_toe_px[0]
                ankle_x = r_ankle_px[0]
                knee_x = r_knee_px[0]
            
            # Relaxed Tolerance: 95px (was 85px)
            tolerance = 95
            
            if toe_x < ankle_x: # Facing Left
                if knee_x < toe_x - tolerance: knee_over_toes = True
            else: # Facing Right
                if knee_x > toe_x + tolerance: knee_over_toes = True

        # 3. Update State Metrics
        self.min_knee_angle = min(self.min_knee_angle, current_knee_angle)
        
        if valgus_detected: self.knee_valgus_flags += 1
        
        # Relaxed Back Angle Check (35 degrees, was 38)
        if current_torso_angle < 35: self.back_angle_flags += 1
            
        if knee_over_toes: self.knee_over_toes_flags += 1

        # Check Heel Lift
        heel_lift_detected = False
        # Relaxed Threshold: 85px (was 75px)
        heel_lift_threshold = 85
        
        l_toe_py = l_toe_px[1]
        l_heel_py = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value], frame_width, frame_height)[1]
        r_toe_py = r_toe_px[1]
        r_heel_py = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value], frame_width, frame_height)[1]
        
        if view == "SIDE":
            if active_side == "LEFT":
                 if l_heel_py < l_toe_py - heel_lift_threshold: heel_lift_detected = True
            else:
                 if r_heel_py < r_toe_py - heel_lift_threshold: heel_lift_detected = True
        else: # FRONT
            if (l_heel_py < l_toe_py - heel_lift_threshold) or (r_heel_py < r_toe_py - heel_lift_threshold):
                heel_lift_detected = True
        
        if heel_lift_detected: self.heel_lift_flags += 1
            
        # 4. State Machine (Shared Logic)
        current_time = time.time()
        
        # Override feedback for Critical Errors immediately
        feed_override = None
        if knee_over_toes: feed_override = "Knees over toes!"
        elif heel_lift_detected: feed_override = "Keep Heels Down!"
        
        if self.state == "STANDING":
            knee_diff = abs(l_knee_angle - r_knee_angle)
            if current_knee_angle < self.descend_threshold:
                if knee_diff > 40: # Significant asymmetry -> likely a lunge
                    self.feedback = "Lunge detected? Switch to Lunge mode."
                    self.state_counter = 0
                else:
                    self.state_counter += 1
                    if self.state_counter > self.state_transition_threshold:
                        self.state = "DESCENDING"
                        self._reset_rep_stats()
                        self.rep_start_time = current_time
                        self.last_state_change_time = current_time
                        self.feedback = "Descending..."
                        self.state_counter = 0
            else:
                 self.state_counter = 0
                
        elif self.state == "DESCENDING":
            if current_knee_angle < self.deep_threshold:
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold:
                    self.state = "BOTTOM"
                    self.bottom_start_time = current_time
                    self.last_state_change_time = current_time
                    self.descent_duration = current_time - self.rep_start_time
                    self.feedback = "Hold bottom..."
                    self.state_counter = 0
            elif current_knee_angle > self.descend_threshold:
                # Add debounce to abort logic too (prevents flickering)
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold:
                    self.state = "STANDING"
                    self.feedback = "Aborted: Go Deeper!"
                    self.last_state_change_time = current_time
                    self.state_counter = 0
            else:
                self.state_counter = 0
                
        elif self.state == "BOTTOM":
            if current_knee_angle > self.deep_threshold + 5: # Reduced hysteresis from 10 to 5 for faster exit
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold:
                    self.state = "ASCENDING"
                    self.bottom_duration = current_time - self.bottom_start_time
                    self.last_state_change_time = current_time
                    self.feedback = "Push up!"
                    self.state_counter = 0
            else:
                self.state_counter = 0
                
        elif self.state == "ASCENDING":
            # Timeout / Stuck Prevention: If stuck in ASCENDING for > 4s and somewhat upright, finish rep
            time_in_state = current_time - self.last_state_change_time
            is_stuck_upright = (time_in_state > 4.0 and current_knee_angle > 145)
            
            if current_knee_angle > self.stand_threshold or is_stuck_upright:
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold or is_stuck_upright:
                    self.state = "STANDING"
                    self.ascent_duration = current_time - (self.bottom_start_time + self.bottom_duration)
                    
                    # Minimum Duration Check (Prevent Fake Reps) - Relaxed
                    rep_total_time = self.descent_duration + self.bottom_duration + self.ascent_duration
                    if rep_total_time > 0.5: # Valid rep must take at least 0.5s (was 1.0)
                        self.rep_count += 1
                        self._score_rep(symmetry_diff)
                        self.feedback = "Good Rep!" if not is_stuck_upright else "Rep Completed (Timeout)"
                    else:
                        self.feedback = f"Rep too fast ({rep_total_time:.1f}s)"
                        
                    self.state_counter = 0
                    self.last_state_change_time = current_time
                self.state_counter = 0
            
            # Feature: Allow chaining reps even if lockout wasn't perfect (prevent getting stuck)
            elif current_knee_angle < self.descend_threshold:
                 self.state = "DESCENDING"
                 self.feedback = "Incomplete extension - Go again!"
                 self.rep_start_time = current_time # Reset timer for new rep
                 self._reset_rep_stats()
                 self.state_counter = 0
                
        if feed_override and self.state in ["DESCENDING", "BOTTOM"]:
             self.feedback = feed_override

        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "l_knee_angle": l_knee_angle,
            "r_knee_angle": r_knee_angle,
            "torso_angle": current_torso_angle,
            "feedback": self.feedback,
            "last_rep_score": self.current_rep_quality.get("score", 0),
            "last_rep_quality": self.current_rep_quality,
            "valgus_detected": valgus_detected,
            "view": view,
            "knee_over_toes": knee_over_toes,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "advice": self.advice,
            "target_muscles": "Quadriceps, Glutes, Hamstrings"
        }

    def _score_rep(self, symmetry_diff):
        score = 100
        deductions = []
        
        # 1. Depth - Relaxed
        if self.min_knee_angle > 150: # was 140
            score -= 30 
            deductions.append("Too shallow")
        elif self.min_knee_angle > 135: # was 125
            score -= 10 
            deductions.append("Depth could be better")
            
        # 2. Tempo
        if self.descent_duration < 0.6: 
            score -= 5
            deductions.append("Too fast")
        
        # 3. Symmetry
        if symmetry_diff > 0.15: 
            score -= 5
            deductions.append("Asymmetrical")
            
        # 4. Back Angle
        if self.back_angle_flags > 20: 
            score -= 5
            deductions.append("Leaning forward")
            
        # 5. Knee Valgus
        if self.knee_valgus_flags > 20: 
            score -= 10
            deductions.append("Knee caving")
            
        # 6. Knee Over Toes
        if self.knee_over_toes_flags > 25: 
            score -= 5
            deductions.append("Knees forward")
            
        # 7. Heel Lift
        if self.heel_lift_flags > 20: 
            score -= 10
            deductions.append("Heels lifted")
            
        self.current_rep_quality = {
            "score": max(0, score),
            "depth": self.min_knee_angle,
            "descent_time": self.descent_duration,
            "comments": ", ".join(deductions) if deductions else "Good Rep!"
        }
        
        # Determine Correct/Incorrect based on CRITICAL FAULTS (RELAXED)
        critical_faults = []
        
        # 1. Depth
        if self.min_knee_angle > 150: # Relaxed from 140
            critical_faults.append("Shallow")
            
        # 2. Valgus
        if self.knee_valgus_flags > 45: # Relaxed from 35
            critical_faults.append("Valgus")
            
        # 3. Knee Over Toes
        if self.knee_over_toes_flags > 60: # Relaxed from 50
            critical_faults.append("Knee Over Toes")
            
        # 4. Heel Lift
        if self.heel_lift_flags > 45: # Relaxed from 35
            critical_faults.append("Heel Lift")
            
        # 5. Back Angle
        if self.back_angle_flags > 60: # Relaxed from 50
            critical_faults.append("Back Lean")

        # Result - Relaxed Passing Score
        if not critical_faults and score >= 60: # Lowered from 70
             self.correct_reps += 1
        else:
             self.incorrect_reps += 1
             
        self.advice = self._get_feedback_advice(critical_faults + deductions)
        self.feedback = f"Rep {self.rep_count}: {self.current_rep_quality['comments']}"
        
    def _get_feedback_advice(self, faults):
        if not faults:
            import random
            pro_tips = [
                "Light weight baby! Brace that core.",
                "Ass to grass! Drive with your hips.",
                "Clean rep. Keep squeezing the glutes.",
                "Textbook. Don't let the tension drop.",
                "Easy money! Focus on bar speed.",
                "Butter! Stay tight at the bottom."
            ]
            return random.choice(pro_tips)
            
        advice_map = {
            "Shallow": "🔴 Go deeper! Aim for parallel.",
            "Too shallow": "🔴 Go deeper! Aim for parallel.",
            "Depth could be better": "🟡 Get lower. Aim for parallel.",
            "Valgus": "🔴 Push knees OUT!",
            "Knee valgus": "🔴 PUSH KNEES OUT!",
            "Knee caving": "🔴 PUSH KNEES OUT!",
            "Knee Over Toes": "🔴 Sit BACK more, weight on heels.",
            "Knees forward": "🔴 Sit BACK more.",
            "Heel Lift": "🔴 Keep heels DOWN.",
            "Heels lifted": "🔴 Keep heels DOWN.",
            "Back Lean": "🔴 Chest UP!",
            "Leaning forward": "🔴 Chest UP!",
            "Too fast": "🟡 Slow down the descent.",
            "Asymmetrical": "🟡 Push evenly with both legs."
        }
        
        advice_list = []
        for fault in faults:
            for key in advice_map:
                if key in fault:
                    advice_list.append(advice_map[key])
                    break
        
        if advice_list:
            if len(advice_list) > 1:
                return f"{advice_list[0]} | Also: {', '.join([a.split(':')[0] for a in advice_list[1:]])}"
            return advice_list[0]
                    
        return "🟡 Focus on good form."

    def _empty_result(self, msg):
        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "l_knee_angle": 0, "r_knee_angle": 0, "torso_angle": 0,
            "feedback": msg, "last_rep_score": 0,
            "last_rep_quality": {}, "valgus_detected": False, "view": "UNKNOWN",
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps
        }
