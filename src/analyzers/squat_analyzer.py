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
        self.current_rep_quality = {}
        self.feedback = ""
        self.feedback = ""
        self.advice = ""
        self.prev_state = "STANDING"
        self.state_counter = 0 # Debounce counter
        self.state_transition_threshold = 8 # Increased from 4 to prevent overcounting
        
        # Thresholds (Balanced for stability)
        self.stand_threshold = 165 # Increased from 160
        self.descend_threshold = 145 # Increased from 140
        self.deep_threshold = 125     # Relaxed from 98 to count standard reps
        
        # Quality Metrics Data for current rep
        self.min_knee_angle = 180
        self.rep_start_time = 0
        self.descent_duration = 0
        self.ascent_duration = 0
        self.bottom_start_time = 0
        self.bottom_duration = 0
        self.knee_valgus_flags = 0
        self.back_angle_flags = 0
        self.heel_lift_flags = 0
        self.frame_count = 0
        
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
        self.knee_valgus_flags = 0
        self.back_angle_flags = 0
        self.knee_over_toes_flags = 0
        self.frame_count = 0
        self.knee_over_toes_flags = 0
        self.heel_lift_flags = 0
        self.advice = ""
        self.current_rep_quality = {}

    def analyze(self, landmarks, frame_width, frame_height):
        """
        Main analysis loop called every frame.
        """
        if not landmarks:
            return {
                "state": self.state,
                "rep_count": self.rep_count,
                "l_knee_angle": 0,
                "r_knee_angle": 0,
                "torso_angle": 0,
                "feedback": "No person detected",
                "last_rep_score": 0,
                "last_rep_quality": {},
                "valgus_detected": False,
                "view": "UNKNOWN"
            }

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
             return {
                "state": self.state,
                "rep_count": self.rep_count,
                "l_knee_angle": 0,
                "r_knee_angle": 0,
                "torso_angle": 0,
                "feedback": "Show Full Body!",
                "last_rep_score": self.current_rep_quality.get("score", 0),
                "last_rep_quality": self.current_rep_quality,
                "valgus_detected": False,
                "view": "UNKNOWN"
            }

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
        # Heuristic: Compare shoulder width to torso height
        shoulder_width = abs(l_shoulder_px[0] - r_shoulder_px[0])
        torso_height = abs((l_shoulder_px[1] + r_shoulder_px[1])/2 - (l_hip_px[1] + r_hip_px[1])/2)
        
        view = "FRONT"
        active_side = "BOTH"
        
        # If width is small relative to height, assume Side View
        # Threshold 0.25 is heuristic
        if torso_height > 0 and (shoulder_width / torso_height) < 0.4:
            view = "SIDE"
            # Determine Active Side (closest to camera). 
            # Use Z-coordinate (negative is closer in MediaPipe).
            l_z = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
            r_z = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
            
            if l_z < r_z: 
                active_side = "LEFT"
            else:
                active_side = "RIGHT"
                
            # Fallback to visibility if Z is unreliable or close
            if not is_right_visible and is_left_visible:
                active_side = "LEFT"
            elif not is_left_visible and is_right_visible:
                active_side = "RIGHT"
        
        # 2. Calculate Angles & Metrics based on View
        l_knee_angle = calculate_angle(l_hip_px, l_knee_px, l_ankle_px)
        r_knee_angle = calculate_angle(r_hip_px, r_knee_px, r_ankle_px)
        
        # Torso Angle (Vertical reference)
        l_torso_angle = calculate_angle((l_hip_px[0], l_hip_px[1]-100), l_hip_px, l_shoulder_px)
        r_torso_angle = calculate_angle((r_hip_px[0], r_hip_px[1]-100), r_hip_px, r_shoulder_px)
        
        current_knee_angle = 0
        current_torso_angle = 0
        symmetry_diff = 0
        knee_over_toes = False
        valgus_detected = False

        if view == "FRONT":
            # Require reasonably balanced visibility
            if not (is_left_visible and is_right_visible):
                 # If we thought it was FRONT but one side is missing, maybe force SIDE or warn?
                 # We'll allow it but rely on avg.
                 pass
                 
            current_knee_angle = (l_knee_angle + r_knee_angle) / 2.0
            current_torso_angle = (l_torso_angle + r_torso_angle) / 2.0
            symmetry_diff = abs(l_knee_angle - r_knee_angle)
            
            # Check Valgus (Front logic) - Robust Width Ratio Method
            knee_width = abs(l_knee_px[0] - r_knee_px[0])
            ankle_width = abs(l_ankle_px[0] - r_ankle_px[0])
            
            # If knees are significantly narrower than ankles (Valgus)
            # If knees are significantly narrower than ankles (Valgus)
            # Threshold: Knees < 63% of ankle width
            if knee_width < ankle_width * 0.63:
                valgus_detected = True
                
        else: # SIDE
            if active_side == "LEFT":
                current_knee_angle = l_knee_angle
                current_torso_angle = l_torso_angle
                # Knee Over Toe Check (Left Side)
                # Facing Left: Toe X < Ankle X. Knee should not be < Toe X.
                # Facing Right: Toe X > Ankle X. Knee should not be > Toe X.
                
                # Check Face Direction using Toes relative to Ankle
                toe_x = l_toe_px[0]
                ankle_x = l_ankle_px[0]
                knee_x = l_knee_px[0]
                
                # Tolerance for "Slightly Forward" (Rule 3)
                # Tolerance for "Slightly Forward" (Rule 3)
                # Tolerance for "Slightly Forward" (Rule 3)
                tolerance = 85 # Very lenient tolerance (85px)
                
                if toe_x < ankle_x: # Facing Left
                    if knee_x < toe_x - tolerance: # Crossed by more than tolerance
                        knee_over_toes = True
                else: # Facing Right
                    if knee_x > toe_x + tolerance:
                        knee_over_toes = True

            else: # RIGHT
                current_knee_angle = r_knee_angle
                current_torso_angle = r_torso_angle
                
                toe_x = r_toe_px[0]
                ankle_x = r_ankle_px[0]
                knee_x = r_knee_px[0]
                
                # Tolerance for "Slightly Forward" (Rule 3)
                # Tolerance for "Slightly Forward" (Rule 3)
                # Tolerance for "Slightly Forward" (Rule 3)
                tolerance = 85 # Very lenient tolerance (85px)
                
                if toe_x < ankle_x: # Facing Left
                    if knee_x < toe_x - tolerance: # Crossed by more than tolerance
                        knee_over_toes = True
                else: # Facing Right
                    if knee_x > toe_x + tolerance:
                        knee_over_toes = True

        # 3. Update State Metrics
        self.min_knee_angle = min(self.min_knee_angle, current_knee_angle)
        
        if valgus_detected:
            self.knee_valgus_flags += 1
        
        # Relaxed Back Angle Check (38 degrees)
        if current_torso_angle < 38:
            self.back_angle_flags += 1
            
        if knee_over_toes:
            self.knee_over_toes_flags += 1

        # Check Heel Lift (Heel Y should not be significantly higher than Toe Y)
        # Note: Y increases downwards. Higher (up) means lower Y value.
        # If Heel Y < Toe Y - threshold, it's lifted.
        heel_lift_detected = False
        # Normalize threshold? 0.02 is rough guess for normalized, but we have pixels here.
        # Let's use relative to ankle-toe distance or just fixed pixel threshold scaled by height?
        # Fixed pixel: 
        heel_lift_threshold = 75 # Very lenient threshold
        
        l_toe_py = l_toe_px[1]
        l_heel_py = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value], frame_width, frame_height)[1]
        
        r_toe_py = r_toe_px[1]
        r_heel_py = get_landmark_pixel(landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value], frame_width, frame_height)[1]
        
        if view == "SIDE":
            if active_side == "LEFT":
                 if l_heel_py < l_toe_py - heel_lift_threshold:
                     heel_lift_detected = True
            else:
                 if r_heel_py < r_toe_py - heel_lift_threshold:
                     heel_lift_detected = True
        else: # FRONT
            # Check both? usually harder to see heel lift in front view but we can try.
            # If either is lifted:
            if (l_heel_py < l_toe_py - heel_lift_threshold) or (r_heel_py < r_toe_py - heel_lift_threshold):
                heel_lift_detected = True
        
        if heel_lift_detected:
            self.heel_lift_flags += 1
            
        # 4. State Machine (Shared Logic)
        current_time = time.time()
        
        # Override feedback for Critical Errors immediately
        feed_override = None
        if knee_over_toes:
            feed_override = "Knees over toes!"
        elif heel_lift_detected:
            feed_override = "Keep Heels Down!"
        
        if self.state == "STANDING":
            # Squat must be bilateral: both knees should bend somewhat together
            # If one knee is bending much more than the other, it's likely a lunge
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
                    self.descent_duration = current_time - self.rep_start_time
                    self.feedback = "Hold bottom..."
                    self.state_counter = 0
            elif current_knee_angle > self.descend_threshold:
                # Abort immediately if back to standing? or debounce? 
                # Abort usually doesn't need strict debounce as it's a fail-safe
                self.state = "STANDING"
                self.feedback = "Aborted: Go Deeper!"
                self.state_counter = 0
            else:
                self.state_counter = 0
                
        elif self.state == "BOTTOM":
            if current_knee_angle > self.deep_threshold + 10:
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold:
                    self.state = "ASCENDING"
                    self.bottom_duration = current_time - self.bottom_start_time
                    self.feedback = "Push up!"
                    self.state_counter = 0
            else:
                self.state_counter = 0
                
        elif self.state == "ASCENDING":
            if current_knee_angle > self.stand_threshold:
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold:
                    self.state = "STANDING"
                    self.ascent_duration = current_time - (self.bottom_start_time + self.bottom_duration)
                    
                    # Minimum Duration Check (Prevent Fake Reps)
                    rep_total_time = self.descent_duration + self.bottom_duration + self.ascent_duration
                    if rep_total_time > 1.0: # Valid rep must take at least 1s
                        self.rep_count += 1
                        self._score_rep(symmetry_diff)
                    else:
                        self.feedback = "Rep too fast (ignored)"
                        
                    self.state_counter = 0
            else:
                self.state_counter = 0
                
        # Use override if immediate warning is needed, but prioritize state feedback if changing state
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
            "view": view,
            "knee_over_toes": knee_over_toes,
            "knee_over_toes": knee_over_toes,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "advice": self.advice,
            "target_muscles": "Quadriceps, Glutes, Hamstrings"
        }

    def _score_rep(self, symmetry_diff):
        """
        Calculates a score (0-100) for the completed rep.
        """
        score = 100
        deductions = []
        
        # 1. Depth - More Inclusive
        if self.min_knee_angle > 140:
            score -= 30 
            deductions.append("Too shallow")
        elif self.min_knee_angle > 125:
            score -= 10 
            deductions.append("Depth could be better")
            
        # 2. Tempo
        if self.descent_duration < 0.6: # More lenient
            score -= 5
            deductions.append("Too fast")
        
        # 3. Symmetry
        if symmetry_diff > 0.15: 
            score -= 5
            deductions.append("Asymmetrical")
            
        # 4. Back Angle
        if self.back_angle_flags > 15: 
            score -= 5
            deductions.append("Leaning forward")
            
        # 5. Knee Valgus
        if self.knee_valgus_flags > 15: 
            score -= 10
            deductions.append("Knee caving")
            
        # 6. Knee Over Toes
        if self.knee_over_toes_flags > 20: 
            score -= 5
            deductions.append("Knees forward")
            
        # 7. Heel Lift
        if self.heel_lift_flags > 15: 
            score -= 10
            deductions.append("Heels lifted")
            
        self.current_rep_quality = {
            "score": max(0, score),
            "depth": self.min_knee_angle,
            "descent_time": self.descent_duration,
            "comments": ", ".join(deductions) if deductions else "Good Rep!"
        }
        
        # Determine Correct/Incorrect based on CRITICAL FAULTS
        critical_faults = []
        
        # 1. Depth (Rule 5)
        if self.min_knee_angle > 140: # More inclusive
            critical_faults.append("Shallow")
            
        # 2. Valgus (Rule 3 - Knee collapse)
        if self.knee_valgus_flags > 35: 
            critical_faults.append("Valgus")
            
        # 3. Knee Over Toes (Rule 3 - Excessive)
        if self.knee_over_toes_flags > 50: 
            critical_faults.append("Knee Over Toes")
            
        # 4. Heel Lift (Rule 6)
        if self.heel_lift_flags > 35: 
            critical_faults.append("Heel Lift")
            
        # 5. Back Angle (Rule 2)
        if self.back_angle_flags > 50: 
            critical_faults.append("Back Lean")

        # Result
        if not critical_faults and score >= 70: # Lowered passing score to 70
             self.correct_reps += 1
        else:
             self.incorrect_reps += 1
             
        # Generate Advice
        self.advice = self._get_feedback_advice(critical_faults + deductions)
        self.feedback = f"Rep {self.rep_count}: {self.current_rep_quality['comments']}"
        
    def _get_feedback_advice(self, faults):
        """
        Returns actionable advice based on detected faults.
        Provides detailed, specific advice for each form issue.
        """
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
            
        # Detailed advice map with comprehensive guidance for each fault
        advice_map = {
            "Shallow": "🔴 DEPTH ISSUE: Go deeper! Your knees should bend to at least 90° (parallel). Focus on sitting back and down, not just bending knees. Think 'hips back, chest up'.",
            "Too shallow": "🔴 DEPTH ISSUE: Go deeper! Your knees should bend to at least 90° (parallel). Focus on sitting back and down, not just bending knees. Think 'hips back, chest up'.",
            "Depth could be better": "🟡 DEPTH: Get lower. Aim for parallel (90°) or below. Engage glutes and hamstrings by going deeper.",
            "Valgus": "🔴 KNEE VALGUS: Knees caving inward! Push knees OUT over toes. Strengthen glutes and focus on external rotation. This prevents knee injury.",
            "Knee valgus": "🔴 KNEE VALGUS: Knees caving inward! Push knees OUT over toes. Strengthen glutes and focus on external rotation. This prevents knee injury.",
            "Knee Valgus (caving in)": "🔴 KNEE VALGUS: Knees caving inward! Push knees OUT over toes. Strengthen glutes and focus on external rotation. This prevents knee injury.",
            "Knee Over Toes": "🔴 KNEE POSITION: Knees too far forward! Sit BACK more, push hips back first. Keep shins more vertical. This protects your knees.",
            "Knees crossed toes": "🔴 KNEE POSITION: Knees too far forward! Sit BACK more, push hips back first. Keep shins more vertical. This protects your knees.",
            "Heel Lift": "🔴 FOOT POSITION: Heels lifting off ground! Keep entire foot planted. Try wider stance or elevate heels slightly if ankle mobility is limited.",
            "Heels lifted": "🔴 FOOT POSITION: Heels lifting off ground! Keep entire foot planted. Try wider stance or elevate heels slightly if ankle mobility is limited.",
            "Back Lean": "🔴 TORSO ANGLE: Leaning too far forward! Keep chest UP and core tight. Imagine showing your chest to the wall in front. This isn't a good morning exercise.",
            "Leaning forward": "🔴 TORSO ANGLE: Leaning too far forward! Keep chest UP and core tight. Imagine showing your chest to the wall in front. This isn't a good morning exercise.",
            "Dive bomb (too fast)": "🟡 TEMPO: Control the descent! Lower yourself slowly (2-3 seconds). This builds strength and prevents injury. Don't drop down.",
            "Asymmetrical": "🟡 SYMMETRY: One side is doing more work! Focus on equal weight distribution. Push evenly through both feet. Check for muscle imbalances."
        }
        
        # Collect all relevant advice messages
        advice_list = []
        for fault in faults:
            for key in advice_map:
                if key in fault:
                    advice_list.append(advice_map[key])
                    break
        
        # Return prioritized advice (most critical first) or combine if multiple issues
        if advice_list:
            # Return the first (most critical) advice, or combine if multiple critical issues
            if len(advice_list) > 1:
                return f"{advice_list[0]} | Also: {', '.join([a.split(':')[0] for a in advice_list[1:]])}"
            return advice_list[0]
                    
        return "🟡 Focus on maintaining proper form throughout the movement."
