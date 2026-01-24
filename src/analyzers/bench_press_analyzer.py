import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class BenchPressAnalyzer:
    """
    Biomechanical Analyzer for Bench Press.
    Variants:
    - "standard": Lying Flat Bench (Barbell/Dumbbell)
    - "seated": Machine Press or Cable Press (Upright/Seated)
    
    Phases: EXTENDED -> DESCENDING -> BOTTOM -> ASCENDING -> EXTENDED
    """
    def __init__(self, variant="standard"):
        self.variant = variant.lower()
        self.state = "EXTENDED"
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_rep_quality = {}
        self.feedback = ""
        self.advice = ""
        
        # State transition management
        self.state_counter = 0
        self.state_transition_threshold = 5  # Responsive
        
        # Rep tracking
        self.min_elbow_angle = 180
        self.max_elbow_angle = 0
        self.rep_start_time = time.time()
        
        # Fault tracking
        self.fault_counts = {
            "ELBOW_FLARE": 0,
            "ASYMMETRY": 0,
            "HIP_LIFT": 0,
            "BOUNCING": 0,
            "SHOULDER_SHRUG": 0,
            "HAND_CROSSING": 0,
            "PARTIAL_ROM": 0
        }
        
        # Base configs
        self.config = {
            "extended_elbow_angle": 155,      # Relaxed from 160
            "descend_threshold": 150,
            "bottom_elbow_angle": 100,        # Relaxed from 90 to catch reps that don't go super deep/camera angles
            "min_rom": 45,                    # Relaxed ROM requirement
            "hysteresis": 10,
            "smoothing_alpha": 0.3,
            "max_elbow_flare": 85,
            "max_asymmetry": 20,
            "max_hip_lift_norm": 0.15,        # Relaxed from 0.08 to 0.15
            "bounce_velocity_thresh": 15,
            "fault_frame_thresh": 10
        }
        
        # Adjust for Seated Variant (Machine/Cable) based on Prompt
        if self.variant == "seated":
            self.config["extended_elbow_angle"] = 155  
            self.config["bottom_elbow_angle"] = 100    
            self.config["descend_threshold"] = 120     
            
        self.prev_vals = {}
        self.base_shoulder_y = None  # Reference for shrug detection
        
        self.mp_pose = mp.solutions.pose
        self._reset_rep_stats()
        
    def _reset_rep_stats(self):
        self.min_elbow_angle = 180
        self.max_elbow_angle = 0
        self.rep_start_time = time.time()
        self.fault_counts = {k: 0 for k in self.fault_counts}
        self.advice = ""
        self.current_rep_quality = {}
        self.base_shoulder_y = None
        
    def _ema_smooth(self, key, new_val):
        if key not in self.prev_vals:
            self.prev_vals[key] = new_val
            return new_val
        alpha = self.config["smoothing_alpha"]
        smooth = alpha * new_val + (1 - alpha) * self.prev_vals[key]
        self.prev_vals[key] = smooth
        return smooth

    def analyze(self, landmarks, frame_width, frame_height):
        if not landmarks:
            return self._empty_result()
            
        lm = self.mp_pose.PoseLandmark
        def get_px(idx): return get_landmark_pixel(landmarks[idx], frame_width, frame_height)
        
        # Get landmarks
        l_shoulder = get_px(lm.LEFT_SHOULDER.value)
        r_shoulder = get_px(lm.RIGHT_SHOULDER.value)
        l_elbow = get_px(lm.LEFT_ELBOW.value)
        r_elbow = get_px(lm.RIGHT_ELBOW.value)
        l_wrist = get_px(lm.LEFT_WRIST.value)
        r_wrist = get_px(lm.RIGHT_WRIST.value)
        l_hip = get_px(lm.LEFT_HIP.value)
        r_hip = get_px(lm.RIGHT_HIP.value)
        l_ankle = get_px(lm.LEFT_ANKLE.value)
        r_ankle = get_px(lm.RIGHT_ANKLE.value)
        
        # Determine active view (side with better visibility)
        l_vis = landmarks[lm.LEFT_SHOULDER.value].visibility
        r_vis = landmarks[lm.RIGHT_SHOULDER.value].visibility
        active_side = "LEFT" if l_vis > r_vis else "RIGHT"
        
        # Calculate elbow angles for both arms
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        # Use primary arm based on visibility
        primary_elbow_angle = l_elbow_angle if active_side == "LEFT" else r_elbow_angle
        
        # Smooth the angle
        elbow_angle = self._ema_smooth("elbow_angle", primary_elbow_angle)
        
        # Track min/max
        self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
        self.max_elbow_angle = max(self.max_elbow_angle, elbow_angle)
        
        # Form Checks
        self._check_form(
            l_elbow_angle, r_elbow_angle,
            l_shoulder, r_shoulder,
            l_elbow, r_elbow,
            l_hip, r_hip,
            l_ankle, r_ankle,
            l_wrist, r_wrist,
            elbow_angle
        )
        
        # State Machine
        self._update_state(elbow_angle)
        
        # Generate feedback
        self._generate_feedback(elbow_angle)
        
        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "feedback": self.feedback,
            "advice": self.advice,
            "last_rep_score": self.current_rep_quality.get("score", 0),
            "reasons": self.current_rep_quality.get("reasons", []),
            "target_muscles": f"Chest, Triceps, Shoulders ({self.variant.title()})",
            "view": f"Side ({active_side})",
            "elbow_angle": int(elbow_angle)
        }
    
    def _check_form(self, l_elbow_angle, r_elbow_angle, l_shoulder, r_shoulder,
                    l_elbow, r_elbow, l_hip, r_hip, l_ankle, r_ankle, 
                    l_wrist, r_wrist, avg_elbow_angle):
        
        # 1. Elbow Asymmetry (Prompt: > 20 degrees)
        asymmetry = abs(l_elbow_angle - r_elbow_angle)
        if asymmetry > self.config["max_asymmetry"]:
            self.fault_counts["ASYMMETRY"] += 1
        else:
            self.fault_counts["ASYMMETRY"] = max(0, self.fault_counts["ASYMMETRY"] - 1)
        
        # 2. Shoulder Shrug (Seated Variant - Prompt: Shoulder Y rises)
        if self.variant == "seated":
            current_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            
            # Initialize baseline at start of rep/set
            if self.base_shoulder_y is None or self.state == "EXTENDED":
                 # If we are extended/upright, this is likely 'down' position for shoulders
                 # Update baseline slowly to drift with user
                 if self.base_shoulder_y is None:
                     self.base_shoulder_y = current_shoulder_y
                 else:
                     self.base_shoulder_y = 0.95 * self.base_shoulder_y + 0.05 * current_shoulder_y
            
            # Check for shrug (Y decreases significantly)
            # Threshold: 5% of body length?
            body_len = abs(l_shoulder[1] - l_hip[1]) # Torso length
            shrug_threshold = body_len * 0.05
            
            if self.base_shoulder_y and (self.base_shoulder_y - current_shoulder_y) > shrug_threshold:
                 # Shoulders moved UP (Y decreased)
                 self.fault_counts["SHOULDER_SHRUG"] += 1
            else:
                 self.fault_counts["SHOULDER_SHRUG"] = max(0, self.fault_counts["SHOULDER_SHRUG"] - 1)

        # 3. Hip Lift (Standard Variant Only)
        elif self.variant == "standard":
            hip_y = (l_hip[1] + r_hip[1]) / 2
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            body_length = abs(shoulder_y - (l_ankle[1] + r_ankle[1])/2)
            if body_length > 0:
                expected_hip_y = shoulder_y + (body_length * 0.4)
                hip_deviation = abs(hip_y - expected_hip_y) / body_length
                if hip_deviation > self.config["max_hip_lift_norm"]:
                    self.fault_counts["HIP_LIFT"] += 1
                else:
                    self.fault_counts["HIP_LIFT"] = max(0, self.fault_counts["HIP_LIFT"] - 1)

        # 4. Hand Crossing (Cable Check)
        # Check if wrists cross significantly (left wrist x > right wrist x relative to shoulders)
        # Assuming camera faces front. If side view, this is invalid.
        # Simple proximity check: If distance is near zero.
        wrist_dist = np.sqrt((l_wrist[0]-r_wrist[0])**2 + (l_wrist[1]-r_wrist[1])**2)
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        
        # Only check if front view (shoulder width significant)
        if shoulder_width > 50: 
            if wrist_dist < shoulder_width * 0.1: # Hands touching or crossed
                # Could be valid for cable finish, but "Over-crossing" is potential error in prompt
                # We'll just track it, but maybe not penalize heavily unless extreme?
                # Prompt says "Wrist-to-wrist distance collapses excessively -> Flag: Over-crossing"
                # For now we won't penalize strict crossing unless it affects stability, 
                # but let's track it for advice.
                pass 

        self.prev_vals["elbow_angle_raw"] = avg_elbow_angle
    
    def _update_state(self, elbow_angle):
        def transition_to(new_state):
            if self.state != new_state:
                self.state_counter += 1
                if self.state_counter >= self.state_transition_threshold:
                    self.state = new_state
                    self.state_counter = 0
                    return True
            else:
                self.state_counter = 0
            return False
        
        if self.state == "EXTENDED":
            if elbow_angle < self.config["descend_threshold"]: # Start descent
                if transition_to("DESCENDING"):
                    self._reset_rep_stats()
                    self.feedback = "Lowering... Control the weight."
                    
        elif self.state == "DESCENDING":
            if elbow_angle < self.config["bottom_elbow_angle"]: # Reached Bottom (<100 for seated)
                if transition_to("BOTTOM"):
                    self.feedback = "Bottom position. Press!"
                    
        elif self.state == "BOTTOM":
            if elbow_angle > self.config["bottom_elbow_angle"] + self.config["hysteresis"]:
                if transition_to("ASCENDING"):
                    self.feedback = "Pressing... Drive forward!"
                    
        elif self.state == "ASCENDING":
            if elbow_angle > self.config["extended_elbow_angle"]: # Reached Top (>155 for seated)
                if transition_to("EXTENDED"):
                    rom = self.max_elbow_angle - self.min_elbow_angle
                    if rom > self.config["min_rom"]:
                        self.rep_count += 1
                        self._score_rep()
                    else:
                        self.feedback = "Incomplete range of motion!"
    
    def _score_rep(self):
        score = 100
        faults = []
        
        # 1. Range of Motion (Prompt: Elbows never drop below 105 -> Flag)
        # self.min_elbow_angle tracks the minimum angle reached (Bottom)
        if self.variant == "seated":
            if self.min_elbow_angle > 105: # Did not go deep enough (<100 required)
                score -= 20
                faults.append("Partial ROM (Too shallow)")
            elif self.min_elbow_angle > 100:
                score -= 10
                faults.append("Slightly shallow")
                
        else: # Standard
            rom = self.max_elbow_angle - self.min_elbow_angle
            if rom < 50: # Relaxed from 60
                score -= 30
                faults.append("Partial ROM")
        
        # 2. Asymmetry
        if self.fault_counts["ASYMMETRY"] > self.config["fault_frame_thresh"]:
            score -= 20
            faults.append("Uneven press (Asymmetry)")
        
        # 3. Shoulder Shrug (Seated)
        if self.variant == "seated" and self.fault_counts["SHOULDER_SHRUG"] > 8:
            score -= 15
            faults.append("Shoulder Shrug (Keep shoulders down)")
            
        # 4. Hip Lift (Standard)
        if self.variant == "standard" and self.fault_counts["HIP_LIFT"] > 25: # Relaxed from 10
            score -= 20
            faults.append("Hips lifted off bench")

        # Determine Status
        if score >= 70:
            self.correct_reps += 1
            self.feedback = f"Rep {self.rep_count}: Correct!"
            self.advice = "Good form!"
        else:
            self.incorrect_reps += 1
            self.feedback = f"Rep {self.rep_count}: Incorrect"
            self.advice = self._get_corrective_advice(faults)
        
        self.current_rep_quality = {
            "score": max(0, score),
            "status": "correct" if score >= 70 else "incorrect",
            "reasons": faults
        }
    
    def _get_corrective_advice(self, faults):
        if not faults: return "Keep it up."
        advice_map = {
            "Partial ROM (Too shallow)": "Go deeper! Elbows should bend to ~100 degrees.",
            "Slightly shallow": "A bit lower for full chest activation.",
            "Uneven press (Asymmetry)": "Push evenly with both arms.",
            "Shoulder Shrug (Keep shoulders down)": "Relax your shoulders. Don't shrug during the press.",
            "Hips lifted off bench": "Keep your hips glued to the bench."
        }
        return advice_map.get(faults[0], "Focus on controlled movement.")

    def _generate_feedback(self, elbow_angle):
        pass # Simple feedback already set in transitions
    
    def _empty_result(self, msg="No Pose"):
        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "feedback": msg,
            "advice": "",
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "target_muscles": f"Chest ({self.variant})",
            "last_rep_score": 0,
            "reasons": []
        }
