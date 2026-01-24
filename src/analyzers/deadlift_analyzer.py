import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class DeadliftAnalyzer:
    """
    Biomechanical Analyzer for Conventional Deadlift.
    Phases: SETUP -> PULLING (LIFT) -> LOCKOUT -> LOWERING (RETURN)
    """
    def __init__(self):
        self.state = "SETUP"
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_rep_quality = {}
        self.feedback = ""
        self.advice = ""
        
        # Rep Validation Flags
        self.is_rep_valid = True
        self.rep_error_reason = None
        self.current_rep_faults = []
        
        # Buffers
        self.prev_vals = {} 
        self.pull_start_metrics = {"hip_y": 0, "shoulder_y": 0}
        
        # Fault Counters (Debounce)
        self.setup_fault_frames = 0
        self.hip_shoot_frames = 0
        self.drift_frames = 0
        self.rounding_frames = 0
        self.overext_frames = 0
        self.lockout_error_frames = 0
        
        # Biomechanical Thresholds (Very Lenient - Only Major Problems)
        self.config = {
            "setup_hip_angle_max": 130,     # Very relaxed: Allow higher starting position
            "setup_torso_angle_min": 5,     # Very relaxed: Allow more forward lean
            
            "lift_hips_shoot_ratio": 4.0,   # Very lenient: Only flag extreme shooting
            "lift_bar_drift_norm": 0.40,    # Drift 40% of shin length allowed (was 25%)
            "lift_rounding_angle": 5,       # Only flag severe rounding (was 10)
            
            "lockout_hip_angle_min": 150,   # More relaxed: Allow softer lockout
            "lockout_knee_angle_min": 150,  # More relaxed: Allow softer lockout
            "lockout_overext_thresh": 60,   # Very relaxed (was 45)
            
            "fault_frame_thresh": 20        # Quadrupled debounce (0.66s hold required)
        }
        
        self.mp_pose = mp.solutions.pose
        self._reset_rep_stats()
        
    def _reset_rep_stats(self):
        self.is_rep_valid = True
        self.rep_error_reason = None
        self.current_rep_faults = []
        self.pull_start_metrics = {"hip_y": 0, "shoulder_y": 0}
        self.current_rep_quality = {}
        
        # Reset counters
        self.hip_shoot_frames = 0
        self.drift_frames = 0
        self.rounding_frames = 0
        self.overext_frames = 0
        self.lockout_error_frames = 0

    def _ema_smooth(self, key, new_val):
        if key not in self.prev_vals:
            self.prev_vals[key] = new_val
            return new_val
        smooth = 0.3 * new_val + 0.7 * self.prev_vals[key]
        self.prev_vals[key] = smooth
        return smooth

    def analyze(self, landmarks, frame_width, frame_height):
        if not landmarks: return self._empty_result()
            
        # 1. Landmarks & View
        lm = self.mp_pose.PoseLandmark
        def get_px(idx): return get_landmark_pixel(landmarks[idx], frame_width, frame_height)
        
        l_vis = landmarks[lm.LEFT_HIP.value].visibility
        r_vis = landmarks[lm.RIGHT_HIP.value].visibility
        active_side = "LEFT" if l_vis > r_vis else "RIGHT"
        
        if active_side == "LEFT":
            s = get_px(lm.LEFT_SHOULDER.value)
            h = get_px(lm.LEFT_HIP.value)
            k = get_px(lm.LEFT_KNEE.value)
            a = get_px(lm.LEFT_ANKLE.value)
            w = get_px(lm.LEFT_WRIST.value)
            toe = get_px(lm.LEFT_FOOT_INDEX.value)
            heel = get_px(lm.LEFT_HEEL.value)
        else:
            s = get_px(lm.RIGHT_SHOULDER.value)
            h = get_px(lm.RIGHT_HIP.value)
            k = get_px(lm.RIGHT_KNEE.value)
            a = get_px(lm.RIGHT_ANKLE.value)
            w = get_px(lm.RIGHT_WRIST.value)
            toe = get_px(lm.RIGHT_FOOT_INDEX.value)
            heel = get_px(lm.RIGHT_HEEL.value)

        # 2. Metrics
        # Torso Angle (Verticality)
        v_point = (s[0], s[1] - 100)
        torso_angle = self._ema_smooth("torso_angle", calculate_angle(v_point, s, h))
        hip_angle = self._ema_smooth("hip_angle", calculate_angle(s, h, k))
        knee_angle = self._ema_smooth("knee_angle", calculate_angle(h, k, a))

        # 3. State Machine & Rules
        
        # Phase 1: Start Position (SETUP)
        if self.state == "SETUP":
            # Rule: Hip angle < 100 (Deep enough hinge)
            is_deep_enough = hip_angle < self.config["setup_hip_angle_max"]
            is_spine_ok = torso_angle > self.config["setup_torso_angle_min"]
            
            if not is_deep_enough:
                self.feedback = "Setup: Hips Lower."
            elif not is_spine_ok:
                self.feedback = "Setup: Chest Up!"
            else:
                self.feedback = "Ready. Pull!"
                
            # Transition to LIFT
            if is_deep_enough and (knee_angle > 90 or hip_angle > 110):
                self.state = "LIFT"
                self._reset_rep_stats()
                self.pull_start_metrics = {"hip_y": h[1], "shoulder_y": s[1]}
                self.feedback = "Drive with legs!"
                
                # Check Start Conditions Validity
                # Strict setup check removed to avoid blocking rep start
                # if not is_spine_ok:
                #    self._mark_fault("Bad Setup (Rounding)")

        # Phase 2: Concentric (LIFT)
        elif self.state == "LIFT":
            # A. Hips Shoot Up
            hip_rise = self.pull_start_metrics["hip_y"] - h[1]
            shoulder_rise = self.pull_start_metrics["shoulder_y"] - s[1]
            
            hips_shooting = False
            if hip_rise > 25 and shoulder_rise < 10: 
                 if hip_rise > (shoulder_rise * self.config["lift_hips_shoot_ratio"]):
                      hips_shooting = True
            
            if hips_shooting:
                self.hip_shoot_frames += 1
                if self.hip_shoot_frames > self.config["fault_frame_thresh"]:
                     self._mark_fault("Early Hip Rise")
                     self.feedback = "Don't shoot hips!"
            else:
                self.hip_shoot_frames = max(0, self.hip_shoot_frames - 1)
            
            # B. Bar Drift
            shin_len = np.linalg.norm(np.array(k) - np.array(a))
            drift = abs(w[0] - a[0])
            norm_drift = drift / shin_len if shin_len > 0 else 0
            
            if norm_drift > self.config["lift_bar_drift_norm"]:
                 self.drift_frames += 1
                 if self.drift_frames > self.config["fault_frame_thresh"]:
                     self._mark_fault("Bar Drift")
                     self.feedback = "Keep bar close!"
            else:
                 self.drift_frames = max(0, self.drift_frames - 1)
                 
            # C. Rounding (Dynamic)
            if torso_angle < self.config["lift_rounding_angle"]:
                 self.rounding_frames += 1
                 if self.rounding_frames > self.config["fault_frame_thresh"]:
                     self._mark_fault("Lumbar Rounding")
                     self.feedback = "Back Rounding!"
            else:
                 self.rounding_frames = max(0, self.rounding_frames - 1)

            # Transition to LOCKOUT
            if knee_angle > self.config["lockout_knee_angle_min"] and hip_angle > self.config["lockout_hip_angle_min"]:
                self.state = "LOCKOUT"
                
                # Verify Lockout (Phase 3)
                if torso_angle < 145: # Very relaxed (was 155)
                      self._mark_fault("Incomplete Lockout")
                          
                # Check Overextension
                facing_left = toe[0] < heel[0]
                is_overext = False
                if facing_left:
                    if s[0] > h[0] + self.config["lockout_overext_thresh"]: is_overext = True
                else:
                    if s[0] < h[0] - self.config["lockout_overext_thresh"]: is_overext = True
                
                if is_overext:
                    self.overext_frames += 1
                    if self.overext_frames > self.config["fault_frame_thresh"]: # Very clear lean needed
                        self._mark_fault("Overextension")
                        self.feedback = "Don't lean back!"
                else:
                    self.overext_frames = 0
                    self.feedback = "Hold... Lower slow."

            elif knee_angle < 70: 
                self.state = "SETUP"

        # Phase 3: Lockout (Hold)
        elif self.state == "LOCKOUT":
             # Transition to RETURN
             if knee_angle < (self.config["lockout_knee_angle_min"] - 10):
                 self.state = "RETURN"
                 self.feedback = "Hips back first."

        # Phase 4: Return (LOWERING)
        elif self.state == "RETURN":
             # Check for "Squatting it down" (Knees first) - Optional
             # Transition to SETUP (Complete Rep)
             if knee_angle < self.config["setup_hip_angle_max"] or hip_angle < 110:
                 self.state = "SETUP"
                 self.feedback = "Rep Complete."
                 
                 # 🏁 FINALIZE REP COUNT 🏁
                 if self.is_rep_valid:
                     self.correct_reps += 1
                     self.rep_count += 1
                     self.advice = "Clean rep! Well done."
                     # Score: 100
                     self.current_rep_quality = {"score": 100, "status": "correct", "reasons": []}
                 else:
                     self.incorrect_reps += 1
                     # Do NOT increment rep_count
                     self.advice = f"Form error: {self.rep_error_reason}. Focus on technique."
                     self.current_rep_quality = {"score": 0, "status": "incorrect", "reasons": self.current_rep_faults}

        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "feedback": self.feedback,
            "advice": self.advice,
            "last_rep_score": self.current_rep_quality.get("score", 0),
            "reasons": self.current_rep_quality.get("reasons", []),
            "target_muscles": "Post. Chain",
            "view": f"Side ({active_side})"
        }

    def _mark_fault(self, reason):
        # Strict: One strike and you're out
        if self.is_rep_valid:
            self.is_rep_valid = False
            self.rep_error_reason = reason
        if reason not in self.current_rep_faults:
            self.current_rep_faults.append(reason)
            
    def _score_rep(self): pass # Deprecated in strict mode
    def _get_feedback_advice(self, faults): pass 
    def _empty_result(self, msg="No Pose"):
        return {"state": self.state, "rep_count": self.rep_count, "feedback": msg, "advice": "", "correct_reps": 0, "incorrect_reps": 0}
