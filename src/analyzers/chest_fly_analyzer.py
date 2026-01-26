import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class ChestFlyAnalyzer:
    """
    Biomechanical Analyzer for Chest Flys (Cable or Machine).
    Variants: 
    - "standing": Cable Flys (Standing)
    - "seated": Pec Deck / Machine Fly (Seated)
    
    Phases: OPENED -> CLOSING -> SQUEEZE -> OPENING -> OPENED
    """
    def __init__(self, variant="standing"):
        self.variant = variant.lower()
        self.state = "OPENED"
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_rep_quality = {}
        self.feedback = ""
        self.advice = ""
        
        self.state_counter = 0
        # Reduce threshold for machine flys to be more responsive to small movements
        self.state_transition_threshold = 2 if self.variant == "seated" else 3
        
        # Tracking
        self.max_wrist_dist = 0
        self.min_wrist_dist = 1000
        self.prev_vals = {}
        
        self.config = {
            "open_dist_thresh": 0.8,  # Normalized by shoulder width
            "closed_dist_thresh": 0.3,
            "max_elbow_flexion": 150, 
            "min_elbow_flexion": 90,  # Relaxed
            "max_torso_lean": 30,     
            "smoothing_alpha": 0.4,   # Faster smoothing
            "fault_frame_thresh": 10
        }
        
        self.fault_counts = {
            "ELBOW_BENT_TOO_MUCH": 0,
            "PARTIAL_ROM": 0,
            "EXCESSIVE_LEAN": 0,
            "SHRUGGING": 0
        }
        
        self.mp_pose = mp.solutions.pose
        self._reset_rep_stats()

    def _reset_rep_stats(self):
        self.max_wrist_dist = 0
        self.min_wrist_dist = 1000
        self.fault_counts = {k: 0 for k in self.fault_counts}
        self.current_rep_quality = {}

    def analyze(self, landmarks, w, h):
        if not landmarks: return self._empty_result()
        
        lm = self.mp_pose.PoseLandmark
        def get_px(idx): return get_landmark_pixel(landmarks[idx], w, h)
        
        l_sh, r_sh = get_px(lm.LEFT_SHOULDER.value), get_px(lm.RIGHT_SHOULDER.value)
        l_el, r_el = get_px(lm.LEFT_ELBOW.value), get_px(lm.RIGHT_ELBOW.value)
        l_wr, r_wr = get_px(lm.LEFT_WRIST.value), get_px(lm.RIGHT_WRIST.value)
        l_hip, r_hip = get_px(lm.LEFT_HIP.value), get_px(lm.RIGHT_HIP.value)
        
        # 1. Calculate Metrics
        # Composite Distance (X + Z depth)
        # Machine flys often have handles that move in an arc, so X-distance alone can be tricky
        l_sh_np, r_sh_np = np.array(l_sh), np.array(r_sh)
        l_wr_np, r_wr_np = np.array(l_wr), np.array(r_wr)
        
        shoulder_width = np.linalg.norm(l_sh_np - r_sh_np)
        
        # We use X-distance primarily but also consider Z-depth (visibility/perspective)
        # Normalized Horizontal distance
        dx = abs(l_wr[0] - r_wr[0])
        norm_dist = dx / shoulder_width if shoulder_width > 0 else 1.0
        
        # Smooth the distance to prevent jitter-based state skipping
        norm_dist = self._ema_smooth("norm_dist", norm_dist)
        
        # Elbow Angles (Internal Squeeze)
        l_elbow_angle = calculate_angle(l_sh, l_el, l_wr)
        r_elbow_angle = calculate_angle(r_sh, r_el, r_wr)
        avg_elbow_angle = (l_elbow_angle + r_elbow_angle) / 2
        
        # Torso Angle (Lean)
        mid_sh = np.mean([l_sh, r_sh], axis=0)
        mid_hip = np.mean([l_hip, r_hip], axis=0)
        # Angle relative to vertical
        vertical_vector = np.array([0, -100])
        torso_lean = calculate_angle(mid_hip + vertical_vector, mid_hip, mid_sh)

        # 2. State Machine Logic
        self._update_state(norm_dist)
        
        # 3. Form Checks
        self._check_form(avg_elbow_angle, torso_lean)
        
        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "feedback": self.feedback,
            "advice": self.advice,
            "target_muscles": "Chest (Pecs), Front Delts",
            "last_rep_score": self.current_rep_quality.get("score", 0),
            "reasons": self.current_rep_quality.get("reasons", []),
            "wrist_dist": round(norm_dist, 2)
        }

    def _ema_smooth(self, key, val):
        alpha = self.config["smoothing_alpha"]
        if key not in self.prev_vals:
            self.prev_vals[key] = val
            return val
        smooth = alpha * val + (1 - alpha) * self.prev_vals[key]
        self.prev_vals[key] = smooth
        return smooth

    def _update_state(self, dist):
        def transition(new):
            if self.state != new:
                self.state_counter += 1
                if self.state_counter >= self.state_transition_threshold:
                    self.state = new
                    self.state_counter = 0
                    return True
            return False

        # Thresholds relaxed significantly for machine fly's restricted ROM
        # Machines often don't allow full closure or full opening
        OPEN_THRESH = self.config["open_dist_thresh"] if self.variant == "standing" else 0.65
        CLOSE_THRESH = self.config["closed_dist_thresh"] if self.variant == "standing" else 0.45

        if self.state == "OPENED":
            if dist < OPEN_THRESH - 0.05:
                if transition("CLOSING"): self.feedback = "Squeeze!"
        elif self.state == "CLOSING":
            if dist < CLOSE_THRESH:
                if transition("SQUEEZE"): self.feedback = "Squeeze tight!"
        elif self.state == "SQUEEZE":
            if dist > CLOSE_THRESH + 0.1:
                if transition("OPENING"): self.feedback = "Open slowly."
        elif self.state == "OPENING":
            if dist > OPEN_THRESH:
                if transition("OPENED"):
                    self.rep_count += 1
                    self._score_rep()
                    
    def _check_form(self, elbow_angle, torso_lean):
        if elbow_angle < self.config["min_elbow_flexion"]:
            self.fault_counts["ELBOW_BENT_TOO_MUCH"] += 1
        if self.variant == "standing" and torso_lean > self.config["max_torso_lean"]:
            self.fault_counts["EXCESSIVE_LEAN"] += 1

    def _score_rep(self):
        score = 100
        reasons = []
        if self.fault_counts["ELBOW_BENT_TOO_MUCH"] > self.config["fault_frame_thresh"]:
            score -= 30
            reasons.append("Elbows bent too much (Avoid pressing)")
        if self.fault_counts["EXCESSIVE_LEAN"] > self.config["fault_frame_thresh"]:
            score -= 20
            reasons.append("Too much torso lean")
        
        if score >= 60:
            self.correct_reps += 1
            self.advice = "Great chest activation!"
        else:
            self.incorrect_reps += 1
            self.advice = reasons[0] if reasons else "Improve form."
            
        self.current_rep_quality = {"score": score, "reasons": reasons}
        self._reset_rep_stats()

    def _empty_result(self):
        return {"state": self.state, "rep_count": self.rep_count, "feedback": "No Pose", "correct_reps": self.correct_reps, "incorrect_reps": self.incorrect_reps}
