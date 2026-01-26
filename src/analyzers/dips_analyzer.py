import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class DipsAnalyzer:
    """
    Biomechanical Analyzer for Dips.
    Variants: 
    - "normal": Parallel Bar Dips (Bodyweight)
    - "seated": Seated Machine Dips / Assisted Dips
    
    Phases: TOP -> DESCENDING -> BOTTOM -> ASCENDING -> TOP
    """
    def __init__(self, variant="normal"):
        self.variant = variant.lower()
        self.state = "TOP"
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_rep_quality = {}
        self.feedback = ""
        self.advice = ""
        
        self.state_counter = 0
        self.state_transition_threshold = 4
        
        # Tracking
        self.min_elbow_angle = 180
        self.max_torso_angle = 0
        
        self.config = {
            "top_elbow_angle": 160,
            "bottom_elbow_angle": 100,      # ROM requirement
            "chest_lean_thresh": 20,        # Degrees forward from vertical for Chest focus
            "tricep_upright_thresh": 10,    # Max degrees for Tricep focus
            "smoothing_alpha": 0.3,
            "fault_frame_thresh": 15
        }
        
        self.fault_counts = {
            "PARTIAL_ROM": 0,
            "LACK_OF_LEAN": 0,  # If user claims chest focus but stays upright
            "SHRUGGING": 0,
            "ELBOW_FLARE": 0
        }
        
        self.mp_pose = mp.solutions.pose
        self._reset_rep_stats()

    def _reset_rep_stats(self):
        self.min_elbow_angle = 180
        self.max_torso_angle = 0
        self.fault_counts = {k: 0 for k in self.fault_counts}

    def analyze(self, landmarks, w, h):
        if not landmarks: return self._empty_result()
        
        lm = self.mp_pose.PoseLandmark
        def get_px(idx): return get_landmark_pixel(landmarks[idx], w, h)
        
        l_sh, r_sh = get_px(lm.LEFT_SHOULDER.value), get_px(lm.RIGHT_SHOULDER.value)
        l_el, r_el = get_px(lm.LEFT_ELBOW.value), get_px(lm.RIGHT_ELBOW.value)
        l_wr, r_wr = get_px(lm.LEFT_WRIST.value), get_px(lm.RIGHT_WRIST.value)
        l_hip, r_hip = get_px(lm.LEFT_HIP.value), get_px(lm.RIGHT_HIP.value)
        
        # 1. Angles
        # Use better visible side
        l_vis = landmarks[lm.LEFT_SHOULDER.value].visibility
        r_vis = landmarks[lm.RIGHT_SHOULDER.value].visibility
        if l_vis > r_vis:
            elbow_angle = calculate_angle(l_sh, l_el, l_wr)
            # Torso angle relative to vertical
            mid_hip = l_hip
            mid_sh = l_sh
        else:
            elbow_angle = calculate_angle(r_sh, r_el, r_wr)
            mid_hip = r_hip
            mid_sh = r_sh
            
        vertical_ref = [mid_hip[0], mid_hip[1] - 100]
        torso_angle = calculate_angle(vertical_ref, mid_hip, mid_sh)
        
        self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
        self.max_torso_angle = max(self.max_torso_angle, torso_angle)
        
        # 2. State Machine
        self._update_state(elbow_angle)
        
        # 3. Form Feedback
        focus = "Chest" if torso_angle > self.config["chest_lean_thresh"] else "Triceps"
        
        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "feedback": self.feedback,
            "advice": self.advice,
            "target_muscles": f"{focus}, Front Delts",
            "last_rep_score": self.current_rep_quality.get("score", 0),
            "reasons": self.current_rep_quality.get("reasons", []),
            "view": f"Focus: {focus}"
        }

    def _update_state(self, angle):
        def transition(new):
            if self.state != new:
                self.state_counter += 1
                if self.state_counter >= self.state_transition_threshold:
                    self.state = new
                    self.state_counter = 0
                    return True
            return False

        if self.state == "TOP":
            if angle < self.config["top_elbow_angle"]:
                if transition("DESCENDING"): self.feedback = "Lower yourself slowly."
        elif self.state == "DESCENDING":
            if angle < self.config["bottom_elbow_angle"]:
                if transition("BOTTOM"): self.feedback = "Bottom position. Push up!"
        elif self.state == "BOTTOM":
            if angle > self.config["bottom_elbow_angle"] + 10:
                if transition("ASCENDING"): self.feedback = "Drive through your palms."
        elif self.state == "ASCENDING":
            if angle > self.config["top_elbow_angle"]:
                if transition("TOP"):
                    self.rep_count += 1
                    self._score_rep()

    def _score_rep(self):
        score = 100
        reasons = []
        
        if self.min_elbow_angle > 110:
            score -= 30
            reasons.append("Partial ROM (Lower deeper for full activation)")
        
        if score >= 60:
            self.correct_reps += 1
            self.advice = "Solid rep!"
        else:
            self.incorrect_reps += 1
            self.advice = reasons[0] if reasons else "Improve depth."
            
        self.current_rep_quality = {"score": score, "reasons": reasons}
        self._reset_rep_stats()

    def _empty_result(self, msg="No Pose"):
        return {"state": self.state, "rep_count": self.rep_count, "feedback": msg, "correct_reps": self.correct_reps, "incorrect_reps": self.incorrect_reps}
