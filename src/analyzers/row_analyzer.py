import numpy as np
import time
from src.core.utils import (
    lm_xyzv, angle_3pts, center_point, angle_with_vertical, choose_side_by_visibility, calculate_angle
)

class RepCounter:
    def __init__(self, pull_threshold=145, extend_threshold=155, min_frames_between=1):
        self.pull_th = pull_threshold
        self.extend_th = extend_threshold
        self.min_frames_between = min_frames_between
        self.state = "DOWN"
        self.reps = 0
        self.last_rep_frame = -10_000
        self.hit_top = False

    def update(self, elbow_angle, frame_idx):
        # DOWN -> UP when pulled enough
        if self.state == "DOWN":
            if elbow_angle <= self.pull_th and (frame_idx - self.last_rep_frame) >= self.min_frames_between:
                self.state = "UP"
                self.hit_top = True

        # UP -> DOWN when extended enough (rep completed)
        elif self.state == "UP":
            if elbow_angle >= self.extend_th and self.hit_top:
                self.reps += 1
                self.last_rep_frame = frame_idx
                self.state = "DOWN"
                self.hit_top = False

        return self.reps, self.state

class RowAnalyzer:
    """
    Bent-over row analyzer
    - Uses elbow flexion as rep signal
    - Checks torso swing + hinge loss + ROM completeness + asymmetry
    """

    def __init__(self, pull_threshold=145, extend_threshold=155):
        self.rep_counter = RepCounter(pull_threshold=pull_threshold, extend_threshold=extend_threshold, min_frames_between=1)
        self.trunk_baseline = None
        self.trunk_baseline_alpha = 0.05  # EMA
        self.last_state = "DOWN"
        self.frame_idx = 0
        self.last_feedback = []
        self.elbow_ema = None
        self.ema_alpha = 0.4
        
        # Persistent trackers for current rep quality
        self.swing_detected = False
        self.hinge_lost = False
        self.asymmetry_detected = False
        self.rep_min_elbow = 999.0
        self.rep_max_elbow = -999.0
        
        # Consistent with other analyzers
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.feedback = "Assume bent-over position to start"
        self.advice = ""

        # thresholds
        self.swing_deg = 15.0
        self.min_lean_deg = 30.0          # too upright if trunk angle < 30°
        self.top_max_deg = 95.0           # incomplete pull if min_elbow > 95°
        self.bottom_min_deg = 135.0       # incomplete extension if max_elbow < 135°
        self.asym_deg = 20.0

    def _compute_metrics(self, lms, w, h):
        # key points
        def get_px(idx):
            l, _ = lm_xyzv(lms, idx)
            return (int(l[0] * w), int(l[1] * h))

        L_sh = get_px(11); R_sh = get_px(12)
        L_hip = get_px(23); R_hip = get_px(24)

        # Trunk vector in normalized space is still fine for angle_with_vertical
        L_sh_n, _ = lm_xyzv(lms, 11); R_sh_n, _ = lm_xyzv(lms, 12)
        L_hip_n, _ = lm_xyzv(lms, 23); R_hip_n, _ = lm_xyzv(lms, 24)
        sh_c_n = (L_sh_n + R_sh_n) / 2
        hip_c_n = (L_hip_n + R_hip_n) / 2
        trunk_vec = sh_c_n - hip_c_n
        trunk_angle = angle_with_vertical(trunk_vec)

        # elbows
        L_el = get_px(13); R_el = get_px(14)
        L_wr = get_px(15); R_wr = get_px(16)

        left_elbow = calculate_angle(L_sh, L_el, L_wr)
        right_elbow = calculate_angle(R_sh, R_el, R_wr)

        # choose best side
        side = choose_side_by_visibility(lms, left_ids=[11,13,15], right_ids=[12,14,16])
        elbow = left_elbow if side == "left" else right_elbow

        return {
            "trunk_angle": trunk_angle,
            "elbow": elbow,
            "left_elbow": left_elbow,
            "right_elbow": right_elbow,
            "side": side,
            "hip_y": float(hip_c_n[1]),
        }

    def analyze(self, landmarks, frame_width, frame_height):
        # Convert MediaPipe landmarks if necessary
        # The lm_xyzv handles both MP landmarks and numpy arrays
        
        m = self._compute_metrics(landmarks, frame_width, frame_height)
        trunk = m["trunk_angle"]
        raw_elbow = m["elbow"]

        # EMA for elbow to prevent jitter-based state flips
        if self.elbow_ema is None:
            self.elbow_ema = raw_elbow
        else:
            self.elbow_ema = (1 - self.ema_alpha) * self.elbow_ema + self.ema_alpha * raw_elbow
        
        elbow = self.elbow_ema

        # EMA baseline for trunk
        if self.trunk_baseline is None:
            if trunk > 20: # Only start baseline if they are somewhat leaned over
                self.trunk_baseline = trunk
        else:
            self.trunk_baseline = (1 - self.trunk_baseline_alpha) * self.trunk_baseline + self.trunk_baseline_alpha * trunk

        reps, state = self.rep_counter.update(elbow, self.frame_idx)
        
        # State detection for reset/evaluate
        rep_just_started = (self.last_state == "DOWN" and state == "UP")
        rep_just_completed = (self.last_state == "UP" and state == "DOWN")
        self.last_state = state

        if rep_just_started:
            self.swing_detected = False
            self.hinge_lost = False
            self.asymmetry_detected = False
            self.rep_min_elbow = elbow
            self.rep_max_elbow = elbow
            self.advice = ""

        # Update metrics during active phase
        if state == "UP":
            self.rep_min_elbow = min(self.rep_min_elbow, elbow)
            self.rep_max_elbow = max(self.rep_max_elbow, elbow)
            
            # Check for swing against baseline
            if self.trunk_baseline is not None and abs(trunk - self.trunk_baseline) > self.swing_deg:
                self.swing_detected = True
            if trunk < self.min_lean_deg:
                self.hinge_lost = True
            if abs(m["left_elbow"] - m["right_elbow"]) > self.asym_deg:
                self.asymmetry_detected = True

        feedback_list = []
        last_rep_score = 100
        reasons = []

        # Real-time feedback
        if trunk < self.min_lean_deg: feedback_list.append("Lean forward more")
        if self.swing_detected: feedback_list.append("Keep torso still")

        # Evaluate on completion
        if rep_just_completed:
            # Penalty calculation
            if self.swing_detected:
                last_rep_score -= 20
                reasons.append("Torso Swing")
            if self.hinge_lost:
                last_rep_score -= 30
                reasons.append("Too Upright")
            if self.asymmetry_detected:
                last_rep_score -= 15
                reasons.append("Asymmetry")
            
            # ROM checks
            if self.rep_min_elbow > self.top_max_deg:
                last_rep_score -= 25
                reasons.append("Pull Higher")
            if self.rep_max_elbow < self.bottom_min_deg:
                last_rep_score -= 25
                reasons.append("Extend Fully")

            if last_rep_score >= 70:
                self.correct_reps += 1
                self.advice = "Good Rep!"
            else:
                self.incorrect_reps += 1
                self.advice = ", ".join(reasons) if reasons else "Improve form"
            
            self.rep_count = reps

        self.last_feedback = feedback_list
        self.feedback = feedback_list[0] if feedback_list else ("Pull!" if self.rep_counter.state == "DOWN" else "Return")
        
        self.frame_idx += 1

        return {
            "state": state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "elbow_angle": elbow,
            "trunk_angle": trunk,
            "feedback": self.feedback,
            "advice": self.advice,
            "last_rep_score": last_rep_score if rep_just_completed else 0,
            "reasons": reasons if rep_just_completed else [],
            "target_muscles": "Back (Lats, Traps), Biceps",
            "view": "SIDE" if trunk > 30 else "FRONT"
        }

class TBarRowAnalyzer(RowAnalyzer):
    """
    T-bar row analyzer
    Typically more stable trunk; cheating often shows as bouncing/standing up.
    """

    def __init__(self):
        super().__init__(pull_threshold=145, extend_threshold=155)
        # T-bar usually allows slightly more forward lean; but must be stable
        self.swing_deg = 10.0
        self.min_lean_deg = 30.0
        self.top_max_deg = 90.0
        self.bottom_min_deg = 135.0
    
    def analyze(self, landmarks, frame_width, frame_height):
        res = super().analyze(landmarks, frame_width, frame_height)
        res["target_muscles"] = "Mid-Back, Lats, Biceps"
        return res
