import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class PushUpAnalyzer:
    def __init__(self):
        self.state = "TOP_PLANK" 
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_rep_quality = {}
        self.feedback = ""
        self.advice = ""
        self.state_counter = 0 
        self.state_transition_threshold = 8 # Increased from 3 to prevent jitter reps
        self.config = {
            "top_elbow_angle": 160,    
            "descend_threshold": 150,  
            "bottom_elbow_angle": 90, 
            "hysteresis": 8,
            "smoothing_alpha": 0.2,   
            "sag_threshold_norm": 0.06,
            "pike_threshold_norm": 0.06
        }
        self.prev_elbow_angle = 0
        self._reset_rep_stats()
        self.mp_pose = mp.solutions.pose
        
    def _reset_rep_stats(self):
        self.min_elbow_angle = 180
        self.rep_start_time = time.time()
        self.fault_counts = {"SAGGING_HIPS": 0, "PIKE_HIPS": 0, "ELBOW_FLARE": 0}
        self.advice = ""
        self.current_rep_quality = {}

    def _ema_smooth(self, new_val, prev_val):
        if prev_val == 0: return new_val
        return (self.config["smoothing_alpha"] * new_val) + ((1 - self.config["smoothing_alpha"]) * prev_val)

    def analyze(self, landmarks, frame_width, frame_height):
        if not landmarks: return self._empty_result()
        
        lm = self.mp_pose.PoseLandmark
        def get_px(l_idx): return get_landmark_pixel(landmarks[l_idx.value], frame_width, frame_height)
        
        l_s, r_s = get_px(lm.LEFT_SHOULDER), get_px(lm.RIGHT_SHOULDER)
        l_e, r_e = get_px(lm.LEFT_ELBOW), get_px(lm.RIGHT_ELBOW)
        l_w, r_w = get_px(lm.LEFT_WRIST), get_px(lm.RIGHT_WRIST)
        l_h, r_h = get_px(lm.LEFT_HIP), get_px(lm.RIGHT_HIP)
        l_a, r_a = get_px(lm.LEFT_ANKLE), get_px(lm.RIGHT_ANKLE)

        # Determine active side
        l_vis = landmarks[lm.LEFT_SHOULDER.value].visibility
        r_vis = landmarks[lm.RIGHT_SHOULDER.value].visibility
        active_side = "LEFT" if l_vis > r_vis else "RIGHT"
        
        if active_side == "LEFT":
            elbow_angle = calculate_angle(l_s, l_e, l_w)
        else:
            elbow_angle = calculate_angle(r_s, r_e, r_w)
            
        elbow_angle = self._ema_smooth(elbow_angle, self.prev_elbow_angle)
        self.prev_elbow_angle = elbow_angle
        self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)

        # Hip Alignment Check (Body Rigidity)
        def get_body_alignment(s, h, a):
            v_sa = (a[0] - s[0], a[1] - s[1])
            v_sh = (h[0] - s[0], h[1] - s[1])
            sa_len = (v_sa[0]**2 + v_sa[1]**2)**0.5
            if sa_len == 0: return 0
            dist = (v_sh[0] * v_sa[1] - v_sh[1] * v_sa[0]) / sa_len
            return dist / sa_len 

        alignment = get_body_alignment(l_s if active_side=="LEFT" else r_s, 
                                       l_h if active_side=="LEFT" else r_h, 
                                       l_a if active_side=="LEFT" else r_a)
        
        if alignment > self.config["sag_threshold_norm"]:
            self.fault_counts["SAGGING_HIPS"] += 1
        elif alignment < -self.config["pike_threshold_norm"]:
            self.fault_counts["PIKE_HIPS"] += 1

        # State Machine
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

        if self.state == "TOP_PLANK":
            if elbow_angle < self.config["descend_threshold"]:
                if transition_to("DESCENDING"):
                    self._reset_rep_stats()
        elif self.state == "DESCENDING":
            if elbow_angle < self.config["bottom_elbow_angle"]:
                transition_to("BOTTOM")
        elif self.state == "BOTTOM":
            if elbow_angle > self.config["bottom_elbow_angle"] + self.config["hysteresis"]:
                transition_to("ASCENDING")
        elif self.state == "ASCENDING":
            if elbow_angle > self.config["top_elbow_angle"]:
                if transition_to("TOP_PLANK"):
                    if (self.config["top_elbow_angle"] - self.min_elbow_angle) > 30:
                        self.rep_count += 1
                        self._score_rep()
                    else:
                        self.feedback = "Incomplete movement!"

        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "feedback": self.feedback,
            "advice": self.advice,
            "last_rep_score": self.current_rep_quality.get("score", 0),
            "target_muscles": "Chest, Triceps, Shoulders"
        }

    def _score_rep(self):
        score = 100
        faults = []
        
        # 1. Depth (Rule 5)
        if self.min_elbow_angle > 100: 
            score -= 30 
            faults.append("Too shallow")
        elif self.min_elbow_angle > 90:
            score -= 10
            faults.append("Slightly shallow")
            
        # 2. Body Rigidity (Rule 2/3)
        if self.fault_counts["SAGGING_HIPS"] > 15: 
            score -= 15 
            faults.append("Sagging hips")
        if self.fault_counts["PIKE_HIPS"] > 15: 
            score -= 15 
            faults.append("Pike hips (butt up)")
            
        # Result
        if score >= 70: 
            self.correct_reps += 1
            self.feedback = f"Rep {self.rep_count}: Correct!"
            self.advice = "Great depth and rigidity!"
        else:
            self.incorrect_reps += 1
            self.feedback = f"Rep {self.rep_count}: Incorrect - {', '.join(faults)}"
            self.advice = "Focus on full range of motion and keeping your core tight."

        self.current_rep_quality = {
            "score": max(0, score),
            "faults": faults
        }

    def _empty_result(self, msg="No Pose"):
        return {"state": "UNKNOWN", "rep_count": self.rep_count, "feedback": msg, "advice": "", "correct_reps": self.correct_reps, "incorrect_reps": self.incorrect_reps, "target_muscles": "Chest, Triceps, Shoulders"}
