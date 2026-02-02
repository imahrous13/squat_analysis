import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class ChestFlyAnalyzer:
    """
    Biomechanical Analyzer for Chest Flys.
    Variants: "standing" (Cable), "seated" (Machine/Pec Deck)
    """
    def __init__(self, variant="standing"):
        self.variant = variant.lower()
        self.state = "START" # START, OPEN, CLOSING, CLOSED, OPENING
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_rep_quality = {}
        self.feedback = "Open arms to start"
        self.advice = ""
        
        self.state_counter = 0
        self.state_transition_threshold = 2
        
        # Tracking Variables
        self.rep_start_time = 0
        self.min_wrist_dist = 1.0 # Min distance achieved in rep (Squeeze)
        self.max_wrist_dist = 0.0 # Max distance (Stretch)
        self.elbow_angles = []    # Track to detect "pressing"
        self.elbow_rom = 0        # Difference between max/min elbow angle
        
        # Configs
        self.open_thresh = 1.1   # Arms wide (Normalized > 1 means wider than shoulders)
        self.close_thresh = 0.5  # Wrists touching or close
        
        if self.variant == "seated":
            self.open_thresh = 0.9
            self.close_thresh = 0.45
            
        self.mp_pose = mp.solutions.pose
        self._reset_rep_stats()

    def _reset_rep_stats(self):
        self.min_wrist_dist = 10.0
        self.max_wrist_dist = 0.0
        self.elbow_angles = []
        self.current_rep_quality = {}
        self.rep_start_time = time.time()
        self.elbow_variance_flags = 0
        self.posture_flags = 0
        self.elbow_rom = 0

    def analyze(self, landmarks, w, h):
        if not landmarks: return self._empty_result("No person detected")
        
        lm = self.mp_pose.PoseLandmark
        
        # 1. Landmark Extraction
        try:
            l_sh = get_landmark_pixel(landmarks[lm.LEFT_SHOULDER.value], w, h)
            r_sh = get_landmark_pixel(landmarks[lm.RIGHT_SHOULDER.value], w, h)
            l_el = get_landmark_pixel(landmarks[lm.LEFT_ELBOW.value], w, h)
            r_el = get_landmark_pixel(landmarks[lm.RIGHT_ELBOW.value], w, h)
            l_wr = get_landmark_pixel(landmarks[lm.LEFT_WRIST.value], w, h)
            r_wr = get_landmark_pixel(landmarks[lm.RIGHT_WRIST.value], w, h)
            l_hip = get_landmark_pixel(landmarks[lm.LEFT_HIP.value], w, h)
        except:
             return self._empty_result("Landmarks missing")

        # 2. Metrics
        # Normalized Distance (Wrists vs Shoulders)
        shoulder_width = np.linalg.norm(np.array(l_sh) - np.array(r_sh))
        wrist_dist_px = np.linalg.norm(np.array(l_wr) - np.array(r_wr))
        
        if shoulder_width == 0: return self._empty_result("Error")
        
        norm_dist = wrist_dist_px / shoulder_width
        
        # Elbow Angles (Rigidity check)
        l_angle = calculate_angle(l_sh, l_el, l_wr)
        r_angle = calculate_angle(r_sh, r_el, r_wr)
        avg_elbow_angle = (l_angle + r_angle) / 2
        
        # Torso Check (Standing only)
        is_upright = True
        if self.variant == "standing":
            torso_angle = calculate_angle((l_hip[0], l_hip[1]-100), l_hip, l_sh)
            if torso_angle < 150: # Leaning forward too much or swaying
                 is_upright = False
        
        # 3. State Machine
        current_time = time.time()
        
        # Capture metrics during movement
        if self.state in ["CLOSING", "CLOSED", "OPENING"]:
            self.elbow_angles.append(avg_elbow_angle)
            self.min_wrist_dist = min(self.min_wrist_dist, norm_dist)
            self.max_wrist_dist = max(self.max_wrist_dist, norm_dist)

        if self.state == "START":
            if norm_dist > self.open_thresh:
                self.state = "OPEN"
                self.feedback = "Ready. Squeeze chest!"
            else:
                 self.feedback = f"Open arms wide ({norm_dist:.2f}/{self.open_thresh})"

        elif self.state == "OPEN":
            if norm_dist < self.open_thresh - 0.1:
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold:
                    self.state = "CLOSING"
                    self._reset_rep_stats()
                    self.feedback = "Squeeze..."
                    self.state_counter = 0
            else:
                self.state_counter = 0
                
        elif self.state == "CLOSING":
            if norm_dist < self.close_thresh:
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold: 
                    self.state = "CLOSED"
                    self.feedback = "Hold squeeze!"
                    self.state_counter = 0
            elif norm_dist > self.open_thresh + 0.1: # Abort
                self.state = "OPEN"
                self.feedback = "Resetting..."
                
        elif self.state == "CLOSED":
            if norm_dist > self.close_thresh + 0.15:
                self.state = "OPENING"
                self.feedback = "Control the negative"
        
        elif self.state == "OPENING":
            if norm_dist > self.open_thresh:
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold:
                    # VALIDATE REP
                    self._score_rep(avg_elbow_angle)
                    self.state = "OPEN"
                    self.state_counter = 0
            elif norm_dist < self.close_thresh: # Went back to squeeze?
                 self.state = "CLOSED"

        # Continuous Form Checks
        if not is_upright: self.posture_flags += 1

        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "feedback": self.feedback,
            "advice": self.advice,
            "target_muscles": "Pectoralis Major",
            "last_rep_score": self.current_rep_quality.get("score", 0),
            "reasons": self.current_rep_quality.get("faults", []),
            "wrist_dist": norm_dist # Useful for debugging
        }

    def _score_rep(self, current_angle):
        score = 100
        faults = []
        
        # 1. Range of Motion (Squeeze)
        # Did they actually touch hands or come close?
        if self.min_wrist_dist > self.close_thresh + 0.1:
            score -= 30
            faults.append("Touch hands together")
        
        # 2. Pressing vs Flying (Elbow variance)
        if len(self.elbow_angles) > 5:
            min_ang = min(self.elbow_angles)
            max_ang = max(self.elbow_angles)
            self.elbow_rom = max_ang - min_ang
            
            # If elbow angle changes significantly (> 35 deg), it's a PRESS not a FLY
            if self.elbow_rom > 35:
                score -= 40
                faults.append("Keep arms rigid (Don't press)")
            elif min_ang < 100: # Arms too bent throughout
                score -= 10
                faults.append("Open elbows more")
                
        # 3. Posture
        if self.posture_flags > 20:
            score -= 10
            faults.append("Stand straight")
            
        self.current_rep_quality = {
            "score": max(0, score),
            "faults": faults
        }
        
        if score >= 60:
            self.rep_count += 1
            self.correct_reps += 1
            self.advice = "Great contraction!"
        else:
            self.rep_count += 1
            self.incorrect_reps += 1
            self.advice = faults[0] if faults else "Improve form"
            
        self.feedback = f"Rep {self.rep_count}: {self.advice}"

    def _empty_result(self, msg="No Pose"):
        return {
            "state": "IDLE", "rep_count": self.rep_count, 
            "feedback": msg, "target_muscles": "Chest",
            "correct_reps": self.correct_reps, "incorrect_reps": self.incorrect_reps
        }
