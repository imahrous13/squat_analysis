import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class SquatAnalyzer:
    def __init__(self):
        # State Machine
        self.state = "START" # START, STANDING, DESCENDING, BOTTOM, ASCENDING
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_rep_quality = {}
        self.feedback = "Stand straight to start"
        self.advice = ""
        self.state_counter = 0 
        self.state_transition_threshold = 2 # Moderate debounce
        
        # Biomechanical Thresholds
        # 180 is straight leg.
        self.stand_angle = 165      # Must be very straight to start/finish
        self.descend_angle = 160    # Begin descent
        self.bottom_angle = 130     # Minimum depth (Partial squat) - Parallel is ~90-100
        
        # Vertical Hip Tracking
        self.rep_start_hip_y = 0
        self.min_hip_y = 0
        self.max_hip_y = 0
        
        # Quality Metrics Data
        self.min_knee_angle = 180
        self.rep_start_time = 0
        self.bottom_start_time = 0
        self.descent_duration = 0
        self.knee_valgus_flags = 0
        self.knee_over_toes_flags = 0
        self.heel_lift_flags = 0
        self.back_angle_flags = 0
        
        self.mp_pose = mp.solutions.pose
        self._reset_rep_stats()

    def _reset_rep_stats(self):
        self.min_knee_angle = 180
        self.rep_start_time = time.time()
        self.descent_duration = 0
        self.bottom_start_time = 0
        self.knee_valgus_flags = 0
        self.back_angle_flags = 0
        self.knee_over_toes_flags = 0
        self.heel_lift_flags = 0
        self.current_rep_quality = {}
        self.min_hip_y = 0
        self.max_hip_y = 10000

    def analyze(self, landmarks, frame_width, frame_height): 
        if not landmarks:
            return self._empty_result("No person detected")

        # 1. Landmark Extraction & Visibility Check
        mp_pose = self.mp_pose.PoseLandmark
        
        required_indices = [
            mp_pose.LEFT_HIP, mp_pose.RIGHT_HIP,
            mp_pose.LEFT_KNEE, mp_pose.RIGHT_KNEE,
            mp_pose.LEFT_ANKLE, mp_pose.RIGHT_ANKLE,
            mp_pose.LEFT_SHOULDER, mp_pose.RIGHT_SHOULDER
        ]
        
        # Relaxed check: Only need hips/knees really
        if not (landmarks[mp_pose.LEFT_HIP.value].visibility > 0.5 or landmarks[mp_pose.RIGHT_HIP.value].visibility > 0.5):
             return self._empty_result("Hips not visible")

        # Get Coordinates (Pixels)
        l_hip = get_landmark_pixel(landmarks[mp_pose.LEFT_HIP.value], frame_width, frame_height)
        r_hip = get_landmark_pixel(landmarks[mp_pose.RIGHT_HIP.value], frame_width, frame_height)
        l_knee = get_landmark_pixel(landmarks[mp_pose.LEFT_KNEE.value], frame_width, frame_height)
        r_knee = get_landmark_pixel(landmarks[mp_pose.RIGHT_KNEE.value], frame_width, frame_height)
        l_ankle = get_landmark_pixel(landmarks[mp_pose.LEFT_ANKLE.value], frame_width, frame_height)
        r_ankle = get_landmark_pixel(landmarks[mp_pose.RIGHT_ANKLE.value], frame_width, frame_height)
        l_shoulder = get_landmark_pixel(landmarks[mp_pose.LEFT_SHOULDER.value], frame_width, frame_height)
        r_shoulder = get_landmark_pixel(landmarks[mp_pose.RIGHT_SHOULDER.value], frame_width, frame_height)

        # 2. Key Metrics Calculation
        # Knee Angles
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_knee_angle + r_knee_angle) / 2
        
        # Vertical Hip Position (Y increases DOWN)
        current_hip_y = (l_hip[1] + r_hip[1]) / 2
        current_knee_y = (l_knee[1] + r_knee[1]) / 2
        
        # View Detection
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        hip_width = abs(l_hip[0] - r_hip[0])
        view = "FRONT" if shoulder_width > abs(l_shoulder[1] - l_hip[1]) * 0.4 else "SIDE"

        # 3. State Machine
        current_time = time.time()
        
        if self.state == "START":
            # Must be standing (knees extended) to initialize
            if avg_knee_angle > self.stand_angle:
                self.state = "STANDING"
                self.feedback = "Ready. Squat down."
            else:
                self.feedback = f"Stand straight ({int(avg_knee_angle)}/{self.stand_angle})"
                
        elif self.state == "STANDING":
            # Check for descent
            if avg_knee_angle < self.descend_angle:
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold:
                    self.state = "DESCENDING"
                    self._reset_rep_stats()
                    self.rep_start_hip_y = current_hip_y
                    self.rep_start_time = current_time
                    self.feedback = "Descending..."
                    self.state_counter = 0
            else:
                self.state_counter = 0
                
        elif self.state == "DESCENDING":
            self.min_knee_angle = min(self.min_knee_angle, avg_knee_angle)
            
            # Check for Bottom
            # Either angle is low enough OR hips are near knee level
            # Tolerance adjusted for resolution (approx 20% of frame height)
            tolerance = frame_height * 0.2
            hips_at_knee_level = current_hip_y > (current_knee_y - tolerance)
            
            if avg_knee_angle < self.bottom_angle or hips_at_knee_level:
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold:
                     self.state = "BOTTOM"
                     self.bottom_start_time = current_time
                     self.feedback = "Good depth! Push up."
                     self.state_counter = 0
            
            # Abort if they stand up without going deep
            elif avg_knee_angle > self.descend_angle + 5:
                # Debounce abort
                self.state = "STANDING"
                self.feedback = "Go deeper."

        elif self.state == "BOTTOM":
            self.min_knee_angle = min(self.min_knee_angle, avg_knee_angle)
            
            # Check for ascent (hysteresis)
            if avg_knee_angle > self.bottom_angle + 10:
                self.state = "ASCENDING"
                self.feedback = "Drive up!"
                
        elif self.state == "ASCENDING":
            # Check for lockout (Standing)
            is_standing = avg_knee_angle > self.stand_angle
            
            # Timeout check (stuck in ascent)
            if (current_time - self.bottom_start_time) > 4.0 and avg_knee_angle > 150:
                 is_standing = True

            if is_standing:
                self.state_counter += 1
                if self.state_counter > self.state_transition_threshold:
                    # VALIDATE REP
                    # 1. Vertical Displacement: Hips must drop significantly
                    # Compare start Hip Y vs Lowest Hip Y (Deepest point)
                    # Note: We didn't track min_hip_y properly in this simplified version, let's assume Angle is proxy
                    
                    depth_score = 0
                    if self.min_knee_angle < 100: depth_score = 100 # Perfect
                    elif self.min_knee_angle < 120: depth_score = 80 # Parallel-ish
                    elif self.min_knee_angle < 135: depth_score = 50 # Partial
                    else: depth_score = 0 # Shallow
                    
                    rep_duration = current_time - self.rep_start_time
                    
                    if depth_score > 0 and rep_duration > 1.0:
                        self.rep_count += 1
                        
                        # Determine Rep Quality
                        has_valgus = self.knee_valgus_flags > 10
                        is_good_depth = depth_score >= 80
                        
                        if is_good_depth and not has_valgus:
                            self.correct_reps += 1
                        else:
                            self.incorrect_reps += 1
                        
                        faults = []
                        if not is_good_depth: faults.append("Go Deeper")
                        if has_valgus: faults.append("Knee Cave")
                        
                        self.feedback = f"Rep {self.rep_count} Counted!"
                        self.advice = ", ".join(faults) if faults else "Good Rep!"
                        
                    else:
                        self.feedback = "Rep too shallow or fast"
                        
                    self.state = "STANDING"
                    self.state_counter = 0

            # Fallback to bottom if they dip again
            elif avg_knee_angle < self.bottom_angle:
                self.state = "BOTTOM"

        # Continuous Fault Checks
        if view == "FRONT":
            knee_dist = abs(l_knee[0] - r_knee[0])
            ankle_dist = abs(l_ankle[0] - r_ankle[0])
            if knee_dist < ankle_dist * 0.65:
                self.knee_valgus_flags += 1

        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "l_knee_angle": l_knee_angle,
            "r_knee_angle": r_knee_angle,
            "torso_angle": 0,
            "feedback": self.feedback,
            "last_rep_score": self.current_rep_quality.get("score", 0),
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "advice": self.advice,
            "target_muscles": "Quadriceps, Glutes",
            "valgus_detected": self.knee_valgus_flags > 5,
            "view": view
        }

    def _empty_result(self, msg):
        return {
            "state": "IDLE", "rep_count": self.rep_count,
            "l_knee_angle": 0, "r_knee_angle": 0, "feedback": msg,
            "correct_reps": self.correct_reps, "incorrect_reps": self.incorrect_reps
        }
