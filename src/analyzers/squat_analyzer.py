import time
import numpy as np
import mediapipe as mp
from src.core.utils import (
    calculate_angle, get_landmark_pixel, choose_side_by_visibility,
    ema_smooth, debounced_transition, SustainedFaultTracker,
)

class SquatAnalyzer:
    def __init__(self):
        self.state = "START"
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_rep_quality = {}
        self.feedback = "Stand straight to start"
        self.advice = ""
        self.state_counter = 0
        self.state_transition_threshold = 3

        self.stand_angle = 165
        self.descend_angle = 160
        self.bottom_angle = 130

        self.smoothing_alpha = 0.35
        self.smoothed_knee_angle = None
        self.locked_side = None

        self.rep_start_hip_y = 0
        self.min_hip_y = 0
        self.max_hip_y = 0

        self.min_knee_angle = 180
        self.rep_start_time = 0
        self.bottom_start_time = 0
        self.descent_duration = 0
        self.knee_valgus_flags = 0
        self.knee_over_toes_flags = 0
        self.heel_lift_flags = 0
        self.back_angle_flags = 0

        self.valgus_tracker = SustainedFaultTracker(frames_required=5, window=10)
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
        self.valgus_tracker.reset()

    def _pick_active_side(self, landmarks, mp_pose):
        """Lock side once per rep cycle; only refresh when standing between reps."""
        if self.locked_side is not None:
            return self.locked_side
        left_ids = [mp_pose.LEFT_HIP.value, mp_pose.LEFT_KNEE.value, mp_pose.LEFT_ANKLE.value]
        right_ids = [mp_pose.RIGHT_HIP.value, mp_pose.RIGHT_KNEE.value, mp_pose.RIGHT_ANKLE.value]
        self.locked_side = choose_side_by_visibility(landmarks, left_ids, right_ids)
        return self.locked_side

    def _knee_angle_for_fsm(self, landmarks, mp_pose, l_angles, r_angles, view):
        if view == "FRONT":
            return (l_angles + r_angles) / 2
        side = self._pick_active_side(landmarks, mp_pose)
        return l_angles if side == "left" else r_angles

    def analyze(self, landmarks, frame_width, frame_height):
        if not landmarks:
            return self._empty_result("No person detected")

        mp_pose = self.mp_pose.PoseLandmark

        if not (landmarks[mp_pose.LEFT_HIP.value].visibility > 0.5
                or landmarks[mp_pose.RIGHT_HIP.value].visibility > 0.5):
            return self._empty_result("Hips not visible")

        l_hip = get_landmark_pixel(landmarks[mp_pose.LEFT_HIP.value], frame_width, frame_height)
        r_hip = get_landmark_pixel(landmarks[mp_pose.RIGHT_HIP.value], frame_width, frame_height)
        l_knee = get_landmark_pixel(landmarks[mp_pose.LEFT_KNEE.value], frame_width, frame_height)
        r_knee = get_landmark_pixel(landmarks[mp_pose.RIGHT_KNEE.value], frame_width, frame_height)
        l_ankle = get_landmark_pixel(landmarks[mp_pose.LEFT_ANKLE.value], frame_width, frame_height)
        r_ankle = get_landmark_pixel(landmarks[mp_pose.RIGHT_ANKLE.value], frame_width, frame_height)
        l_shoulder = get_landmark_pixel(landmarks[mp_pose.LEFT_SHOULDER.value], frame_width, frame_height)
        r_shoulder = get_landmark_pixel(landmarks[mp_pose.RIGHT_SHOULDER.value], frame_width, frame_height)

        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        view = "FRONT" if shoulder_width > abs(l_shoulder[1] - l_hip[1]) * 0.4 else "SIDE"

        raw_knee_angle = self._knee_angle_for_fsm(landmarks, mp_pose, l_knee_angle, r_knee_angle, view)
        self.smoothed_knee_angle = ema_smooth(self.smoothed_knee_angle, raw_knee_angle, self.smoothing_alpha)
        knee_angle = self.smoothed_knee_angle

        current_hip_y = (l_hip[1] + r_hip[1]) / 2
        current_knee_y = (l_knee[1] + r_knee[1]) / 2

        current_time = time.time()

        if self.state == "START":
            if knee_angle > self.stand_angle:
                self.state = "STANDING"
                self.feedback = "Ready. Squat down."
            else:
                self.feedback = f"Stand straight ({int(knee_angle)}/{self.stand_angle})"

        elif self.state == "STANDING":
            self.state_counter, triggered = debounced_transition(
                self.state_counter, knee_angle < self.descend_angle, self.state_transition_threshold)
            if triggered:
                self.state = "DESCENDING"
                self._pick_active_side(landmarks, mp_pose)
                self._reset_rep_stats()
                self.rep_start_hip_y = current_hip_y
                self.rep_start_time = current_time
                self.feedback = "Descending..."
                self.state_counter = 0

        elif self.state == "DESCENDING":
            self.min_knee_angle = min(self.min_knee_angle, knee_angle)
            hips_at_knee_level = current_hip_y > (current_knee_y - 100)

            self.state_counter, triggered = debounced_transition(
                self.state_counter,
                knee_angle < self.bottom_angle or hips_at_knee_level,
                self.state_transition_threshold)
            if triggered:
                self.state = "BOTTOM"
                self.bottom_start_time = current_time
                self.feedback = "Good depth! Push up."
                self.state_counter = 0
            elif knee_angle > self.descend_angle + 5:
                self.state = "STANDING"
                self.locked_side = None
                self.smoothed_knee_angle = None
                self.feedback = "Go deeper."
                self.state_counter = 0

        elif self.state == "BOTTOM":
            self.min_knee_angle = min(self.min_knee_angle, knee_angle)
            if knee_angle > self.bottom_angle + 10:
                self.state = "ASCENDING"
                self.feedback = "Drive up!"

        elif self.state == "ASCENDING":
            is_standing = knee_angle > self.stand_angle
            if (current_time - self.bottom_start_time) > 4.0 and knee_angle > 150:
                is_standing = True

            self.state_counter, triggered = debounced_transition(
                self.state_counter, is_standing, self.state_transition_threshold)
            if triggered:
                depth_score = 0
                if self.min_knee_angle < 100:
                    depth_score = 100
                elif self.min_knee_angle < 120:
                    depth_score = 80
                elif self.min_knee_angle < 135:
                    depth_score = 50

                rep_duration = current_time - self.rep_start_time

                if depth_score > 0 and rep_duration > 1.0:
                    self.rep_count += 1
                    has_valgus = self.knee_valgus_flags > 10
                    is_good_depth = depth_score >= 80

                    if is_good_depth and not has_valgus:
                        self.correct_reps += 1
                    else:
                        self.incorrect_reps += 1

                    faults = []
                    if not is_good_depth:
                        faults.append("Go Deeper")
                    if has_valgus:
                        faults.append("Knee Cave")

                    self.feedback = f"Rep {self.rep_count} Counted!"
                    self.advice = ", ".join(faults) if faults else "Good Rep!"
                else:
                    self.feedback = "Rep too shallow or fast"

                self.state = "STANDING"
                self.locked_side = None
                self.smoothed_knee_angle = None
                self.state_counter = 0
            elif knee_angle < self.bottom_angle:
                self.state = "BOTTOM"

        valgus_now = False
        if view == "FRONT":
            knee_dist = abs(l_knee[0] - r_knee[0])
            ankle_dist = abs(l_ankle[0] - r_ankle[0])
            if ankle_dist > 0 and knee_dist < ankle_dist * 0.65:
                valgus_now = True
                self.knee_valgus_flags += 1

        valgus_sustained = self.valgus_tracker.update(valgus_now)

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
            "valgus_detected": valgus_sustained,
            "view": view,
        }

    def _empty_result(self, msg):
        return {
            "state": "IDLE", "rep_count": self.rep_count,
            "l_knee_angle": 0, "r_knee_angle": 0, "feedback": msg,
            "correct_reps": self.correct_reps, "incorrect_reps": self.incorrect_reps,
            "advice": self.advice, "target_muscles": "Quadriceps, Glutes",
        }
