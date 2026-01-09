import time
import numpy as np
import mediapipe as mp
from src.core.utils import calculate_angle, get_landmark_pixel

class DeadliftAnalyzer:
    """
    Biomechanical Analyzer for Conventional Deadlift.
    Phases: SETUP -> PULLING -> LOCKOUT -> LOWERING
    Tracks: Hip Hinge, Knee Flexion, Torso Angle, Bar Path Proxy.
    """
    def __init__(self):
        self.state = "SETUP"
        self.rep_count = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.current_rep_quality = {}
        self.feedback = ""
        self.advice = ""
        
        # Buffers for smoothing and velocity calculation
        self.prev_vals = {} 
        self.prev_metrics = {
            "hip_y": 0,
            "shoulder_y": 0,
            "time": 0
        }
        
        # State Debounce
        self.frames_in_state = 0
        self.state_transition_threshold = 3
        
        # Configuration (Tunable Biomechanics)
        self.config = {
            # Smoothing
            "smoothing_alpha": 0.2,
            
            # State Thresholds
            "setup_knee_angle_max": 140,    # Knees must be bent to start
            "lockout_knee_angle_min": 160,  # Relaxed from 170
            "lockout_hip_angle_min": 160,   # Relaxed from 170
            "lockout_torso_angle_min": 160, # Relaxed from 165
            "hysteresis": 5,                # Deg hysteresis for transitions
            
            # Fault Thresholds (Balanced)
            "hips_shoot_ratio": 1.7,        # Balanced from 1.6/1.9
            "bar_drift_norm": 0.20,         # Balanced from 0.18/0.25
            "overextension_angle": 18,      # Balanced from 15/25
            "rounding_torso_angle": 70,     # If torso is too horizontal (<70) at lockout or midway? 
                                            # Actually rounding is geometry. Proxy: Torso angle vs phase.
        }
        
        self.mp_pose = mp.solutions.pose
        self._reset_rep_stats()
        
    def _reset_rep_stats(self):
        self.fault_counts = {
            "HIPS_SHOOT_UP": 0,
            "BAR_DRIFT": 0,
            "OVEREXTENSION": 0,
            "INCOMPLETE_LOCKOUT": 0,
            "ROUNDED_BACK": 0,
            "KNEES_FIRST_LOWERING": 0
        }
        self.pull_start_metrics = {"hip_y": 0, "shoulder_y": 0}
        self.lowering_start_knee = 180
        self.lowering_start_hip = 180
        self.current_rep_quality = {}
        self.advice = ""

    def _ema_smooth(self, key, new_val):
        if key not in self.prev_vals:
            self.prev_vals[key] = new_val
            return new_val
        smooth = (self.config["smoothing_alpha"] * new_val) + ((1 - self.config["smoothing_alpha"]) * self.prev_vals[key])
        self.prev_vals[key] = smooth
        return smooth

    def analyze(self, landmarks, frame_width, frame_height):
        if not landmarks:
            return self._empty_result()
            
        # 1. Select Active Side (Visibility Gating)
        l_vis = np.mean([landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].visibility,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].visibility,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility])
        r_vis = np.mean([landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility])
                         
        if max(l_vis, r_vis) < 0.5:
             return self._empty_result("Low Visibility")

        active_side = "LEFT" if l_vis > r_vis else "RIGHT"
        
        lm = self.mp_pose.PoseLandmark
        def get_px(idx): return get_landmark_pixel(landmarks[idx], frame_width, frame_height)
        
        # Extract Coordinates
        if active_side == "LEFT":
            s = get_px(lm.LEFT_SHOULDER.value)
            h = get_px(lm.LEFT_HIP.value)
            k = get_px(lm.LEFT_KNEE.value)
            a = get_px(lm.LEFT_ANKLE.value)
            w = get_px(lm.LEFT_WRIST.value)
        else:
            s = get_px(lm.RIGHT_SHOULDER.value)
            h = get_px(lm.RIGHT_HIP.value)
            k = get_px(lm.RIGHT_KNEE.value)
            a = get_px(lm.RIGHT_ANKLE.value)
            w = get_px(lm.RIGHT_WRIST.value)

        # 2. Compute Angles & Metrics
        # Torso Angle: Vertical = 180 (or 0 deviation). 
        # Using atan2(dx, dy) where dy is vertical. 
        # V: (0, 1). Vector S->H: (Hx-Sx, Hy-Sy).
        # We want angle against vertical UP.
        # If H is below S (Standing): Vector S->H is (0, +y). Vertical is (0, -y). Angle 180.
        # Let's use simple convention: 0 = Horizontal, 90 = Vertical Upright.
        # No, let's use standard joint convention. calculate_angle uses 3 points.
        # Virtual vertical point above shoulder: (Sx, Sy - 100)
        v_point = (s[0], s[1] - 100)
        torso_angle = calculate_angle(v_point, s, h) # 0 if H directly below S. 180 if H directly above.
        # Wait, calculate_angle(A, B, C): B is vertex.
        # If A=(0,-100), B=(0,0), C=(0,100) (Standing). Angle is 180.
        # If Bent over (C=(100,0)): Angle is 90.
        torso_angle = self._ema_smooth("torso_angle", torso_angle)
        
        # Hip Hinge (Shoulder-Hip-Knee)
        hip_angle = self._ema_smooth("hip_angle", calculate_angle(s, h, k))
        
        # Knee Flexion (Hip-Knee-Ankle)
        knee_angle = self._ema_smooth("knee_angle", calculate_angle(h, k, a))
        
        # 3. Fault Detection Logic
        
        # A. Hips Shoot Up Early (Displacement Logic)
        # Compare total vertical distance traveled since start of pull.
        # If hips traveled up significantly more than shoulders, torso is tipping over.
        if self.state == "PULLING":
            # Calculate Rise (Start - Current) -> Positive if moving Up
            hip_rise = self.pull_start_metrics["hip_y"] - h[1]
            shoulder_rise = self.pull_start_metrics["shoulder_y"] - s[1]
            
            # Check if movement has started significantly
            if hip_rise > 25: # Moved at least ~25 pixels (Increased from 15 to avoid Setup noise)
                # If Shoulders haven't moved much, or Hips moved much more
                # Ideal: shoulder_rise >= hip_rise
                # Fault: hip_rise > shoulder_rise * threshold
                if shoulder_rise < 10 or (hip_rise > (shoulder_rise * self.config["hips_shoot_ratio"])): 
                     self.fault_counts["HIPS_SHOOT_UP"] += 1

        # B. Bar Drift (Proxy)
        # Wrist X vs Ankle X (Midfoot proxy)
        shin_len = np.linalg.norm(np.array(k) - np.array(a))
        drift = abs(w[0] - a[0])
        norm_drift = drift / shin_len if shin_len > 0 else 0
        if norm_drift > self.config["bar_drift_norm"]:
            if self.state in ["PULLING", "LOWERING"]:
                self.fault_counts["BAR_DRIFT"] += 1
                
        # C. Overextension (Lockout Leaning Back)
        if self.state == "LOCKOUT":
             # Check if Shoulders are behind Hips horizontally (relative to facing)
             # If facing Left (shoulders left of hips?), 'Overextension' means Shoulders further Left?
             # Simple proxy: Torso angle vs vertical.
             # If torso_angle > 185 (hyperextended) or < 175 (if bending back means angle decreases? depends on calculation)
             # Our torso_angle calc: 0 vertical up.
             # If leaning back, angle typically stays near 0 but geometry shifts.
             pass

        # 4. State Machine (Phase Detection)
        
        is_knee_ext = knee_angle >= self.config["lockout_knee_angle_min"]
        is_hip_ext = hip_angle >= self.config["lockout_hip_angle_min"]
        is_torso_up = torso_angle >= self.config["lockout_torso_angle_min"]
        
        # Check Setup Depth (Knees bent)
        is_setup_depth = knee_angle < self.config["setup_knee_angle_max"]
        
        prev_state = self.state
        
        if self.state == "SETUP":
            # Check Setup posture (Chest Up, Hips not too high)
            if not is_setup_depth: # Knees are bent
                if torso_angle < 10: # Almost horizontal - Bad
                     self.advice = "🔴 SETUP: Chest UP! Show your logo/chest to the wall. Keep spine neutral and core braced."
                elif hip_angle > 110: # Hips too high relative to knees
                     self.advice = "🟡 SETUP: Drop hips slightly. Find the sweet spot where you can maintain tension."
                else:
                     self.advice = "✅ SETUP: Good position. Take slack out of the bar. Brace your core. Ready to pull!"
            
            if not is_setup_depth and hip_angle > 100: # Starting to extend
                 self.state = "PULLING"
                 self._reset_rep_stats()
                 self.pull_start_metrics = {"hip_y": h[1], "shoulder_y": s[1]}
                 self.feedback = "Push the floor away!"
                 
        elif self.state == "PULLING":
            self.frames_in_state += 1
            if is_knee_ext and is_hip_ext and is_torso_up:
                self.state = "LOCKOUT"
                self.feedback = "Stand Tall. Glutes Check."
                self.rep_count += 1
                self._score_rep()
            elif knee_angle < 90: 
                self.state = "SETUP"
                self.feedback = "Resetting..."
            
            # Advice during pulling - detailed feedback
            if self.fault_counts["HIPS_SHOOT_UP"] > 0:
                self.feedback = "⚠️ Leg drive! Don't let hips fly."
                self.advice = "🔴 Push the floor away with your LEGS first. Keep hips down and drive through heels."
            elif self.fault_counts["BAR_DRIFT"] > 0:
                self.feedback = "⚠️ Drag bar up legs!"
                self.advice = "🔴 Keep bar CLOSE to your body. It should almost scrape your shins and thighs."

        elif self.state == "LOCKOUT":
            # Overextension check
            # If torso goes back past vertical (e.g. angle indicates leaning back)
            # Simple check: If Shoulder X is behind Hip X (relative to facing) excessively
            # For now, simplistic advice
            self.advice = "✅ LOCKOUT: Stand tall with hips and knees fully extended. Don't lean back excessively."
            
            if knee_angle < (self.config["lockout_knee_angle_min"] - self.config["hysteresis"]):
                self.state = "LOWERING"
                self.feedback = "Hips Back First!"
                self.lowering_start_knee = knee_angle
                self.lowering_start_hip = hip_angle
                
        elif self.state == "LOWERING":
            # Check: Hips First vs Knees First
            # Ideally hip angle decreases (flexes) MORE than knee angle initially
            # or Knees shouldn't break excessively immediately.
            current_knee_flex = self.lowering_start_knee - knee_angle
            current_hip_flex = self.lowering_start_hip - hip_angle
            
            if current_knee_flex > 20 and current_hip_flex < 10:
                self.feedback = "⚠️ Hips Back! Close door with butt."
                self.advice = "🟡 Lower by pushing hips BACK first (hip hinge), then bend knees. Maintain bar path."
                self.fault_counts["KNEES_FIRST_LOWERING"] = 1 # Using loose dict key
                
            if is_setup_depth:
                self.state = "SETUP"
                self.feedback = "Reset. Breathe."

        if prev_state != self.state:
            self.frames_in_state = 0
            
        # Immediate Feedback overrides
        if self.state == "PULLING":
            if self.fault_counts["HIPS_SHOOT_UP"] > 2:
                self.feedback = "⚠️ Hips Shooting Up!"
            elif self.fault_counts["BAR_DRIFT"] > 5:
                self.feedback = "⚠️ Bar Drift (Keep Close)"
                
        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "correct_reps": self.correct_reps,
            "incorrect_reps": self.incorrect_reps,
            "feedback": self.feedback,
            "advice": self.advice, 
            "last_rep_score": self.current_rep_quality.get("score", 0),
            "view": f"SIDE ({active_side})",
            # Metrics for potential graphing
            "hip_angle": hip_angle,
            "knee_angle": knee_angle,
            "target_muscles": "Hamstrings, Glutes, Lower Back"
        }

    def _score_rep(self):
        score = 100
        faults_detected = []
        
        # 1. Map Faults to Standard Reasons
        # Thresholds: Trigger if fault happened significantly (count > threshold)
        if self.fault_counts["HIPS_SHOOT_UP"] > 6: # Balanced from 4/10
            score -= 15 # Balanced from 10/20
            faults_detected.append("hips_shoot_up_early")
            
        if self.fault_counts["BAR_DRIFT"] > 10: # Balanced from 8/15
            score -= 12 # Balanced from 10/15
            faults_detected.append("bar_drifts_forward")
            
        if self.fault_counts["OVEREXTENSION"] > 8: # Balanced from 5/12
            score -= 8 # Balanced from 5/10
            faults_detected.append("overextension_at_top")
            
        if self.fault_counts["ROUNDED_BACK"] > 8: # Balanced from 5/12
            score -= 15 # Balanced from 10/20
            faults_detected.append("rounded_back_risk")
            
        # Check incomplete lockout (angles at moment of counting)
        # Note: State machine enforces lockout angles to even Enter "LOCKOUT" state, 
        # so "incomplete_lockout" effectively means "didn't trigger rep count" usually.
        # But if we want to flag barely-made reps:
        # if self.metrics["knee_angle"] < 172: faults_detected.append("incomplete_lockout")
        
        # 2. Determine Verdict
        verdict = "correct" if len(faults_detected) == 0 else "incorrect"
        
        # 3. Calculate Confidence
        # Simulating confidence based on visibility stability and metric clarity
        # For now, high confidence if landmarks were good. 
        confidence = 0.95 # Placeholder, or derived from mean visibility
        
        # 4. Limit to top 2 reasons
        top_reasons = faults_detected[:2]
        
        # 5. Save Structured Result
        self.current_rep_quality = {
            "score": max(0, score),
            "verdict": verdict,
            "reasons": top_reasons,
            "confidence": confidence,
            "comments": ", ".join([f.replace('_', ' ').title() for f in top_reasons]) if top_reasons else "Perfect Rep!"
        }
        
        if verdict == "correct":
            self.correct_reps += 1
        else:
            self.incorrect_reps += 1
        
        # Generate detailed advice based on faults
        self.advice = self._get_feedback_advice(faults_detected)
        
    def _get_feedback_advice(self, faults):
        """
        Returns actionable advice based on detected faults for deadlift.
        Provides detailed, specific advice for each form issue.
        """
        if not faults:
            import random
            pro_tips = [
                "Perfect form! Keep that bar path tight.",
                "Excellent rep! Maintain that neutral spine.",
                "Great lockout! Drive those hips through.",
                "Textbook technique! Keep it up.",
                "Solid rep! Bar stayed close to your legs."
            ]
            return random.choice(pro_tips)
        
        # Detailed advice map with comprehensive guidance for each fault
        advice_map = {
            "hips_shoot_up_early": "🔴 HIP POSITION: Hips rising before shoulders! This means you're using your back too much. Focus on LEG DRIVE - push the floor away with your legs first. Keep hips down and drive through your heels.",
            "bar_drifts_forward": "🔴 BAR PATH: Bar drifting away from your body! Keep the bar CLOSE to your shins and legs throughout. Drag it up your legs - it should almost scrape your shins. This maintains proper leverage.",
            "overextension_at_top": "🟡 LOCKOUT: Don't lean back excessively at the top! Just stand tall with hips and knees fully extended. Leaning back puts unnecessary stress on your lower back.",
            "rounded_back_risk": "🔴 SPINE POSITION: Back rounding detected! This is dangerous. Keep your chest UP, core BRACED, and maintain a neutral spine. If you can't maintain this, reduce weight and work on mobility.",
            "knees_first_lowering": "🟡 LOWERING: Don't bend knees first! Lower by pushing hips BACK first (hip hinge), then bend knees. This maintains proper bar path and protects your back."
        }
        
        # Collect all relevant advice messages
        advice_list = []
        for fault in faults:
            if fault in advice_map:
                advice_list.append(advice_map[fault])
        
        # Return prioritized advice (most critical first) or combine if multiple issues
        if advice_list:
            # Return the first (most critical) advice, or combine if multiple critical issues
            if len(advice_list) > 1:
                return f"{advice_list[0]} | Also check: {', '.join([a.split(':')[0] for a in advice_list[1:]])}"
            return advice_list[0]
        
        return "🟡 Focus on maintaining proper form throughout the movement."
        
    def _empty_result(self, msg="No Pose"):
        return {
            "state": self.state, "rep_count": self.rep_count,
            "feedback": msg, "advice": "", "last_rep_score": 0,
            "correct_reps": self.correct_reps, "incorrect_reps": self.incorrect_reps, "view": "UNKNOWN"
        }
