import numpy as np
from collections import deque

class ExerciseDetector:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.exercises = ["Squat", "Deadlift", "Push-Up", "Lunge", "Jumping Jacks", "Plank"]
        
    def add_frame(self, landmarks):
        if landmarks:
            self.buffer.append(landmarks)
            
    def detect(self):
        if len(self.buffer) < self.window_size:
            return None
        
        # Extract features from the buffer
        # MediaPipe Indices: 11:L_SH, 12:R_SH, 13:L_ELB, 14:R_ELB, 15:L_WR, 16:R_WR, 23:L_HIP, 24:R_HIP, 25:L_KNEE, 26:R_KNEE, 27:L_ANKLE, 28:R_ANKLE
        
        def get_std_y(indices):
            return np.std([np.mean([lm[i].y for i in indices]) for lm in self.buffer])

        def get_std_x_diff(idx1, idx2):
            return np.std([abs(lm[idx1].x - lm[idx2].x) for lm in self.buffer])

        # 1. Jumping Jacks: High lateral movement of arms AND legs
        hands_above_shoulders = [ (lm[15].y < lm[11].y or lm[16].y < lm[12].y) for lm in self.buffer ]
        arm_spread_std = get_std_x_diff(15, 16)
        leg_spread_std = get_std_x_diff(27, 28)
        wrist_y_std = get_std_y([15, 16])
        
        # Jumping Jacks: Arms and legs move in/out laterally, and arms move up/down
        # In a squat, leg_spread_std is near 0 because feet are planted.
        if (sum(hands_above_shoulders) > self.window_size * 0.3 and arm_spread_std > 0.05) or \
           (arm_spread_std > 0.1 and leg_spread_std > 0.05) or \
           (wrist_y_std > 0.1 and leg_spread_std > 0.03):
            return "Jumping Jacks"
            
        # 2. Horizontal Exercises (Push-Up / Plank)
        def is_horizontal(lm):
            # Body length (Shoulder to Ankle)
            body_y_diff = abs(lm[11].y - lm[27].y)
            body_x_diff = abs(lm[11].x - lm[27].x)
            
            # In a pushup/plank, the horizontal length (X) is much greater than the vertical height (Y)
            # In a squat, the vertical height (shoulder to ankle) is always significant
            return body_x_diff > (body_y_diff * 1.2)
            
        horizontal_frames = [ is_horizontal(lm) for lm in self.buffer ]
        if sum(horizontal_frames) > self.window_size * 0.5: # Relaxed from 0.6
            shoulder_std = get_std_y([11, 12])
            def get_elbow_angle(lm, side="L"):
                s = (lm[11].x, lm[11].y) if side=="L" else (lm[12].x, lm[12].y)
                e = (lm[13].x, lm[13].y) if side=="L" else (lm[14].x, lm[14].y)
                w = (lm[15].x, lm[15].y) if side=="L" else (lm[16].x, lm[16].y)
                import math
                a = math.atan2(w[1]-e[1], w[0]-e[0]) - math.atan2(s[1]-e[1], s[0]-e[0])
                return abs(math.degrees(a))
            elbow_std = np.std([get_elbow_angle(lm) for lm in self.buffer])
            if shoulder_std > 0.01 or elbow_std > 5.0:
                return "Push-Up"
            return "Plank"
            
        # 3. Vertical Exercises (Squat, Lunge, Deadlift)
        # Ensure we are actually vertical (Shoulders well above ankles)
        shoulder_y = np.mean([np.mean([lm[11].y, lm[12].y]) for lm in self.buffer])
        ankle_y = np.mean([np.mean([lm[27].y, lm[28].y]) for lm in self.buffer])
        if abs(shoulder_y - ankle_y) < 0.2: # Too horizontal for vertical exercises
            return None

        knee_diffs = [ abs(lm[25].y - lm[26].y) for lm in self.buffer ]
        avg_knee_diff = np.mean(knee_diffs)
        
        # Foot spread (X distance between ankles)
        ankle_dists = [ abs(lm[27].x - lm[28].x) for lm in self.buffer ]
        avg_ankle_dist = np.mean(ankle_dists)
        
        # Lunge: Significant asymmetry in knee height OR large foot spread
        if avg_knee_diff > 0.1 or avg_ankle_dist > 0.2:
            return "Lunge"
            
        # Deadlift vs Squat:
        torso_heights = [ abs(lm[11].y - lm[23].y) for lm in self.buffer ]
        avg_torso_height = np.mean(torso_heights)
        
        # Majority vote for hands being low (below hips)
        hands_low_frames = [ (lm[15].y > lm[23].y and lm[16].y > lm[24].y) for lm in self.buffer ]
        is_hands_low = sum(hands_low_frames) > self.window_size * 0.7
        
        # Deadlift: Torso goes horizontal (low vertical height) AND hands stay low
        if avg_torso_height < 0.22 and is_hands_low: # Relaxed from 0.15
            return "Deadlift"
            
        # Squat: Significant hip movement, feet planted (low leg spread std)
        hip_y_std = get_std_y([23, 24])
        if hip_y_std > 0.02:
            return "Squat"
            
        return None
