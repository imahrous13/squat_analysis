import numpy as np
from collections import deque, Counter

class ExerciseDetector:
    def __init__(self, window_size=90): # Reduced from 120 to 90 for faster detection (~3 seconds)
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.confidence_votes = deque(maxlen=45)  # Reduced from 60 to 45
        self.exercises = ["Squat", "Deadlift", "Push-Up", "Bench Press", "Lunge", "Jumping Jacks", "Plank"]
        self.last_guess = None  # Track last guess for debugging
        
    def add_frame(self, landmarks):
        if landmarks:
            self.buffer.append(landmarks)
    
    def reset(self):
        """Reset the detector buffer and votes for fresh detection"""
        self.buffer.clear()
        self.confidence_votes.clear()
        self.last_guess = None
            
    def detect(self):
        # Reduced minimum buffer requirement for faster initial detection
        if len(self.buffer) < int(self.window_size * 0.7):  # 70% of window size
            return None
        
        # 1. Motion Analysis - More lenient thresholds
        hip_y = [np.mean([lm[23].y, lm[24].y]) for lm in self.buffer]
        elbow_y = [np.mean([lm[13].y, lm[14].y]) for lm in self.buffer]
        shoulder_y = [np.mean([lm[11].y, lm[12].y]) for lm in self.buffer]
        
        hip_rom = max(hip_y) - min(hip_y)
        elbow_rom = max(elbow_y) - min(elbow_y)
        shoulder_rom = max(shoulder_y) - min(shoulder_y)
        
        # Reduced movement threshold from 15% to 10% to catch smaller movements (like push-ups)
        # Include shoulder ROM for push-up detection
        # EXCEPTION: Allow static exercises (like planks) to pass through
        has_movement = hip_rom >= 0.10 or elbow_rom >= 0.10 or shoulder_rom >= 0.10
        
        # Check for horizontal position (potential plank) even with low movement
        def is_horizontal(lm):
            body_y_diff = abs(lm[11].y - lm[27].y)
            body_x_diff = abs(lm[11].x - lm[27].x)
            return body_x_diff > (body_y_diff * 1.0)
        
        horizontal_frames = [is_horizontal(lm) for lm in self.buffer]
        horizontal_ratio = sum(horizontal_frames) / len(horizontal_frames)
        is_likely_plank = horizontal_ratio > 0.7  # High horizontal ratio with low movement = plank
        
        # Filter out low movement UNLESS it's a plank or horizontal exercise
        # Relaxed to allow push-ups with smaller ROM
        is_likely_horizontal_exercise = horizontal_ratio > 0.5  # Could be push-up or plank
        if not has_movement and not is_likely_plank and not is_likely_horizontal_exercise:
            return None

        # 2. Inflection Point Check - More lenient
        # A full rep should have the min/max value somewhere in the middle, not at the ends
        def has_inflection(data, threshold=0.08):  # Reduced from 0.1 to 0.08
            if max(data) - min(data) < threshold: return False
            mid_idx = np.argmin(data) if np.mean(data) > 0.5 else np.argmax(data)
            # More lenient inflection check - allow closer to edges
            return 5 < mid_idx < (len(data) - 5)  # Reduced from 10 to 5

        # 3. Feature Extraction & Heuristics
        def get_std_y(indices):
            return np.std([np.mean([lm[i].y for i in indices]) for lm in self.buffer])

        def get_std_x_diff(idx1, idx2):
            return np.std([abs(lm[idx1].x - lm[idx2].x) for lm in self.buffer])

        hands_above_shoulders = [ (lm[15].y < lm[11].y or lm[16].y < lm[12].y) for lm in self.buffer ]
        arm_spread_std = get_std_x_diff(15, 16)
        leg_spread_std = get_std_x_diff(27, 28)
        
        current_guess = None
        
        # PRIORITY CHECK: Deadlift Detection (Before Horizontal Check)
        # Deadlifts have horizontal torso but hands are LOW (near ground), not at shoulder level
        # This prevents deadlifts from being misclassified as planks
        hands_below_hips = [(lm[15].y > lm[23].y and lm[16].y > lm[24].y) for lm in self.buffer]
        hands_below_hips_ratio = sum(hands_below_hips) / len(hands_below_hips)
        
        # Check if hips are elevated (not on ground like plank)
        hip_heights = [np.mean([lm[23].y, lm[24].y]) for lm in self.buffer]
        avg_hip_height = np.mean(hip_heights)
        
        # Check body orientation - deadlifts have more VERTICAL body, push-ups are HORIZONTAL
        def is_body_vertical(lm):
            body_y_diff = abs(lm[11].y - lm[27].y)  # Shoulder to ankle vertical distance
            body_x_diff = abs(lm[11].x - lm[27].x)  # Shoulder to ankle horizontal distance
            return body_y_diff > body_x_diff  # Vertical if height > width
        
        vertical_frames = [is_body_vertical(lm) for lm in self.buffer]
        vertical_ratio = sum(vertical_frames) / len(vertical_frames)
        
        # Deadlift indicators:
        # 1. Hands consistently below hips (>70% of frames)
        # 2. Hips are elevated (avg height < 0.75, meaning higher on screen)
        # 3. Significant hip vertical movement (ROM > 0.12)
        # 4. Body is MORE VERTICAL than horizontal (>40% vertical frames) - KEY FIX
        # 5. Body is NOT in horizontal plank position (horizontal_ratio < 0.5)
        if (hands_below_hips_ratio > 0.70 and 
            avg_hip_height < 0.75 and 
            hip_rom > 0.12 and 
            vertical_ratio > 0.40 and
            horizontal_ratio < 0.5):  # NOT horizontal like push-up
            current_guess = "Deadlift"
        
        # Jumping Jacks: High energy lateral + vertical
        # Must include leg spread/movement AND VERTICAL BODY to distinguish from Bench Press (horizontal)
        if current_guess is None and vertical_ratio > 0.6 and \
           ((sum(hands_above_shoulders) > self.window_size * 0.2 and arm_spread_std > 0.08 and leg_spread_std > 0.05) or \
            (arm_spread_std > 0.12 and leg_spread_std > 0.06)):
            current_guess = "Jumping Jacks"
            
        # Horizontal Check - Improved for Push-Ups and Planks
        if current_guess is None:
            # Check if TORSO is horizontal (Shoulder to Hip) - Critical for distinguishing from Lunges
            def is_torso_horizontal(lm):
                body_y_diff = abs(lm[11].y - lm[23].y)
                body_x_diff = abs(lm[11].x - lm[23].x)
                return body_x_diff > body_y_diff

            torso_horizontal_frames = [is_torso_horizontal(lm) for lm in self.buffer]
            torso_horizontal_ratio = sum(torso_horizontal_frames) / len(torso_horizontal_frames)

            # Reuse horizontal_ratio already calculated above
            # Require BOTH body and torso to be horizontal to confirm Push-Up/Plank/Bench Press
            # This prevents Lunges (vertical torso) from being misidentified
            # REMOVED hands_below_hips_ratio check - it was blocking push-ups (where hands are on ground/below hips)
            # Deadlifts are already excluded because they don't meet the horizontal_ratio > 0.5 check
            if horizontal_ratio > 0.5 and torso_horizontal_ratio > 0.5:
                # Multiple indicators for push-up vs plank vs bench press
                # 1. Elbow ROM (reduced threshold from 0.1 to 0.08)
                # 2. Shoulder vertical movement
                # 3. Check if hands are near shoulders (push-up position)
                # 4. Check if wrists are ABOVE shoulders (bench press - supine position)
                
                shoulder_rom_std = get_std_y([11, 12])
                
                # Check hand-shoulder distance (push-ups have hands near shoulders)
                hand_shoulder_dists = []
                wrists_above_shoulders = []
                
                for lm in self.buffer:
                    left_dist = abs(lm[15].y - lm[11].y)  # Left wrist to left shoulder
                    right_dist = abs(lm[16].y - lm[12].y)  # Right wrist to right shoulder
                    hand_shoulder_dists.append((left_dist + right_dist) / 2)
                    
                    # Check if wrists are above shoulders (y-coordinate smaller = higher on screen)
                    # Bench press: wrists above shoulders (supine)
                    # Push-up: wrists at/below shoulders (prone)
                    wrist_above = (lm[15].y < lm[11].y and lm[16].y < lm[12].y)
                    wrists_above_shoulders.append(wrist_above)
                
                avg_hand_shoulder_dist = np.mean(hand_shoulder_dists)
                wrists_above_ratio = sum(wrists_above_shoulders) / len(wrists_above_shoulders)
                
                # Bench Press indicators:
                # - Wrists consistently ABOVE shoulders (relaxed to >40%) = supine position
                # - Elbow movement (pressing motion)
                # - Hands relatively close to shoulders
                if wrists_above_ratio > 0.40 and (elbow_rom > 0.08 or shoulder_rom > 0.08):
                    current_guess = "Bench Press"
                        
                # Push-up indicators:
                # - Elbow movement OR shoulder movement (relaxed thresholds)
                # - Hands relatively close to shoulders (< 0.50 normalized distance - relaxed from 0.45)
                # - Wrists NOT above shoulders (prone position)
                elif (elbow_rom > 0.06 or shoulder_rom > 0.06 or shoulder_rom_std > 0.015) and avg_hand_shoulder_dist < 0.50:
                    current_guess = "Push-Up"
                else:
                    current_guess = "Plank"

        # Vertical Check
        if current_guess is None:
            shoulder_y_avg = np.mean([np.mean([lm[11].y, lm[12].y]) for lm in self.buffer])
            hip_y_avg = np.mean([np.mean([lm[23].y, lm[24].y]) for lm in self.buffer])
            ankle_y_avg = np.mean([np.mean([lm[27].y, lm[28].y]) for lm in self.buffer])
            
            # Check for Seated Bench Press (Machine) (Vertical Body + Hands at Chest Level)
            # 1. Body is vertical (Shoulders above Hips)
            # Relaxed threshold to 0.15 to account for partial visibility or tighter framing
            is_vertical = hip_y_avg > shoulder_y_avg + 0.15
            
            # 2. Hands are in CHEST ZONE (between slightly above shoulders and hips)
            # NOT above head (Shoulder Press) and NOT below hips (Deadlift)
            # Relaxed from strict 'close to shoulder' check
            wrist_y_avg = np.mean([np.mean([lm[15].y, lm[16].y]) for lm in self.buffer])
            # Hands can be slightly above shoulder (top of press) but mostly below
            # Hands must be above hips
            hands_in_chest_zone = (wrist_y_avg > shoulder_y_avg - 0.1) and (wrist_y_avg < hip_y_avg)
            
            # 3. Pushing movement (Elbows extend)
            elbow_rom = max(elbow_y) - min(elbow_y)
            
            if is_vertical and hands_in_chest_zone and elbow_rom > 0.08:
                current_guess = "Seated Bench Press"
            
            elif abs(shoulder_y_avg - ankle_y_avg) > 0.2:
                # Improved Lunge Detection: Check for asymmetric leg movement
                knee_y_diffs = [ abs(lm[25].y - lm[26].y) for lm in self.buffer ]
                knee_diff_rom = max(knee_y_diffs) - min(knee_y_diffs)
                avg_knee_diff = np.mean(knee_y_diffs)
                
                ankle_dists = [ abs(lm[27].x - lm[28].x) for lm in self.buffer ]
                avg_ankle_dist = np.mean(ankle_dists)
                
                # Lunge indicators:
                # 1. Significant knee height difference (one leg forward)
                # 2. Wide ankle distance (split stance)
                # 3. Knee height difference changes during movement
                is_lunge = (avg_knee_diff > 0.12 and avg_ankle_dist > 0.22) or \
                           (knee_diff_rom > 0.08 and avg_ankle_dist > 0.20)
                
                if is_lunge:
                    current_guess = "Lunge"
                else:
                    # Check for Deadlift vs Squat
                    torso_heights = [ abs(lm[11].y - lm[23].y) for lm in self.buffer ]
                    hands_low_frames = [ (lm[15].y > lm[23].y and lm[16].y > lm[24].y) for lm in self.buffer ]
                    # Deadlift: Torso height changes significantly while hands stay low
                    torso_rom = max(torso_heights) - min(torso_heights)
                    
                    # Improved deadlift detection: hands should stay consistently low
                    if torso_rom > 0.12 and sum(hands_low_frames) > self.window_size * 0.65:
                        current_guess = "Deadlift"
                    elif hip_rom > 0.12:  # Reduced threshold for squat detection
                        current_guess = "Squat"

        # 4. Voting & Confidence - More responsive
        if current_guess:
            self.confidence_votes.append(current_guess)
            self.last_guess = current_guess  # Track for debugging
            
        # Allow detection with partial buffer for faster response
        if len(self.confidence_votes) >= int(self.confidence_votes.maxlen * 0.8):  # 80% of voting window
            counts = Counter(self.confidence_votes)
            most_common, count = counts.most_common(1)[0]
            
            # Reduced confidence threshold from 90% to 75% for faster detection
            confidence_ratio = count / len(self.confidence_votes)
            if confidence_ratio > 0.75:
                # Final check: Ensure we saw an inflection (the bottom of the rep)
                # Skip this check for static exercises (Plank) and continuous exercises (Jumping Jacks)
                if most_common in ["Squat", "Push-Up", "Bench Press", "Lunge", "Deadlift"]:
                    # Only require inflection if we have enough data
                    if len(self.buffer) >= self.window_size * 0.9:
                        # Use shoulder_y for push-ups, elbow_y for bench press (main vertical movement)
                        if most_common == "Push-Up":
                            check_data = shoulder_y
                        elif most_common == "Bench Press":
                            check_data = elbow_y
                        else:
                            check_data = hip_y
                        if not has_inflection(check_data):
                            return None
                return most_common
                
        return None
