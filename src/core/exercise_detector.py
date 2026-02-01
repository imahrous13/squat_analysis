import numpy as np
from collections import deque, Counter

class ExerciseDetector:
    def __init__(self, window_size=60): # Reduced from 90 to 60 for even faster detection (~2 seconds)
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.confidence_votes = deque(maxlen=30)  # Reduced from 45 to 30 for faster voting
        self.exercises = ["Squat", "Deadlift", "Push-Up", "Bench Press", "Seated Bench Press", "Lunge", "Jumping Jacks", "Plank", "Chest Fly", "Seated Chest Fly", "Dips", "Seated Dips"]
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
        if len(self.buffer) < int(self.window_size * 0.5):  # 50% of window size (30 frames)
            return None
        
        # 1. Motion Analysis - More lenient thresholds
        hip_y = [np.mean([lm[23].y, lm[24].y]) for lm in self.buffer]
        elbow_y = [np.mean([lm[13].y, lm[14].y]) for lm in self.buffer]
        shoulder_y = [np.mean([lm[11].y, lm[12].y]) for lm in self.buffer]
        
        hip_x = [np.mean([lm[23].x, lm[24].x]) for lm in self.buffer]
        elbow_x = [np.mean([lm[13].x, lm[14].x]) for lm in self.buffer]
        shoulder_x = [np.mean([lm[11].x, lm[12].x]) for lm in self.buffer]
        
        hip_rom_y = max(hip_y) - min(hip_y)
        elbow_rom_y = max(elbow_y) - min(elbow_y)
        shoulder_rom_y = max(shoulder_y) - min(shoulder_y)

        hip_rom_x = max(hip_x) - min(hip_x)
        elbow_rom_x = max(elbow_x) - min(elbow_x)
        shoulder_rom_x = max(shoulder_x) - min(shoulder_x)
        
        # Combined ROM for more robust movement detection
        hip_rom = max(hip_rom_y, hip_rom_x * 0.7) # X-movement weighted slightly less for trunk
        elbow_rom = max(elbow_rom_y, elbow_rom_x)
        shoulder_rom = max(shoulder_rom_y, shoulder_rom_x)
        combined_elbow_rom = elbow_rom # Consistent alias for exercise heuristics
        
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
        
        # PRIORITY CHECK: Deadlift High ROM
        # Used to distinguish from squat
        hip_rom_val = max(hip_y) - min(hip_y) # Pure vertical ROM for hips

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
            hip_rom_val > 0.12 and 
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
                # - Wrists consistently ABOVE shoulders (relaxed to >30%) = supine position
                # - Even small movement should trigger Bench Press over Plank if wrists are above shoulders
                # CRITICAL FIX: Ensure body is "flat" vertically (shoulder and ankle Y are close)
                # Squats are standing (large Y diff), Bench Press is supine (small Y diff)
                body_height_y = abs(lm[11].y - lm[27].y)
                is_vertically_flat = body_height_y < 0.30
                
                # KINEMATIC CHECK: Bench Press vs Push-Up
                # Bench Press: Hands Move (High ROM), Shoulders Fixed (Low ROM)
                # Push-Up: Shoulders Move (High ROM), Hands Fixed (Low ROM)
                all_wrist_y = [np.mean([lm[15].y, lm[16].y]) for lm in self.buffer]
                wrist_rom_y = max(all_wrist_y) - min(all_wrist_y)
                
                # Ratio > 1.5 means Hands move MORE than Shoulders => Bench Press
                # Ratio < 0.7 means Shoulders move MORE than Hands => Push-Up
                hand_vs_shoulder_movement = wrist_rom_y / (shoulder_rom_y + 0.001)
                is_bench_movement = hand_vs_shoulder_movement > 1.5  # Increased from 1.2
                is_pushup_movement = hand_vs_shoulder_movement < 0.7  # Decreased from 0.8

                # Additional check: In bench press, elbows typically move outward (X-axis)
                # In push-ups, body moves up/down (Y-axis dominant)
                elbow_x_positions = [np.mean([lm[13].x, lm[14].x]) for lm in self.buffer]
                elbow_rom_x = max(elbow_x_positions) - min(elbow_x_positions)
                
                # Bench press has more horizontal elbow movement
                is_bench_elbow_pattern = elbow_rom_x > 0.08

                # Combined Check: Position (Supine) OR Movement (Hands Moving)
                # STRENGTHENED: Require BOTH position AND movement for bench press
                if ((wrists_above_ratio > 0.40 and is_bench_movement) or 
                    (wrists_above_ratio > 0.50 and is_bench_elbow_pattern)) and is_vertically_flat:
                    # If there's some movement, it's definitely Bench Press
                    # INCREASED THRESHOLDS to prevent static "waiting" from counting
                    if (elbow_rom_y > 0.08 or shoulder_rom_y > 0.08 or shoulder_rom_std > 0.03 or wrist_rom_y > 0.05):
                        # DIFFERENTIATE Bench Press vs Chest Fly (Lying)
                        # Bench Press: Elbows Extend (90->180). High Angle ROM.
                        # Chest Fly: Elbows Fixed Bend. Low Angle ROM.
                        def get_angle(a, b, c):
                            ang = np.degrees(np.arctan2(c.y-b.y, c.x-b.x) - np.arctan2(a.y-b.y, a.x-b.x))
                            return abs(ang) if abs(ang) < 180 else 360 - abs(ang)

                        l_angles = [get_angle(lm[11], lm[13], lm[15]) for lm in self.buffer]
                        r_angles = [get_angle(lm[12], lm[14], lm[16]) for lm in self.buffer]
                        avg_elbow_rom = ((max(l_angles)-min(l_angles)) + (max(r_angles)-min(r_angles))) / 2
                        
                        if avg_elbow_rom < 45:
                            current_guess = "Chest Fly"
                        else:
                            current_guess = "Bench Press"
                        
                # Push-up indicators:
                # - Wrists NOT above shoulders (prone position)
                # - Elbow movement OR shoulder movement
                # - Shoulders moving MORE than hands (is_pushup_movement)
                # STRENGTHENED: Require clear push-up movement pattern
                elif ((is_pushup_movement and (elbow_rom_y > 0.06 or shoulder_rom_y > 0.06)) or 
                      (wrists_above_ratio < 0.20 and shoulder_rom_y > 0.08 and avg_hand_shoulder_dist < 0.50)):
                        current_guess = "Push-Up"
                else:
                    # Fallback for horizontal position with low movement and prone-style hands
                    # Also enforce flatness for Plank to avoid weird false positives
                    if is_vertically_flat:
                        current_guess = "Plank"

        # Vertical Check
        if current_guess is None:
            shoulder_y_avg = np.mean([np.mean([lm[11].y, lm[12].y]) for lm in self.buffer])
            hip_y_avg = np.mean([np.mean([lm[23].y, lm[24].y]) for lm in self.buffer])
            ankle_y_avg = np.mean([np.mean([lm[27].y, lm[28].y]) for lm in self.buffer])
            
            # Check for Seated Bench Press (Machine) (Vertical Body + Hands at Chest Level)
            is_vertical_torso = abs(shoulder_y_avg - hip_y_avg) > 0.08
            
            # Hands in CHEST/MID ZONE (Relaxed lower bound for Dips/Press)
            wrist_y_avg = np.mean([np.mean([lm[15].y, lm[16].y]) for lm in self.buffer])
            hands_in_chest_zone = (wrist_y_avg > shoulder_y_avg - 0.15) and (wrist_y_avg < hip_y_avg + 0.25)
            
            # Check if person is STANDING (Large distance between shoulder and ankle)
            # If Standing, we should NOT detect "Seated" exercises
            standing_height = abs(shoulder_y_avg - ankle_y_avg)
            # Threshold adjusted to robustly identify standing
            is_standing_pose = standing_height > 0.20
            
            # 1. CHECK LEG/FULL BODY EXERCISES (Lunge, Squat, Deadlift)
            # Prioritize this because Squats often have hands in chest zone (which was blocking them)
            # Use standing_height or leg checks
            if standing_height > 0.15:
                # Improved Lunge Detection: Check for asymmetric leg movement
                knee_y_diffs = [ abs(lm[25].y - lm[26].y) for lm in self.buffer ]
                knee_diff_rom = max(knee_y_diffs) - min(knee_y_diffs)
                avg_knee_diff = np.mean(knee_y_diffs)
                
                ankle_dists = [ abs(lm[27].x - lm[28].x) for lm in self.buffer ]
                avg_ankle_dist = np.mean(ankle_dists)
                
                # Lunge indicators:
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
                    if torso_rom > 0.10 and sum(hands_low_frames) > self.window_size * 0.65:
                        current_guess = "Deadlift"
                    elif hip_rom > 0.08:  # Further reduced threshold for squat detection
                        # Differentiate Squat vs Dips
                        # Logic:
                        # 1. Hands: Fixed (Dips/Assisted Squat) vs Moving (Standard Squat)
                        # 2. Legs: Rigid (Dips) vs Compressing (Squats)
                        
                        # 1. Elbow Angle Check: DIFFERENTIATOR
                        # Dips: High Elbow ROM (Extension 90->180). Squats: Static Arms (Holding Bar).
                        def get_angle(a, b, c):
                            ang = np.degrees(np.arctan2(c.y-b.y, c.x-b.x) - np.arctan2(a.y-b.y, a.x-b.x))
                            return abs(ang) if abs(ang) < 180 else 360 - abs(ang)

                        l_angles = [get_angle(lm[11], lm[13], lm[15]) for lm in self.buffer]
                        r_angles = [get_angle(lm[12], lm[14], lm[16]) for lm in self.buffer]
                        avg_elbow_rom = ((max(l_angles)-min(l_angles)) + (max(r_angles)-min(r_angles))) / 2
                        
                        # Typical Dips: > 60 deg change. Squats: < 20 deg change.
                        # Also keep hand check as secondary guard? No, angle determines mechanism.
                        if avg_elbow_rom > 35:
                            current_guess = "Dips"
                        else:
                            current_guess = "Squat"

            # 2. CHECK SEATED / UPPER BODY EXERCISES (Fly, Press)
            # Only if no leg exercise detected AND NOT CLEARLY STANDING
            # Additional Guards:
            # - Hip ROM must be low (Seated user shouldn't mimic squat motion)
            # - Ankle distance should be normal (Not split stance like lunge)
            ankle_dist_x = np.mean([abs(lm[27].x - lm[28].x) for lm in self.buffer])
            
            if current_guess is None and is_vertical_torso and hands_in_chest_zone:
                # Strict exclusions for Seated exercises
                is_excluded_by_hip_movement = hip_rom_y > 0.10 # Relaxed from 0.06 to allow slight shift
                is_excluded_by_stance = ankle_dist_x > 0.18 # Wide/Split stance = Lunge/Squat
                
                # Removed is_excluded_by_standing check as it blocks legitimate Seated exercises (Dips/Press)
                if not (is_excluded_by_hip_movement or is_excluded_by_stance):
                    # 1. CHEST FLY (Hand-to-Hand distance changes significantly)
                    # Switched to ROM check (Max - Min) for robustness
                    # Flys have large width change (>0.15). Press has fixed width (<0.10).
                    arm_spread_dists = [abs(lm[15].x - lm[16].x) for lm in self.buffer]
                    arm_spread_rom = max(arm_spread_dists) - min(arm_spread_dists)
                    
                    # Elbow Angle ROM Check: DIFFERENTIATOR
                    # Press: Elbows Extend (Angle changes 90->180, ROM > 60).
                    # Fly: Elbows Fixed Bend (Angle constant, ROM < 30).
                    def get_angle(a, b, c):
                        ang = np.degrees(np.arctan2(c.y-b.y, c.x-b.x) - np.arctan2(a.y-b.y, a.x-b.x))
                        return abs(ang) if abs(ang) < 180 else 360 - abs(ang)

                    l_angles = [get_angle(lm[11], lm[13], lm[15]) for lm in self.buffer]
                    r_angles = [get_angle(lm[12], lm[14], lm[16]) for lm in self.buffer]
                    avg_elbow_rom = ((max(l_angles)-min(l_angles)) + (max(r_angles)-min(r_angles))) / 2

                    if arm_spread_rom > 0.15 and avg_elbow_rom < 45:
                        # Seated (Butterfly Machine) has almost ZERO hip vertical movement
                        if hip_rom_y < 0.04 and horizontal_ratio < 0.2:
                            current_guess = "Seated Chest Fly"
                        else:
                            current_guess = "Chest Fly"
                    
                    else:
                        # Differentiate Seated Press vs Seated Dips using HAND POSITION + MOVEMENT
                        # Seated Dips: Hands are generally LOWER (near hips) and move VERTICALLY.
                        # Seated Bench Press: Hands are HIGHER (chest/shoulder level) and move HORIZONTALLY.
                        
                        wrist_y = [np.mean([lm[15].y, lm[16].y]) for lm in self.buffer]
                        wrist_x = [np.mean([lm[15].x, lm[16].x]) for lm in self.buffer]
                        wrist_rom_y = max(wrist_y) - min(wrist_y)
                        wrist_rom_x = max(wrist_x) - min(wrist_x)
                        
                        # Calculate Elbow ROM explicitly to avoid UnboundLocalError
                        l_elbow_x = [lm[13].x for lm in self.buffer]
                        r_elbow_x = [lm[14].x for lm in self.buffer]
                        combined_elbow_rom = (max(l_elbow_x) - min(l_elbow_x) + max(r_elbow_x) - min(r_elbow_x)) / 2

                        avg_wrist_y = np.mean(wrist_y)
                        avg_shoulder_y = np.mean([np.mean([lm[11].y, lm[12].y]) for lm in self.buffer])
                        
                        # Check if hands are typically lower than chest level
                        # Y increases downwards. So Wrist > Shoulder means Wrist is Lower.
                        hands_are_low = avg_wrist_y > (avg_shoulder_y + 0.10)
                        
                        # 2. SEATED DIPS (Tricep Press)
                        # High Elbow ROM + (Hands Low OR Dominant Vertical Movement)
                        # Relaxed X-ROM check because oblique angles cause X movement projection
                        if combined_elbow_rom > 0.06 and (hands_are_low or (wrist_rom_y > 0.08 and wrist_rom_x < 0.10)):
                            current_guess = "Seated Dips"
                            
                        # 3. SEATED BENCH PRESS
                        # Fallback if hands are high (Chest level)
                        elif combined_elbow_rom > 0.06:
                             current_guess = "Seated Bench Press"
            
            # 3. DIPS
            if current_guess is None and is_vertical_torso and shoulder_rom_y > 0.08 and hands_below_hips_ratio > 0.6:
                current_guess = "Dips"
            
            # 4. Voting & Confidence - Ultra responsive
            if current_guess:
                self.confidence_votes.append(current_guess)
                self.last_guess = current_guess
            
            # Allow detection with much smaller vote pool for faster response
            if len(self.confidence_votes) >= int(self.confidence_votes.maxlen * 0.6):  # 60% of voting window (18 votes)
                counts = Counter(self.confidence_votes)
                most_common, count = counts.most_common(1)[0]
                
                # Use 70% confidence for faster initial lock
                confidence_ratio = count / len(self.confidence_votes)
                if confidence_ratio > 0.70:
                    return most_common
                    
            return None

        # 4. Voting & Confidence - Ultra responsive
        if current_guess:
            self.confidence_votes.append(current_guess)
            self.last_guess = current_guess
            
        # Allow detection with much smaller vote pool for faster response
        if len(self.confidence_votes) >= int(self.confidence_votes.maxlen * 0.6):  # 60% of voting window (18 votes)
            counts = Counter(self.confidence_votes)
            most_common, count = counts.most_common(1)[0]
            
            # Use 70% confidence for faster initial lock
            confidence_ratio = count / len(self.confidence_votes)
            if confidence_ratio > 0.70:
                # Removed complex inflection check for initial detection to speed up start
                return most_common
                
        return None
