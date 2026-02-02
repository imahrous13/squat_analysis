import cv2
import numpy as np

class RoiManager:
    def __init__(self, padding_pct=0.2, smooth_factor=0.3):
        self.padding_pct = padding_pct
        self.smooth_factor = smooth_factor
        self.last_roi = None 
        
    def get_roi(self, box, frame_w, frame_h):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        pad_w = int(w * self.padding_pct)
        pad_h = int(h * self.padding_pct)
        
        target_x1 = max(0, int(x1 - pad_w))
        target_y1 = max(0, int(y1 - pad_h))
        target_x2 = min(frame_w, int(x2 + pad_w))
        target_y2 = min(frame_h, int(y2 + pad_h))
        
        # Apply EMA Smoothing to ROI coordinates to prevent jitter
        if self.last_roi is not None:
            nx1 = int(self.last_roi[0] * (1 - self.smooth_factor) + target_x1 * self.smooth_factor)
            ny1 = int(self.last_roi[1] * (1 - self.smooth_factor) + target_y1 * self.smooth_factor)
            nx2 = int(self.last_roi[2] * (1 - self.smooth_factor) + target_x2 * self.smooth_factor)
            ny2 = int(self.last_roi[3] * (1 - self.smooth_factor) + target_y2 * self.smooth_factor)
        else:
            nx1, ny1, nx2, ny2 = target_x1, target_y1, target_x2, target_y2

        self.last_roi = (nx1, ny1, nx2, ny2)
        return (nx1, ny1, nx2, ny2)

    def crop_frame(self, frame, roi):
        x1, y1, x2, y2 = roi
        return frame[y1:y2, x1:x2]

    def map_landmark_to_full_frame(self, landmark, frame_w, frame_h):
        if self.last_roi is None: return landmark
        x1, y1, x2, y2 = self.last_roi
        roi_w = x2 - x1
        roi_h = y2 - y1
        roi_px_x = landmark.x * roi_w
        roi_px_y = landmark.y * roi_h
        full_px_x = x1 + roi_px_x
        full_px_y = y1 + roi_px_y
        
        new_lm = type(landmark)() 
        new_lm.x = full_px_x / frame_w if frame_w > 0 else 0
        new_lm.y = full_px_y / frame_h if frame_h > 0 else 0
        new_lm.z = landmark.z
        new_lm.visibility = landmark.visibility
        return new_lm

class HybridPoseEstimator:
    """
    Hybrid Pipeline: YOLOv8 Detection -> ROI Crop -> MediaPipe Pose -> Full Frame Mapping
    Drop-in replacement for PoseDetector.
    """
    def __init__(self, model_path='yolov8n.pt', min_detection_conf=0.5, min_pose_conf=0.5):
        from ultralytics import YOLO
        from mediapipe.solutions import pose as mp_pose
        from mediapipe.solutions import drawing_utils as mp_drawing
        
        self.yolo = YOLO(model_path)
        self.roi_manager = RoiManager(padding_pct=0.25)
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # Use Balanced model for significantly better stability
            smooth_landmarks=True, # Enable smoothing to reduce jitter
            min_detection_confidence=min_pose_conf,
            min_tracking_confidence=min_pose_conf
        )
        self.min_det_conf = min_detection_conf
        self.last_landmarks = None 
        
        # Sticky Tracking State
        self.locked_track_id = None
        self.missed_track_frames = 0
        self.max_missed_frames = 10
        
        # Optimization: Frame Skipping
        self.frame_counter = 0
        self.yolo_interval = 15 # Run YOLO every 15 frames for speed

    def find_pose(self, frame, draw=True):
        h, w, c = frame.shape
        self.last_landmarks = None
        self.frame_counter += 1
        
        # Determine if we should run YOLO
        # Increase YOLO frequency to every 3 frames for maximum responsiveness
        run_yolo = (self.frame_counter % 3 == 0) or (self.locked_track_id is None)
        
        best_box = None
        
        if run_yolo:
            # 1. YOLO Track with Persistence
            results = self.yolo.track(frame, persist=True, verbose=False, classes=[0], conf=self.min_det_conf)
            
            best_box = None
            current_id_found = False
            
            if results and results[0].boxes:
                all_boxes = results[0].boxes
                
                # Candidate Logic
                candidates = []
                
                for box in all_boxes:
                    if box.id is None: continue 
                    
                    tid = int(box.id[0].item())
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    
                    # Selection Stats
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Store candidate info
                    # Score is now PURELY Area-based to prioritize the "Bigger Person" (closest to camera)
                    # We normalize area by frame size for consistent scoring
                    score = (area / (w * h)) * 100
                    
                    candidates.append({
                        "box": coords,
                        "id": tid,
                        "area": area,
                        "score": score
                    })

                # If we have a locked ID, find it among candidates
                if self.locked_track_id is not None:
                    locked_candidate = next((c for c in candidates if c["id"] == self.locked_track_id), None)
                    
                    if locked_candidate:
                        # Check if there is now a SIGNIFICANTLY bigger person (> 30% larger area)
                        # This allows the tracker to switch if the main subject changes or if it locked onto a background person initially
                        best_overall = max(candidates, key=lambda x: x["area"])
                        if best_overall["area"] > locked_candidate["area"] * 1.3:
                            self.locked_track_id = best_overall["id"]
                            best_box = best_overall["box"]
                        else:
                            best_box = locked_candidate["box"]
                        
                        current_id_found = True
                        self.missed_track_frames = 0
                
                # If no current lock or lost lock, pick the LARGEST person
                if not current_id_found:
                    if self.locked_track_id is not None:
                        self.missed_track_frames += 1
                        if self.missed_track_frames > self.max_missed_frames:
                            self.locked_track_id = None
                            
                    if self.locked_track_id is None and candidates:
                        # Pick the absolute biggest person (highest area)
                        candidates.sort(key=lambda x: x["area"], reverse=True)
                        best_candidate = candidates[0]
                        self.locked_track_id = best_candidate["id"]
                        best_box = best_candidate["box"]
                        self.missed_track_frames = 0
        else:
             # Skip YOLO, try to use last known ROI (if exists)
             if self.roi_manager.last_roi:
                  # We don't have a specific box, but we have an ROI.
                  # Logic below expects 'best_box' to calculate ROI.
                  # Let's just use the logic below but bypass box calculation if we have cached ROI
                  pass 
        
        mp_results_obj = None
        
        # Decide ROI Source: New Box OR Cached ROI
        roi_coords = None
        
        if best_box is not None:
             roi_coords = self.roi_manager.get_roi(best_box, w, h)
        elif not run_yolo and self.roi_manager.last_roi:
             roi_coords = self.roi_manager.last_roi
             
        if roi_coords is not None:
            # 2. ROI
            # roi_coords already set
            roi_crop = self.roi_manager.crop_frame(frame, roi_coords)
            roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
            
            # 3. Pose
            mp_results = self.pose.process(roi_rgb)
            
            if mp_results.pose_landmarks:
                full_frame_landmarks = []
                for lm in mp_results.pose_landmarks.landmark:
                    mapped_lm = self.roi_manager.map_landmark_to_full_frame(lm, w, h)
                    full_frame_landmarks.append(mapped_lm)
                
                self.last_landmarks = full_frame_landmarks
                
                # Mock Results Object
                class ResultsWrapper:
                    def __init__(self, lms):
                        self.pose_landmarks = type('obj', (object,), {'landmark': lms})
                
                mp_results_obj = ResultsWrapper(full_frame_landmarks)
                
                if draw:
                    self.mp_drawing.draw_landmarks(
                        frame, mp_results_obj.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                    )
                    rx1, ry1, rx2, ry2 = roi_coords
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                    cv2.putText(frame, "ROI", (rx1, ry1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        else:
            # Fallback
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = self.pose.process(img_rgb)
            if mp_results.pose_landmarks:
                self.last_landmarks = mp_results.pose_landmarks.landmark
                mp_results_obj = mp_results
                if draw:
                    self.mp_drawing.draw_landmarks(
                        frame, mp_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                    )
        
        return frame, mp_results_obj

    def get_landmarks(self):
        return self.last_landmarks
