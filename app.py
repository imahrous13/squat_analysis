import os
print("🚀 App starting... Setting environment variables.")
# CRITICAL: Set environment variables BEFORE any other imports to prevent crashes on Cloud
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MP_GPU_MODE'] = '0' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import streamlit as st
import cv2
import tempfile
import time
import subprocess
import numpy as np
import gc

# Optional imports for webcam feature (may not work on Cloud)
WEBRTC_AVAILABLE = False
WEBRTC_ERROR = ""
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError as e:
    WEBRTC_ERROR = str(e)
    # Create dummy classes to prevent errors
    class VideoProcessorBase:
        pass
    class RTCConfiguration:
        def __init__(self, *args, **kwargs):
            pass

# Lazy loading functions to prevent Cloud memory crashes
def get_pose_detector(use_hybrid=False):
    if use_hybrid:
        from src.core.hybrid_pose import HybridPoseEstimator
        return HybridPoseEstimator(model_path='yolov8n.pt')
    else:
        from src.core.movenet_pose import MoveNetEstimator
        return MoveNetEstimator()

def get_exercise_detector():
    from src.core.exercise_detector import ExerciseDetector
    return ExerciseDetector()

def get_analyzer_class(exercise_type):
    """Lazy load analyzer classes only when needed"""
    if exercise_type == "Squat":
        from src.analyzers.squat_analyzer import SquatAnalyzer
        return SquatAnalyzer
    elif exercise_type == "Push-Up":
        from src.analyzers.pushup_analyzer import PushUpAnalyzer
        return PushUpAnalyzer
    elif exercise_type == "Bench Press":
        from src.analyzers.bench_press_analyzer import BenchPressAnalyzer
        return BenchPressAnalyzer
    elif exercise_type == "Seated Bench Press":
        from src.analyzers.bench_press_analyzer import BenchPressAnalyzer
        return BenchPressAnalyzer
    elif exercise_type == "Deadlift":
        from src.analyzers.deadlift_analyzer import DeadliftAnalyzer
        return DeadliftAnalyzer
    elif exercise_type == "Lunge":
        from src.analyzers.lunge_analyzer import LungeAnalyzer
        return LungeAnalyzer
    elif exercise_type == "Jumping Jacks":
        from src.analyzers.jumping_jacks_analyzer import JumpingJacksAnalyzer
        return JumpingJacksAnalyzer
    elif exercise_type == "Plank":
        from src.analyzers.plank_analyzer import PlankAnalyzer
        return PlankAnalyzer
    elif exercise_type == "Chest Fly":
        from src.analyzers.chest_fly_analyzer import ChestFlyAnalyzer
        return ChestFlyAnalyzer
    elif exercise_type == "Seated Chest Fly":
        from src.analyzers.chest_fly_analyzer import ChestFlyAnalyzer
        return ChestFlyAnalyzer
    elif exercise_type == "Dips":
        from src.analyzers.dips_analyzer import DipsAnalyzer
        return DipsAnalyzer
    elif exercise_type == "Seated Dips":
        from src.analyzers.dips_analyzer import DipsAnalyzer
        return DipsAnalyzer
    return None

def get_utils():
    from src.core.utils import draw_text_with_background, get_landmark_pixel
    return draw_text_with_background, get_landmark_pixel

def get_mp_pose():
    """Lazy load MediaPipe pose module"""
    try:
        # Standard import
        import mediapipe as mp
        return mp.solutions.pose
    except (ImportError, AttributeError):
        try:
            # Direct submodule import
            from mediapipe.solutions import pose as mp_pose
            return mp_pose
        except (ImportError, AttributeError):
            # Surgical internal import
            import mediapipe.python.solutions.pose as mp_pose
            return mp_pose

# RTC Configuration placeholder (set in main)
RTC_CONFIGURATION = None

def reencode_video_for_browser(input_path, output_path=None):
    """Re-encode using PyAV for maximum compatibility on both Windows (no FFmpeg) and Cloud."""
    if not os.path.exists(input_path): return input_path, "Not found"
    if output_path is None: output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    try:
        import av
        input_container = av.open(input_path)
        output_container = av.open(output_path, mode='w', format='mp4')
        in_stream = input_container.streams.video[0]
        out_stream = output_container.add_stream('libx264', rate=in_stream.base_rate)
        out_stream.width, out_stream.height = in_stream.width, in_stream.height
        out_stream.pix_fmt = 'yuv420p'
        out_stream.options = {'preset': 'veryfast', 'crf': '23'}
        for frame in input_container.decode(video=0):
            for packet in out_stream.encode(frame): output_container.mux(packet)
        for packet in out_stream.encode(): output_container.mux(packet)
        input_container.close(); output_container.close()
        return output_path, None
    except Exception as e: return input_path, str(e)


class BaseVideoProcessor(VideoProcessorBase):
    def __init__(self, analyzer_class, recording_path=None, use_hybrid=False, **analyzer_args):
        self.detector = get_pose_detector(use_hybrid)
            
        self.analyzer = analyzer_class(**analyzer_args) if analyzer_class else None
        self.exercise_name = analyzer_class.__name__.replace('Analyzer', '') if analyzer_class else "Detecting..."
        self.exercise_detector = get_exercise_detector()
        self.p_time = 0
        self.recording_path = recording_path
        self.out = None

    def stop_recording(self):
        if self.out:
            self.out.release()
            self.out = None

    def __del__(self):
        self.stop_recording()

    def recv(self, frame):
        draw_text_with_background, get_landmark_pixel = get_utils()
        img = frame.to_ndarray(format="bgr24")
        h, w, c = img.shape
        
        # Pose Detection
        img, results = self.detector.find_pose(img, draw=True)
        landmarks = self.detector.get_landmarks()
        
        analysis_data = {}
        if landmarks:
            # Auto-Detect Exercise if not set
            if self.analyzer is None:
                self.exercise_detector.add_frame(landmarks)
                detected = self.exercise_detector.detect()
                if detected:
                    self.exercise_name = detected
                    analyzer_class = get_analyzer_class(detected)
                    
                    if detected in ["Seated Bench Press", "Seated Chest Fly", "Seated Dips"]:
                        self.analyzer = analyzer_class(variant="seated")
                    elif analyzer_class:
                        self.analyzer = analyzer_class()
                    
                    if self.analyzer and detected == "Plank": 
                        self.analyzer.is_active = True
                        self.analyzer.feedback = "Exercise Detected! Hold your plank."
                    elif self.analyzer:
                        self.analyzer.feedback = "Exercise Detected! Start your reps."
            
            if self.analyzer:
                try:
                    analysis_data = self.analyzer.analyze(landmarks, w, h)
                except Exception as e:
                    print(f"Analyzer Error: {e}")
                    analysis_data = {"state": "ERROR", "feedback": f"Error: {str(e)}", "rep_count": 0}
            else:
                # Show more detailed feedback during auto-detection
                buffer_size = len(self.exercise_detector.buffer)
                required_size = int(self.exercise_detector.window_size * 0.7)
                votes_size = len(self.exercise_detector.confidence_votes)
                required_votes = int(self.exercise_detector.confidence_votes.maxlen * 0.8)
                
                feedback_msg = "Keep moving! Analyzing exercise..."
                if buffer_size < required_size:
                    progress = int((buffer_size / required_size) * 100)
                    feedback_msg = f"Collecting data... {progress}%"
                elif votes_size < required_votes:
                    progress = int((votes_size / required_votes) * 100)
                    feedback_msg = f"Analyzing pattern... {progress}%"
                    if self.exercise_detector.last_guess:
                        feedback_msg += f" (Detecting: {self.exercise_detector.last_guess})"
                
                analysis_data = {
                    "state": "ANALYZING...", 
                    "rep_count": 0, 
                    "feedback": feedback_msg, 
                    "target_muscles": "Detecting...",
                    "correct_reps": 0,
                    "incorrect_reps": 0
                }
            self._draw_specifics(img, landmarks, analysis_data, w, h)
        
        # FPS
        c_time = time.time()
        fps = 1 / (c_time - self.p_time) if self.p_time > 0 else 0
        self.p_time = c_time
        
        # Overlay
        draw_text_with_background(img, f"FPS: {int(fps)}", (10, 30), text_color=(0, 255, 0))
        
        if analysis_data:
            self._draw_common_overlay(img, analysis_data, w, h)
            
        # Record Frame
        if self.recording_path:
            if self.out is None:
                try:
                    # Try H264 codec first (best browser compatibility)
                    fourcc = cv2.VideoWriter_fourcc(*'H264')
                    self.out = cv2.VideoWriter(self.recording_path, fourcc, 20.0, (w, h))
                    if not self.out.isOpened():
                        raise Exception("H264 failed")
                except:
                    try:
                        # Fallback to X264
                        fourcc = cv2.VideoWriter_fourcc(*'X264')
                        self.out = cv2.VideoWriter(self.recording_path, fourcc, 20.0, (w, h))
                        if not self.out.isOpened():
                            raise Exception("X264 failed")
                    except:
                        # Last resort: mp4v
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.out = cv2.VideoWriter(self.recording_path, fourcc, 20.0, (w, h))
            self.out.write(img)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        pass

    def _draw_common_overlay(self, img, analysis_data, w, h):
        draw_text_with_background, _ = get_utils()
        state_color = (0, 255, 255)
        if analysis_data.get('state') in ["BOTTOM", "LOCKOUT", "TOP_PLANK"]: 
            state_color = (0, 255, 0)
            
        y_pos = 40
        draw_text_with_background(img, f"Exercise: {self.exercise_name}", (10, y_pos), text_color=(255, 255, 255))
        
        y_pos += 40
        draw_text_with_background(img, f"State: {analysis_data['state']}", (10, y_pos), text_color=state_color)
        
        y_pos += 40
        draw_text_with_background(img, f"Muscles: {analysis_data.get('target_muscles', 'N/A')}", (10, y_pos), font_scale=0.6, text_color=(255, 150, 0))
        
        y_pos += 40
        c_reps = analysis_data.get('correct_reps', 0)
        i_reps = analysis_data.get('incorrect_reps', 0)
        
        display_rep_count = analysis_data.get('rep_count', 0)
        if self.exercise_name not in ["Plank", "Deadlift", "Lunge"] and analysis_data.get('state') != "ANALYZING...":
            display_rep_count = c_reps + i_reps

        draw_text_with_background(img, f"Reps: {display_rep_count}", (10, y_pos), font_scale=0.8, thickness=2)
        
        y_pos += 35
        
        draw_text_with_background(img, f"Correct: {c_reps}", (10, y_pos), font_scale=0.6, text_color=(0, 255, 0))
        draw_text_with_background(img, f"Incorrect: {i_reps}", (160, y_pos), font_scale=0.6, text_color=(0, 0, 255))
        
        y_pos += 40
        feedback = analysis_data.get('feedback', '')
        if feedback:
            draw_text_with_background(img, f"Feedback: {feedback}", (10, y_pos), text_color=(0, 100, 255))
        
        y_pos += 40
        advice = analysis_data.get('advice', '')
        if advice:
            for line in self._wrap_text(advice, 50)[:3]:
                draw_text_with_background(img, line, (10, y_pos), text_color=(255, 200, 0), font_scale=0.5)
                y_pos += 25
        y_pos += 40
        score = analysis_data.get('last_rep_score', 0)
        draw_text_with_background(img, f"Last Score: {score}", (10, y_pos), 
                                  text_color=(0, 255, 0) if score >= 75 else (0, 0, 255))
        
        if score > 0 and score < 75 and analysis_data.get('reasons'):
            y_pos += 30
            draw_text_with_background(img, "Faults Detected:", (10, y_pos), text_color=(0, 0, 255), font_scale=0.5)
            y_pos += 20
            for reason in analysis_data['reasons'][:3]:
                draw_text_with_background(img, f"- {reason}", (20, y_pos), text_color=(0, 0, 255), font_scale=0.4)
                y_pos += 15

        if analysis_data.get('valgus_detected'):
            y_pos += 30
            draw_text_with_background(img, "KNEE VALGUS!", (10, y_pos), text_color=(0, 0, 255), bg_color=(255, 255, 255))
        
        y_pos += 30
        view_mode = analysis_data.get('view', 'Unknown')
        draw_text_with_background(img, f"View: {view_mode}", (10, y_pos), text_color=(200, 200, 200), font_scale=0.4)

    def _wrap_text(self, text, limit):
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            if len(current_line + word) < limit:
                current_line += word + " "
            else:
                if current_line: lines.append(current_line.strip())
                current_line = word + " "
        if current_line: lines.append(current_line.strip())
        return lines

class SquatVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        super().__init__(get_analyzer_class("Squat"), recording_path, use_hybrid)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        _, get_landmark_pixel = get_utils()
        mp_pose = get_mp_pose()
        for side in [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE]:
            pos = get_landmark_pixel(landmarks[side.value], w, h)
            angle = analysis_data.get('l_knee_angle' if side == mp_pose.PoseLandmark.LEFT_KNEE else 'r_knee_angle', 0)
            if angle > 0:
                cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class PushUpVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        super().__init__(get_analyzer_class("Push-Up"), recording_path, use_hybrid)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        _, get_landmark_pixel = get_utils()
        mp_pose = get_mp_pose()
        angle = analysis_data.get('elbow_angle', 0)
        if angle > 0:
            l_vis = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
            r_vis = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            target = mp_pose.PoseLandmark.LEFT_ELBOW if l_vis > r_vis else mp_pose.PoseLandmark.RIGHT_ELBOW
            pos = get_landmark_pixel(landmarks[target.value], w, h)
            cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class DeadliftVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        super().__init__(get_analyzer_class("Deadlift"), recording_path, use_hybrid)

class LungeVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        super().__init__(get_analyzer_class("Lunge"), recording_path, use_hybrid)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        _, get_landmark_pixel = get_utils()
        mp_pose = get_mp_pose()
        lead = analysis_data.get('lead_leg')
        if lead:
            target = mp_pose.PoseLandmark.LEFT_KNEE if lead == "LEFT" else mp_pose.PoseLandmark.RIGHT_KNEE
            pos = get_landmark_pixel(landmarks[target.value], w, h)
            angle = analysis_data.get('l_knee_angle' if lead == 'LEFT' else 'r_knee_angle', 0)
            cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class JumpingJacksVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        super().__init__(get_analyzer_class("Jumping Jacks"), recording_path, use_hybrid)

class PlankVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        super().__init__(get_analyzer_class("Plank"), recording_path, use_hybrid)

class BenchPressVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        super().__init__(get_analyzer_class("Bench Press"), recording_path, use_hybrid, variant="standard")
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        _, get_landmark_pixel = get_utils()
        mp_pose = get_mp_pose()
        angle = analysis_data.get('elbow_angle', 0)
        if angle > 0:
            l_vis = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
            r_vis = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            target = mp_pose.PoseLandmark.LEFT_ELBOW if l_vis > r_vis else mp_pose.PoseLandmark.RIGHT_ELBOW
            pos = get_landmark_pixel(landmarks[target.value], w, h)
            cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class SeatedBenchPressVideoProcessor(BenchPressVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        BaseVideoProcessor.__init__(self, get_analyzer_class("Seated Bench Press"), recording_path, use_hybrid, variant="seated")

class ChestFlyVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        super().__init__(get_analyzer_class("Chest Fly"), recording_path, use_hybrid, variant="standing")

class SeatedChestFlyVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        super().__init__(get_analyzer_class("Seated Chest Fly"), recording_path, use_hybrid, variant="seated")

class DipsVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        super().__init__(get_analyzer_class("Dips"), recording_path, use_hybrid, variant="normal")

class SeatedDipsVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        super().__init__(get_analyzer_class("Seated Dips"), recording_path, use_hybrid, variant="seated")

class AutoDetectVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(None, recording_path, use_hybrid)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        pass

def process_video(input_path, output_path, mode="Squat", use_hybrid=False):
    draw_text_with_background, get_landmark_pixel = get_utils()
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    detector = get_pose_detector(use_hybrid)
        
    # Smart Rotation Fix for Phone Videos
    rotation_code = None
    try:
        # Check first 30 frames for a valid pose to determine rotation
        for _ in range(30):
            ret, frame = cap.read()
            if not ret: break
            
            h, w, c = frame.shape
            # Only check if video is Landscape (Possible raw phone video)
            if w > h:
                test_img, _ = detector.find_pose(frame, draw=False)
                lm = detector.get_landmarks()
                if lm:
                    nose = lm[0]
                    l_hip, r_hip = lm[23], lm[24]
                    mid_hip_x = (l_hip.x + r_hip.x) / 2
                    mid_hip_y = (l_hip.y + r_hip.y) / 2
                    
                    dx = abs(nose.x - mid_hip_x)
                    dy = abs(nose.y - mid_hip_y)
                    
                    # If Nose X is far from Hip X (Sideways)
                    # Relaxed threshold: if dx > dy (horizontal distance > vertical), it's sideways
                    if dx > dy * 1.1: 
                        if nose.x < mid_hip_x: rotation_code = cv2.ROTATE_90_CLOCKWISE
                        else: rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
                        print(f"Smart Rotation Detected: {rotation_code}")
                    break
            
        # Reset to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except: pass
    
    if rotation_code is not None:
        width, height = height, width

    # Robust VideoWriter selection for both Local and Cloud
    out = None
    try:
        # Standard MJPG - most compatible with raw server containers
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
             # Fallback 1: XVID
             fourcc = cv2.VideoWriter_fourcc(*'XVID')
             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
             
        if not out.isOpened():
             # Fallback 2: mp4v
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
             
        if not out.isOpened():
             # Fallback 3: MJPG raw (0)
             out = cv2.VideoWriter(output_path, 0, fps, (width, height))
    except Exception as e:
        print(f"Video Writer Failed: {str(e)}")
        # Create an empty file to avoid downstream errors
        open(output_path, 'a').close()
        return output_path

    exercise_detector = get_exercise_detector()
    
    if mode == "Auto-Detect":
        analyzer = None
        exercise_name = "Detecting..."
    else:
        analyzer_class = get_analyzer_class(mode)
        if mode in ["Seated Bench Press", "Seated Chest Fly", "Seated Dips"]:
            analyzer = analyzer_class(variant="seated") if analyzer_class else None
        else:
            analyzer = analyzer_class() if analyzer_class else None
        exercise_name = mode
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)
        
        frame, _ = detector.find_pose(frame, draw=True)
        landmarks = detector.get_landmarks()
        
        if landmarks:
            if analyzer is None:
                exercise_detector.add_frame(landmarks)
                detected = exercise_detector.detect()
                if detected:
                    exercise_name = detected
                    analyzer_class = get_analyzer_class(detected)
                    if detected in ["Seated Bench Press", "Seated Chest Fly", "Seated Dips"]:
                        analyzer = analyzer_class(variant="seated") if analyzer_class else None
                    else:
                        analyzer = analyzer_class() if analyzer_class else None
                    
                    if analyzer and detected not in ["Plank"]: 
                        analyzer.rep_count = 1
                        analyzer.correct_reps = 1
                        analyzer.feedback = "Exercise Detected! Rep 1 counted."
            
            if analyzer:
                analysis_data = analyzer.analyze(landmarks, width, height)
            else:
                buffer_size = len(exercise_detector.buffer)
                required_size = int(exercise_detector.window_size * 0.7)
                votes_size = len(exercise_detector.confidence_votes)
                required_votes = int(exercise_detector.confidence_votes.maxlen * 0.8)
                
                feedback_msg = "Analyzing exercise pattern..."
                if buffer_size < required_size:
                    progress = int((buffer_size / required_size) * 100)
                    feedback_msg = f"Collecting data... {progress}%"
                elif votes_size < required_votes:
                    progress = int((votes_size / required_votes) * 100)
                    feedback_msg = f"Analyzing pattern... {progress}%"
                    if exercise_detector.last_guess:
                        feedback_msg += f" (Detecting: {exercise_detector.last_guess})"
                
                analysis_data = {
                    "state": "ANALYZING...", 
                    "rep_count": 0, 
                    "feedback": feedback_msg, 
                    "target_muscles": "Detecting...",
                    "correct_reps": 0,
                    "incorrect_reps": 0
                }

            state_color = (0, 255, 255)
            if analysis_data.get('state') in ["BOTTOM", "LOCKOUT", "TOP_PLANK"]: state_color = (0, 255, 0)
            
            y_pos = 40
            draw_text_with_background(frame, f"Exercise: {exercise_name}", (10, y_pos), text_color=(255, 255, 255))
            
            y_pos += 40
            draw_text_with_background(frame, f"State: {analysis_data['state']}", (10, y_pos), text_color=state_color)
            
            y_pos += 40
            draw_text_with_background(frame, f"Muscles: {analysis_data.get('target_muscles', 'N/A')}", (10, y_pos), font_scale=0.6, text_color=(255, 150, 0))
            
            y_pos += 40
            c_reps = analysis_data.get('correct_reps', 0)
            i_reps = analysis_data.get('incorrect_reps', 0)
            
            display_rep_count = analysis_data.get('rep_count', 0)
            if exercise_name not in ["Plank", "Deadlift", "Lunge"] and analysis_data.get('state') != "ANALYZING...":
                display_rep_count = c_reps + i_reps

            draw_text_with_background(frame, f"Reps: {display_rep_count}", (10, y_pos), font_scale=0.8, thickness=2)
            
            y_pos += 35
            
            draw_text_with_background(frame, f"Correct: {c_reps}", (10, y_pos), font_scale=0.6, text_color=(0, 255, 0))
            draw_text_with_background(frame, f"Incorrect: {i_reps}", (160, y_pos), font_scale=0.6, text_color=(0, 0, 255))
            
            y_pos += 40
            feedback = analysis_data.get('feedback', '')
            if feedback:
                draw_text_with_background(frame, f"Feedback: {feedback}", (10, y_pos), text_color=(0, 100, 255))
            
            score = analysis_data.get('last_rep_score', 0)
            if score > 0 and score < 70 and analysis_data.get('reasons'): 
                y_pos += 40
                for r in analysis_data['reasons'][:2]:
                    draw_text_with_background(frame, f"Fault: {r}", (10, y_pos), text_color=(0, 0, 255), font_scale=0.5)
                    y_pos += 25

        out.write(frame)
        frame_count += 1
        if total_frames > 0: progress_bar.progress(frame_count / total_frames)

    cap.release()
    out.release()
    progress_bar.empty()

def main():
    st.set_page_config(page_title="AI Fitness Coach", layout="wide")
    st.title("🏋️ AI Fitness Analysis Coach")
    
    if not WEBRTC_AVAILABLE:
        st.warning(f"⚠️ Webcam feature disabled: {WEBRTC_ERROR}. Video upload will still work!")
    
    # RTC Configuration for WebRTC - Initialize here
    global RTC_CONFIGURATION
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    if "webcam_recording" not in st.session_state:
        st.session_state.webcam_recording = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    # Setup Sidebar Options
    with st.sidebar:
        st.header("Settings")
        use_hybrid = st.checkbox("Use AI-Enhanced Detection (YOLO + MediaPipe)", value=False, 
                                 help="Turning this on improves person detection but requires more hardware.")
    
    exercise_type = st.radio("Select Exercise:", ["Auto-Detect", "Squat", "Lunge", "Push-Up", "Bench Press", "Seated Bench Press", "Deadlift", "Jumping Jacks", "Plank", "Chest Fly", "Seated Chest Fly", "Dips", "Seated Dips"], horizontal=True)
    
    # Conditionally create tabs based on WebRTC availability
    if WEBRTC_AVAILABLE:
        tab1, tab2 = st.tabs(["📹 Upload Video", "🎥 Live Webcam"])
    else:
        tab1 = st.container()
        st.info("ℹ️ Webcam feature is not available on this platform. Please use video upload.")
    
    with tab1:
        uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
            tfile.write(uploaded_file.read())
            tfile.close()
            st.video(tfile.name)
            
            if st.button(f'Analyze {exercise_type}'):
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                with st.spinner('Analyzing video...'):
                    process_video(tfile.name, temp_output, mode=exercise_type, use_hybrid=use_hybrid)
                
                # Re-encode for browser compatibility
                with st.spinner('Optimizing video for playback...'):
                    output_path, err = reencode_video_for_browser(temp_output)
                    if err: st.warning(f"Optimization warning: {err}")
                    
                st.success("Done!")
                st.video(output_path)
                
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Analyzed Video",
                        data=file,
                        file_name=f"analyzed_{exercise_type.lower()}.mp4",
                        mime="video/mp4"
                    )
    
    # Webcam tab - only if WebRTC is available
    if WEBRTC_AVAILABLE:
        with tab2:
            processors = {
                "Auto-Detect": AutoDetectVideoProcessor,
                "Squat": SquatVideoProcessor,
                "Lunge": LungeVideoProcessor,
                "Push-Up": PushUpVideoProcessor,
                "Bench Press": BenchPressVideoProcessor,
                "Seated Bench Press": SeatedBenchPressVideoProcessor,
                "Deadlift": DeadliftVideoProcessor,
                "Jumping Jacks": JumpingJacksVideoProcessor,
                "Plank": PlankVideoProcessor,
                "Chest Fly": ChestFlyVideoProcessor,
                "Seated Chest Fly": SeatedChestFlyVideoProcessor,
                "Dips": DipsVideoProcessor,
                "Seated Dips": SeatedDipsVideoProcessor
            }
            
            recording_path = st.session_state.webcam_recording
            
            ctx = webrtc_streamer(
                key=f"{exercise_type.lower()}-analysis",
                video_processor_factory=lambda: processors[exercise_type](recording_path, use_hybrid=use_hybrid),
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            if ctx.video_processor:
                st.info("Recording session... The video will be available after you stop the webcam.")
            
            if not ctx.state.playing and ctx.video_processor:
                ctx.video_processor.stop_recording()
                
            if not ctx.state.playing and os.path.exists(st.session_state.webcam_recording):
                if os.path.getsize(st.session_state.webcam_recording) > 1000:
                    st.subheader("📊 Last Session Results")
                    
                    # Re-encode for browser compatibility if not already done
                    if not hasattr(st.session_state, 'webcam_recording_encoded') or \
                       st.session_state.webcam_recording_encoded != st.session_state.webcam_recording:
                        with st.spinner('Optimizing video for playback...'):
                            encoded_path, err = reencode_video_for_browser(st.session_state.webcam_recording)
                            if err: st.warning(f"Optimization warning: {err}")
                            st.session_state.webcam_recording_display = encoded_path
                            st.session_state.webcam_recording_encoded = st.session_state.webcam_recording
                    
                    display_video = st.session_state.get('webcam_recording_display', st.session_state.webcam_recording)
                    st.video(display_video)
                    
                    with open(display_video, "rb") as file:
                        st.download_button(
                            label="📥 Download Webcam Session",
                            data=file,
                            file_name=f"webcam_{exercise_type.lower()}.mp4",
                            mime="video/mp4"
                        )
                    
                    if st.button("🗑️ Clear Recording"):
                        if os.path.exists(st.session_state.webcam_recording):
                            os.remove(st.session_state.webcam_recording)
                        st.session_state.webcam_recording = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
                        st.rerun()
                elif os.path.exists(st.session_state.webcam_recording) and os.path.getsize(st.session_state.webcam_recording) > 0:
                    st.warning("Recording was too short or could not be processed.")

if __name__ == '__main__':
    main()
