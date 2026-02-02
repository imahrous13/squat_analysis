import os
# CRITICAL: Set environment variables BEFORE any other imports to prevent GPU crashes
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python' # Fixes common Cloud segfaults
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MP_GPU_MODE'] = '0' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-streamlit'

import streamlit as st
import cv2
import tempfile
import time
import subprocess
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Gym Analyzer v1.0.1
# Internal imports
from src.core.movenet_pose import MoveNetEstimator as PoseDetector
from src.core.hybrid_pose import HybridPoseEstimator
from src.core.exercise_detector import ExerciseDetector
from src.analyzers.squat_analyzer import SquatAnalyzer
from src.analyzers.pushup_analyzer import PushUpAnalyzer
from src.analyzers.bench_press_analyzer import BenchPressAnalyzer
from src.analyzers.deadlift_analyzer import DeadliftAnalyzer
from src.analyzers.lunge_analyzer import LungeAnalyzer
from src.analyzers.jumping_jacks_analyzer import JumpingJacksAnalyzer
from src.analyzers.plank_analyzer import PlankAnalyzer
from src.analyzers.chest_fly_analyzer import ChestFlyAnalyzer
from src.analyzers.dips_analyzer import DipsAnalyzer
from src.core.utils import draw_text_with_background, get_landmark_pixel

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
def reencode_video_for_browser(input_path, output_path=None):
    """Re-encode video using PyAV (internal) for browser compatibility, avoiding external ffmpeg binary."""
    if not os.path.exists(input_path):
        return input_path, "Input file not found."
    
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

    try:
        import av
        input_container = av.open(input_path)
        output_container = av.open(output_path, mode='w', format='mp4')
        
        # We assume first video stream
        in_stream = input_container.streams.video[0]
        
        # Configure output stream: H.264 with high compatibility
        out_stream = output_container.add_stream('libx264', rate=in_stream.base_rate)
        out_stream.width = in_stream.width
        out_stream.height = in_stream.height
        out_stream.pix_fmt = 'yuv420p' # Crucial for browser playback
        out_stream.options = {'preset': 'veryfast', 'crf': '23'}

        for frame in input_container.decode(video=0):
            # Convert frame to output stream's pixel format
            for packet in out_stream.encode(frame):
                output_container.mux(packet)

        # Flush output
        for packet in out_stream.encode():
            output_container.mux(packet)

        input_container.close()
        output_container.close()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, None
        else:
            return input_path, "Re-encoding produced an empty file."

    except Exception as e:
        return input_path, f"Internal re-encoding failed: {str(e)}. Please ensure 'av' library is properly installed."


class BaseVideoProcessor(VideoProcessorBase):
    def __init__(self, analyzer_class, recording_path=None, use_hybrid=False, **analyzer_args):
        if use_hybrid:
            # Check if model exists, if not warn/fallback (though we downloaded it)
            if os.path.exists('yolov8n.pt'):
                self.detector = HybridPoseEstimator(model_path='yolov8n.pt')
            else:
                self.detector = PoseDetector()
                print("Warning: yolov8n.pt not found, falling back to Standard Pose.")
        else:
            self.detector = PoseDetector()
            
        self.analyzer = analyzer_class(**analyzer_args) if analyzer_class else None
        self.exercise_name = analyzer_class.__name__.replace('Analyzer', '') if analyzer_class else "Detecting..."
        self.exercise_detector = ExerciseDetector()
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
                    analyzers = {
                        "Squat": SquatAnalyzer, "Push-Up": PushUpAnalyzer, 
                        "Bench Press": BenchPressAnalyzer, "Seated Bench Press": BenchPressAnalyzer,
                        "Deadlift": DeadliftAnalyzer,
                        "Lunge": LungeAnalyzer, "Jumping Jacks": JumpingJacksAnalyzer,
                        "Plank": PlankAnalyzer,
                        "Chest Fly": ChestFlyAnalyzer, "Seated Chest Fly": ChestFlyAnalyzer,
                        "Dips": DipsAnalyzer, "Seated Dips": DipsAnalyzer
                    }

                    if detected == "Seated Bench Press":
                        self.analyzer = BenchPressAnalyzer(variant="seated")
                    elif detected == "Seated Chest Fly":
                        self.analyzer = ChestFlyAnalyzer(variant="seated")
                    elif detected == "Seated Dips":
                        self.analyzer = DipsAnalyzer(variant="seated")
                    elif detected in analyzers:
                         self.analyzer = analyzers[detected]()
                    
                    # Credit the rep used for detection (Auto-Detect "Free" Rep)
                    if self.analyzer and detected not in ["Plank"]: 
                        self.analyzer.rep_count = 1
                        self.analyzer.correct_reps = 1
                        self.analyzer.feedback = "Exercise Detected! Rep 1 counted."
            
            if self.analyzer:
                analysis_data = self.analyzer.analyze(landmarks, w, h)
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
                    # Use MJPG as codec - more stable on headless servers
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    self.out = cv2.VideoWriter(self.recording_path, fourcc, 20.0, (w, h))
                    if not self.out.isOpened():
                        # Fallback to mp4v
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.out = cv2.VideoWriter(self.recording_path, fourcc, 20.0, (w, h))
                except Exception as e:
                    print(f"Webcam Recorder Error: {e}")
            
            if self.out and self.out.isOpened():
                self.out.write(img)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        pass

    def _draw_common_overlay(self, img, analysis_data, w, h):
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
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(SquatAnalyzer, recording_path, use_hybrid)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        for side in [mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_KNEE]:
            pos = get_landmark_pixel(landmarks[side.value], w, h)
            angle = analysis_data.get('l_knee_angle' if side == mp.solutions.pose.PoseLandmark.LEFT_KNEE else 'r_knee_angle', 0)
            if angle > 0:
                cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class PushUpVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(PushUpAnalyzer, recording_path, use_hybrid)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        angle = analysis_data.get('elbow_angle', 0)
        if angle > 0:
            l_vis = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].visibility
            r_vis = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            target = mp.solutions.pose.PoseLandmark.LEFT_ELBOW if l_vis > r_vis else mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
            pos = get_landmark_pixel(landmarks[target.value], w, h)
            cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class DeadliftVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(DeadliftAnalyzer, recording_path, use_hybrid)

class LungeVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(LungeAnalyzer, recording_path, use_hybrid)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        lead = analysis_data.get('lead_leg')
        if lead:
            target = mp.solutions.pose.PoseLandmark.LEFT_KNEE if lead == "LEFT" else mp.solutions.pose.PoseLandmark.RIGHT_KNEE
            pos = get_landmark_pixel(landmarks[target.value], w, h)
            angle = analysis_data.get('l_knee_angle' if lead == 'LEFT' else 'r_knee_angle', 0)
            cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class JumpingJacksVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(JumpingJacksAnalyzer, recording_path, use_hybrid)

class PlankVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(PlankAnalyzer, recording_path, use_hybrid)

class BenchPressVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(BenchPressAnalyzer, recording_path, use_hybrid, variant="standard")
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        angle = analysis_data.get('elbow_angle', 0)
        if angle > 0:
            l_vis = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].visibility
            r_vis = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            target = mp.solutions.pose.PoseLandmark.LEFT_ELBOW if l_vis > r_vis else mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
            pos = get_landmark_pixel(landmarks[target.value], w, h)
            cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class SeatedBenchPressVideoProcessor(BenchPressVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): 
        # Call Base init directly to pass variant="seated"
        BaseVideoProcessor.__init__(self, BenchPressAnalyzer, recording_path, use_hybrid, variant="seated")

class ChestFlyVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(ChestFlyAnalyzer, recording_path, use_hybrid, variant="standing")

class SeatedChestFlyVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(ChestFlyAnalyzer, recording_path, use_hybrid, variant="seated")

class DipsVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(DipsAnalyzer, recording_path, use_hybrid, variant="normal")

class SeatedDipsVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(DipsAnalyzer, recording_path, use_hybrid, variant="seated")

class AutoDetectVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None, use_hybrid=False): super().__init__(None, recording_path, use_hybrid)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        pass

def process_video(input_path, output_path, mode="Squat", use_hybrid=False):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60: fps = 20.0
    
    # Silent processing for better UX
    
    if use_hybrid and os.path.exists('yolov8n.pt'):
        detector = HybridPoseEstimator(model_path='yolov8n.pt')
    else:
        detector = PoseDetector()
        
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
            
        # Reset capture properly
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except Exception as e: 
        print(f"Rotation check error: {e}")
        cap.release()
        cap = cv2.VideoCapture(input_path) # Hard re-open only on failure
    
    if rotation_code is not None:
        width, height = height, width

    # Robust VideoWriter Selection
    out = None
    try:
        # For .avi intermediate, MJPG or XVID are the best choices
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
             fourcc = cv2.VideoWriter_fourcc(*'XVID')
             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
             
        if not out.isOpened():
             # Last resort fallbacks
             for codec in ['mp4v', 'XVID', 'X264', 'av1 ', 'DIVX']:
                 try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if out.isOpened(): break
                 except: continue
             
             if not out.isOpened():
                # Ultimate fallback: MJPG via raw 0
                out = cv2.VideoWriter(output_path, 0, fps, (width, height))
    except Exception as e:
        print(f"Video Writer Failed: {str(e)}")
        open(output_path, 'a').close()
        return output_path

    exercise_detector = ExerciseDetector()
    analyzers = {
        "Squat": SquatAnalyzer, "Push-Up": PushUpAnalyzer, 
        "Bench Press": BenchPressAnalyzer, "Seated Bench Press": BenchPressAnalyzer,
        "Deadlift": DeadliftAnalyzer, "Lunge": LungeAnalyzer, 
        "Jumping Jacks": JumpingJacksAnalyzer, "Plank": PlankAnalyzer,
        "Chest Fly": ChestFlyAnalyzer, "Seated Chest Fly": ChestFlyAnalyzer,
        "Dips": DipsAnalyzer, "Seated Dips": DipsAnalyzer
    }
    
    if mode == "Auto-Detect":
        analyzer = None
        exercise_name = "Detecting..."
    elif mode == "Seated Bench Press":
         analyzer = BenchPressAnalyzer(variant="seated")
         exercise_name = mode
    elif mode == "Seated Chest Fly":
         analyzer = ChestFlyAnalyzer(variant="seated")
         exercise_name = mode
    elif mode == "Seated Dips":
         analyzer = DipsAnalyzer(variant="seated")
         exercise_name = mode
    else:
        analyzer = analyzers.get(mode, SquatAnalyzer)()
        exercise_name = mode
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)
        
        # Default data for frames with no landmarks
        analysis_data = {
            "state": "NO PERSON", "rep_count": 0, "correct_reps": 0, "incorrect_reps": 0,
            "feedback": "Looking for person...", "advice": "", "last_rep_score": 0,
            "reasons": [], "view": "N/A", "target_muscles": "N/A"
        }
        state_color = (0, 255, 255)

        frame, _ = detector.find_pose(frame, draw=True)
        landmarks = detector.get_landmarks()
        
        if landmarks:
            if analyzer is None:
                exercise_detector.add_frame(landmarks)
                detected = exercise_detector.detect()
                if detected:
                    exercise_name = detected
                    if detected == "Seated Bench Press":
                        analyzer = BenchPressAnalyzer(variant="seated")
                    elif detected == "Seated Chest Fly":
                        analyzer = ChestFlyAnalyzer(variant="seated")
                    elif detected == "Seated Dips":
                        analyzer = DipsAnalyzer(variant="seated")
                    elif detected in analyzers:
                        analyzer = analyzers[detected]()
                    
                    if analyzer and detected == "Plank": 
                        analyzer.is_active = True
                        analyzer.feedback = "Exercise Detected! Hold your plank."
                    elif analyzer:
                        analyzer.feedback = "Exercise Detected! Start your reps."
                        # DO NOT hardcode rep_count = 1 here to avoid double counting
            
            if analyzer:
                try:
                    analysis_data = analyzer.analyze(landmarks, width, height)
                except Exception as e:
                    print(f"Analyzer Error: {e}")
                    analysis_data["feedback"] = "Processing error, skipping frame..."
                    analysis_data["state"] = "ERROR"
            else:
                buffer_size = len(exercise_detector.buffer)
                required_size = int(exercise_detector.window_size * 0.7)
                votes_size = len(exercise_detector.confidence_votes)
                required_votes = int(exercise_detector.confidence_votes.maxlen * 0.8)
                
                feedback_msg = "Analyzing exercise pattern..."
                if buffer_size < required_size:
                    feedback_msg += f" ({buffer_size}/{required_size} frames)"
                elif votes_size < required_votes:
                    feedback_msg += f" (Confidence {votes_size}/{required_votes})"
                
                analysis_data["state"] = "ANALYZING..."
                analysis_data["feedback"] = feedback_msg

        # Always draw UI
        if analysis_data.get('state') in ["BOTTOM", "LOCKOUT", "TOP_PLANK"]: 
            state_color = (0, 255, 0)
        
        y_pos = 40
        draw_text_with_background(frame, f"Exercise: {exercise_name}", (10, y_pos), text_color=(255, 255, 255))
        
        y_pos += 40
        draw_text_with_background(frame, f"State: {analysis_data.get('state', 'UNKNOWN')}", (10, y_pos), text_color=state_color)
        
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

        if out and out.isOpened():
            out.write(frame)
        
        frame_count += 1
        if total_frames > 0: 
            progress_bar.progress(min(1.0, frame_count / total_frames))

    cap.release()
    if out: out.release()
    progress_bar.empty()
    
    import gc
    gc.collect() # Crucial for Streamlit Cloud memory limits

def main():
    st.set_page_config(page_title="AI Fitness Coach", layout="wide")
    st.title("🏋️ AI Fitness Analysis Coach")
    
    if "webcam_recording" not in st.session_state:
        st.session_state.webcam_recording = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    
    # Setup Sidebar Options
    with st.sidebar:
        st.header("Settings")
        use_hybrid = st.checkbox("Use AI-Enhanced Detection (YOLO + MediaPipe)", value=False, 
                                 help="Turning this on improves person detection but requires more hardware.")
        show_debug = st.checkbox("Show Debug Diagnostics", value=False)
    
    if show_debug:
        st.info("🔍 System Information:")
        st.write(f"OS: {os.name}")
        st.write(f"OpenCV: {cv2.__version__}")
        st.write(f"MediaPipe: {mp.__version__}")
        try:
            import torch
            st.write(f"Torch Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        except: st.write("Torch: Not found")
    
    exercise_type = st.radio("Select Exercise:", ["Auto-Detect", "Squat", "Lunge", "Push-Up", "Bench Press", "Seated Bench Press", "Deadlift", "Jumping Jacks", "Plank", "Chest Fly", "Seated Chest Fly", "Dips", "Seated Dips"], horizontal=True)
    tab1, tab2 = st.tabs(["📹 Upload Video", "🎥 Live Webcam"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
            tfile.write(uploaded_file.read())
            tfile.close()
            st.video(tfile.name)
            
            if st.button(f'Analyze {exercise_type}'):
                try:
                    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name 
                    with st.spinner('Analyzing video...'):
                        process_video(tfile.name, temp_output, mode=exercise_type, use_hybrid=use_hybrid)
                    
                    # Re-encode for browser compatibility
                    with st.spinner('Optimizing video for playback...'):
                        output_path, err = reencode_video_for_browser(temp_output)
                        if err: st.warning(f"⚠️ {err}")
                        
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                        st.success("Done!")
                        st.video(output_path)
                        
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="Download Analyzed Video",
                                data=file,
                                file_name=f"analyzed_{exercise_type.lower()}.mp4",
                                mime="video/mp4"
                            )
                    else:
                        st.error("Generated video is empty. The server could not write the video file.")
                        if os.path.exists(temp_output):
                            st.info(f"Intermediate file size: {os.path.getsize(temp_output)} bytes")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.exception(e)
    
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
                        if err: st.warning(f"⚠️ {err}")
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
    try:
        main()
    except Exception as e:
        st.error("🚀 A critical application error occurred.")
        st.exception(e)
