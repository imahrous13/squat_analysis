import streamlit as st
import cv2
import tempfile
import os
import time
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Internal imports
from src.core.movenet_pose import MoveNetEstimator as PoseDetector
from src.core.exercise_detector import ExerciseDetector
from src.analyzers.squat_analyzer import SquatAnalyzer
from src.analyzers.pushup_analyzer import PushUpAnalyzer
from src.analyzers.bench_press_analyzer import BenchPressAnalyzer
from src.analyzers.deadlift_analyzer import DeadliftAnalyzer
from src.analyzers.lunge_analyzer import LungeAnalyzer
from src.analyzers.jumping_jacks_analyzer import JumpingJacksAnalyzer
from src.analyzers.plank_analyzer import PlankAnalyzer
from src.core.utils import draw_text_with_background, get_landmark_pixel

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class BaseVideoProcessor(VideoProcessorBase):
    def __init__(self, analyzer_class, recording_path=None, **analyzer_args):
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
                        "Plank": PlankAnalyzer
                    }

                    if detected == "Seated Bench Press":
                        self.analyzer = BenchPressAnalyzer(variant="seated")
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
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    self.out = cv2.VideoWriter(self.recording_path, fourcc, 20.0, (w, h))
                    if not self.out.isOpened():
                        raise Exception("avc1 failed")
                except:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.out = cv2.VideoWriter(self.recording_path, fourcc, 20.0, (w, h))
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
    def __init__(self, recording_path=None): super().__init__(SquatAnalyzer, recording_path)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        for side in [mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_KNEE]:
            pos = get_landmark_pixel(landmarks[side.value], w, h)
            angle = analysis_data.get('l_knee_angle' if side == mp.solutions.pose.PoseLandmark.LEFT_KNEE else 'r_knee_angle', 0)
            if angle > 0:
                cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class PushUpVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None): super().__init__(PushUpAnalyzer, recording_path)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        angle = analysis_data.get('elbow_angle', 0)
        if angle > 0:
            l_vis = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].visibility
            r_vis = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            target = mp.solutions.pose.PoseLandmark.LEFT_ELBOW if l_vis > r_vis else mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
            pos = get_landmark_pixel(landmarks[target.value], w, h)
            cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class DeadliftVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None): super().__init__(DeadliftAnalyzer, recording_path)

class LungeVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None): super().__init__(LungeAnalyzer, recording_path)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        lead = analysis_data.get('lead_leg')
        if lead:
            target = mp.solutions.pose.PoseLandmark.LEFT_KNEE if lead == "LEFT" else mp.solutions.pose.PoseLandmark.RIGHT_KNEE
            pos = get_landmark_pixel(landmarks[target.value], w, h)
            angle = analysis_data.get('l_knee_angle' if lead == 'LEFT' else 'r_knee_angle', 0)
            cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class JumpingJacksVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None): super().__init__(JumpingJacksAnalyzer, recording_path)

class PlankVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None): super().__init__(PlankAnalyzer, recording_path)

class BenchPressVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None): super().__init__(BenchPressAnalyzer, recording_path, variant="standard")
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        angle = analysis_data.get('elbow_angle', 0)
        if angle > 0:
            l_vis = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].visibility
            r_vis = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            target = mp.solutions.pose.PoseLandmark.LEFT_ELBOW if l_vis > r_vis else mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
            pos = get_landmark_pixel(landmarks[target.value], w, h)
            cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class SeatedBenchPressVideoProcessor(BenchPressVideoProcessor):
    def __init__(self, recording_path=None): 
        BaseVideoProcessor.__init__(self, BenchPressAnalyzer, recording_path, variant="seated")

class AutoDetectVideoProcessor(BaseVideoProcessor):
    def __init__(self, recording_path=None): super().__init__(None, recording_path)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        pass

def process_video(input_path, output_path, mode="Squat"):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    detector = PoseDetector()
    exercise_detector = ExerciseDetector()
    analyzers = {
        "Squat": SquatAnalyzer, "Push-Up": PushUpAnalyzer, "Bench Press": BenchPressAnalyzer,
        "Deadlift": DeadliftAnalyzer, "Lunge": LungeAnalyzer, 
        "Jumping Jacks": JumpingJacksAnalyzer, "Plank": PlankAnalyzer
    }
    
    if mode == "Auto-Detect":
        analyzer = None
        exercise_name = "Detecting..."
    elif mode == "Seated Bench Press":
         analyzer = BenchPressAnalyzer(variant="seated")
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
                    elif detected in analyzers:
                        analyzer = analyzers[detected]()
                    
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
    
    if "webcam_recording" not in st.session_state:
        st.session_state.webcam_recording = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    exercise_type = st.radio("Select Exercise:", ["Auto-Detect", "Squat", "Lunge", "Push-Up", "Bench Press", "Seated Bench Press", "Deadlift", "Jumping Jacks", "Plank"], horizontal=True)
    tab1, tab2 = st.tabs(["📹 Upload Video", "🎥 Live Webcam"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
            tfile.write(uploaded_file.read())
            tfile.close()
            st.video(tfile.name)
            
            if st.button(f'Analyze {exercise_type}'):
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                with st.spinner('Analyzing...'):
                    process_video(tfile.name, output_path, mode=exercise_type)
                    st.success("Done!")
                    st.video(output_path)
                    
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="Download Analyzed Video",
                            data=file,
                            file_name=f"analyzed_{exercise_type.lower()}.mp4",
                            mime="video/mp4"
                        )
    
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
            "Plank": PlankVideoProcessor
        }
        
        recording_path = st.session_state.webcam_recording
        
        ctx = webrtc_streamer(
            key=f"{exercise_type.lower()}-analysis",
            video_processor_factory=lambda: processors[exercise_type](recording_path),
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
                st.video(st.session_state.webcam_recording)
                
                with open(st.session_state.webcam_recording, "rb") as file:
                    st.download_button(
                        label="📥 Download Webcam Session",
                        data=file,
                        file_name=f"webcam_{exercise_type.lower()}.mp4",
                        mime="video/mp4"
                    )
                
                if st.button("🗑️ Clear Recording"):
                    if os.path.exists(st.session_state.webcam_recording):
                        os.remove(st.session_state.webcam_recording)
                    st.session_state.webcam_recording = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    st.rerun()
            elif os.path.exists(st.session_state.webcam_recording) and os.path.getsize(st.session_state.webcam_recording) > 0:
                st.warning("Recording was too short or could not be processed.")

if __name__ == '__main__':
    main()
