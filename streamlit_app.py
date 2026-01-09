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
    def __init__(self, analyzer_class):
        self.detector = PoseDetector()
        self.analyzer = analyzer_class() if analyzer_class else None
        self.exercise_name = analyzer_class.__name__.replace('Analyzer', '') if analyzer_class else "Detecting..."
        self.exercise_detector = ExerciseDetector()
        self.p_time = 0

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
                        "Deadlift": DeadliftAnalyzer, "Lunge": LungeAnalyzer,
                        "Jumping Jacks": JumpingJacksAnalyzer, "Plank": PlankAnalyzer
                    }
                    self.analyzer = analyzers[detected]()
            
            if self.analyzer:
                analysis_data = self.analyzer.analyze(landmarks, w, h)
            else:
                analysis_data = {"state": "DETECTING...", "rep_count": 0, "feedback": "Perform an exercise to start", "target_muscles": "None"}
            self._draw_specifics(img, landmarks, analysis_data, w, h)
        
        # FPS
        c_time = time.time()
        fps = 1 / (c_time - self.p_time) if self.p_time > 0 else 0
        self.p_time = c_time
        
        # Overlay
        draw_text_with_background(img, f"FPS: {int(fps)}", (10, 30), text_color=(0, 255, 0))
        
        if analysis_data:
            self._draw_common_overlay(img, analysis_data, w, h)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        pass

    def _draw_common_overlay(self, img, analysis_data, w, h):
        state_color = (0, 255, 255)
        if analysis_data.get('state') in ["BOTTOM", "LOCKOUT", "TOP_PLANK"]: 
            state_color = (0, 255, 0)
            
        # Vertical stacking with consistent spacing
        y_pos = 40
        draw_text_with_background(img, f"Exercise: {self.exercise_name}", (10, y_pos), text_color=(255, 255, 255))
        
        y_pos += 40
        draw_text_with_background(img, f"State: {analysis_data['state']}", (10, y_pos), text_color=state_color)
        
        y_pos += 40
        draw_text_with_background(img, f"Muscles: {analysis_data.get('target_muscles', 'N/A')}", (10, y_pos), font_scale=0.6, text_color=(255, 150, 0))
        
        y_pos += 40
        draw_text_with_background(img, f"Total Reps: {analysis_data['rep_count']}", (10, y_pos), font_scale=0.8, thickness=2)
        
        y_pos += 35
        c_reps = analysis_data.get('correct_reps', 0)
        i_reps = analysis_data.get('incorrect_reps', 0)
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
        
        # Display reasons for failure if score is low
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
    def __init__(self): super().__init__(SquatAnalyzer)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        for side in [mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_KNEE]:
            pos = get_landmark_pixel(landmarks[side.value], w, h)
            angle = analysis_data.get('l_knee_angle' if side == mp.solutions.pose.PoseLandmark.LEFT_KNEE else 'r_knee_angle', 0)
            if angle > 0:
                cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class PushUpVideoProcessor(BaseVideoProcessor):
    def __init__(self): super().__init__(PushUpAnalyzer)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        angle = analysis_data.get('elbow_angle', 0)
        if angle > 0:
            l_vis = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].visibility
            r_vis = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            target = mp.solutions.pose.PoseLandmark.LEFT_ELBOW if l_vis > r_vis else mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
            pos = get_landmark_pixel(landmarks[target.value], w, h)
            cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class DeadliftVideoProcessor(BaseVideoProcessor):
    def __init__(self): super().__init__(DeadliftAnalyzer)

class LungeVideoProcessor(BaseVideoProcessor):
    def __init__(self): super().__init__(LungeAnalyzer)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        lead = analysis_data.get('lead_leg')
        if lead:
            target = mp.solutions.pose.PoseLandmark.LEFT_KNEE if lead == "LEFT" else mp.solutions.pose.PoseLandmark.RIGHT_KNEE
            pos = get_landmark_pixel(landmarks[target.value], w, h)
            angle = analysis_data.get('l_knee_angle' if lead == 'LEFT' else 'r_knee_angle', 0)
            cv2.putText(img, f"{int(angle)}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

class JumpingJacksVideoProcessor(BaseVideoProcessor):
    def __init__(self): super().__init__(JumpingJacksAnalyzer)

class PlankVideoProcessor(BaseVideoProcessor):
    def __init__(self): super().__init__(PlankAnalyzer)

class AutoDetectVideoProcessor(BaseVideoProcessor):
    def __init__(self): super().__init__(None)
    def _draw_specifics(self, img, landmarks, analysis_data, w, h):
        pass

def process_video(input_path, output_path, mode="Squat"):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Codec: Try 'avc1' (H.264) for better browser support. Fallback to 'mp4v' if needed.
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    detector = PoseDetector()
    exercise_detector = ExerciseDetector()
    analyzers = {
        "Squat": SquatAnalyzer, "Push-Up": PushUpAnalyzer, "Deadlift": DeadliftAnalyzer, 
        "Lunge": LungeAnalyzer, "Jumping Jacks": JumpingJacksAnalyzer, "Plank": PlankAnalyzer
    }
    
    if mode == "Auto-Detect":
        analyzer = None
        exercise_name = "Detecting..."
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
                    analyzer = analyzers[detected]()
            
            if analyzer:
                analysis_data = analyzer.analyze(landmarks, width, height)
            else:
                analysis_data = {"state": "DETECTING...", "rep_count": 0, "target_muscles": "None"}

            # Full Dashboard for Video
            state_color = (0, 255, 255)
            if analysis_data.get('state') in ["BOTTOM", "LOCKOUT", "TOP_PLANK"]: state_color = (0, 255, 0)
            
            y_pos = 40
            draw_text_with_background(frame, f"Exercise: {exercise_name}", (10, y_pos), text_color=(255, 255, 255))
            
            y_pos += 40
            draw_text_with_background(frame, f"State: {analysis_data['state']}", (10, y_pos), text_color=state_color)
            
            y_pos += 40
            draw_text_with_background(frame, f"Muscles: {analysis_data.get('target_muscles', 'N/A')}", (10, y_pos), font_scale=0.6, text_color=(255, 150, 0))
            
            y_pos += 40
            draw_text_with_background(frame, f"Total Reps: {analysis_data['rep_count']}", (10, y_pos), font_scale=0.8, thickness=2)
            
            y_pos += 35
            c_reps = analysis_data.get('correct_reps', 0)
            i_reps = analysis_data.get('incorrect_reps', 0)
            draw_text_with_background(frame, f"Correct: {c_reps}", (10, y_pos), font_scale=0.6, text_color=(0, 255, 0))
            draw_text_with_background(frame, f"Incorrect: {i_reps}", (160, y_pos), font_scale=0.6, text_color=(0, 0, 255))
            
            y_pos += 40
            feedback = analysis_data.get('feedback', '')
            if feedback:
                draw_text_with_background(frame, f"Feedback: {feedback}", (10, y_pos), text_color=(0, 100, 255))
            
            # Display reasons if incorrect
            score = analysis_data.get('last_rep_score', 0)
            if score > 0 and score < 70 and analysis_data.get('reasons'): # Updated to 70
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
    
    exercise_type = st.radio("Select Exercise:", ["Auto-Detect", "Squat", "Lunge", "Push-Up", "Deadlift", "Jumping Jacks", "Plank"], horizontal=True)
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
                    
                    # Add Download Button
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
            "Deadlift": DeadliftVideoProcessor,
            "Jumping Jacks": JumpingJacksVideoProcessor,
            "Plank": PlankVideoProcessor
        }
        webrtc_streamer(
            key=f"{exercise_type.lower()}-analysis",
            video_processor_factory=processors[exercise_type],
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

if __name__ == '__main__':
    main()
