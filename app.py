import os
# CRITICAL: Set environment variables BEFORE any other imports to prevent GPU crashes
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MP_GPU_MODE'] = '0' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-streamlit'
os.makedirs(os.environ['XDG_RUNTIME_DIR'], exist_ok=True)

import streamlit as st
import cv2
import tempfile
import time
import subprocess
import mediapipe as mp
import numpy as np
import gc

# Headless stability
try:
    import matplotlib
    matplotlib.use('Agg')
except: pass

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
except ImportError:
    webrtc_streamer, VideoProcessorBase, RTCConfiguration = None, None, None

# Lazy Loader for components
def get_analyzer(mode):
    if mode == "Squat":
        from src.analyzers.squat_analyzer import SquatAnalyzer
        return SquatAnalyzer()
    elif mode == "Push-Up":
        from src.analyzers.pushup_analyzer import PushUpAnalyzer
        return PushUpAnalyzer()
    elif mode == "Bench Press":
        from src.analyzers.bench_press_analyzer import BenchPressAnalyzer
        return BenchPressAnalyzer(variant="standard")
    elif mode == "Seated Bench Press":
        from src.analyzers.bench_press_analyzer import BenchPressAnalyzer
        return BenchPressAnalyzer(variant="seated")
    elif mode == "Deadlift":
        from src.analyzers.deadlift_analyzer import DeadliftAnalyzer
        return DeadliftAnalyzer()
    elif mode == "Lunge":
        from src.analyzers.lunge_analyzer import LungeAnalyzer
        return LungeAnalyzer()
    elif mode == "Jumping Jacks":
        from src.analyzers.jumping_jacks_analyzer import JumpingJacksAnalyzer
        return JumpingJacksAnalyzer()
    elif mode == "Plank":
        from src.analyzers.plank_analyzer import PlankAnalyzer
        return PlankAnalyzer()
    elif mode == "Chest Fly":
        from src.analyzers.chest_fly_analyzer import ChestFlyAnalyzer
        return ChestFlyAnalyzer(variant="standing")
    elif mode == "Seated Chest Fly":
        from src.analyzers.chest_fly_analyzer import ChestFlyAnalyzer
        return ChestFlyAnalyzer(variant="seated")
    elif mode == "Dips":
        from src.analyzers.dips_analyzer import DipsAnalyzer
        return DipsAnalyzer(variant="normal")
    elif mode == "Seated Dips":
        from src.analyzers.dips_analyzer import DipsAnalyzer
        return DipsAnalyzer(variant="seated")
    return None

def get_detector(use_hybrid=False):
    if use_hybrid:
        from src.core.hybrid_pose import HybridPoseEstimator
        return HybridPoseEstimator(model_path='yolov8n.pt')
    else:
        from src.core.movenet_pose import MoveNetEstimator
        return MoveNetEstimator()

def reencode_video_for_browser(input_path, output_path=None):
    """Re-encode using PyAV for maximum compatibility."""
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
    def __init__(self, mode, recording_path=None, use_hybrid=False):
        from src.core.exercise_detector import ExerciseDetector
        self.detector = get_detector(use_hybrid)
        self.analyzer = get_analyzer(mode)
        self.exercise_name = mode
        self.exercise_detector = ExerciseDetector() if mode == "Auto-Detect" else None
        self.recording_path = recording_path
        self.out = None

    def recv(self, frame):
        from src.core.utils import draw_text_with_background
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # Process Pose
        img, _ = self.detector.find_pose(img, draw=True)
        landmarks = self.detector.get_landmarks()
        
        analysis_data = {"state": "N/A", "feedback": "Looking for person..."}
        if landmarks:
            if self.analyzer is None and self.exercise_detector:
                self.exercise_detector.add_frame(landmarks)
                detected = self.exercise_detector.detect()
                if detected: 
                    self.exercise_name = detected
                    self.analyzer = get_analyzer(detected)
            
            if self.analyzer:
                analysis_data = self.analyzer.analyze(landmarks, w, h)
        
        # Simple Overlay for Live
        draw_text_with_background(img, f"Exercise: {self.exercise_name}", (10, 40))
        draw_text_with_background(img, f"Reps: {analysis_data.get('rep_count', 0)}", (10, 80))
        draw_text_with_background(img, f"Feedback: {analysis_data.get('feedback', '')}", (10, 120))
        
        if self.recording_path and self.out is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.out = cv2.VideoWriter(self.recording_path, fourcc, 20.0, (w, h))
        
        if self.out: self.out.write(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def stop_recording(self):
        if self.out: self.out.release(); self.out = None

# Video Processing logic
def process_video(input_path, output_path, mode="Squat", use_hybrid=False):
    from src.core.utils import draw_text_with_background, get_landmark_pixel
    from src.core.exercise_detector import ExerciseDetector
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    
    detector = get_detector(use_hybrid)
    
    # Init Writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    analyzer = get_analyzer(mode)
    ex_detector = ExerciseDetector() if mode == "Auto-Detect" else None
    exercise_name = mode
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame, _ = detector.find_pose(frame.copy(), draw=True)
        lm = detector.get_landmarks()
        
        analysis_data = {"state": "N/A", "feedback": "Searching..."}
        if lm:
            if analyzer is None and ex_detector:
                ex_detector.add_frame(lm)
                detected = ex_detector.detect()
                if detected:
                    exercise_name = detected
                    analyzer = get_analyzer(detected)
            
            if analyzer: analysis_data = analyzer.analyze(lm, width, height)
            
        # Draw UI
        draw_text_with_background(frame, f"Analysis: {exercise_name}", (10, 40))
        draw_text_with_background(frame, f"Reps: {analysis_data.get('rep_count', 0)}", (10, 80))
        draw_text_with_background(frame, f"Feedback: {analysis_data.get('feedback', '')}", (10, 120))
        
        if out.isOpened(): out.write(frame)
        frame_count += 1
        progress_bar.progress(min(1.0, frame_count/total_frames))
        
    cap.release(); out.release(); progress_bar.empty(); gc.collect()

def main():
    st.set_page_config(page_title="AI Fitness Coach", layout="wide")
    st.title("🏋️ AI Fitness Analysis Coach")
    
    if "webcam_recording" not in st.session_state:
        st.session_state.webcam_recording = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
    
    with st.sidebar:
        st.header("Settings")
        use_hybrid = st.checkbox("AI-Enhanced Detection", value=False)
        if st.button("Clear Cache"): st.cache_data.clear(); gc.collect()

    exercise_type = st.radio("Exercise:", ["Auto-Detect", "Squat", "Lunge", "Push-Up", "Deadlift", "Plank", "Bench Press", "Dips"], horizontal=True)
    tab1, tab2 = st.tabs(["📹 Upload", "🎥 Live"])
    
    with tab1:
        u_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
        if u_file:
            t_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_in.write(u_file.read()); t_in.close()
            st.video(t_in.name)
            if st.button(f'Analyze {exercise_type}'):
                t_out = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
                with st.spinner('Analyzing...'): process_video(t_in.name, t_out, exercise_type, use_hybrid)
                with st.spinner('Optimizing...'): 
                    final, err = reencode_video_for_browser(t_out)
                    if err: st.error(err)
                st.video(final)

    with tab2:
        if webrtc_streamer:
            webrtc_streamer(
                key="fitness",
                video_processor_factory=lambda: BaseVideoProcessor(exercise_type, st.session_state.webcam_recording, use_hybrid),
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            )

if __name__ == '__main__':
    try: main()
    except Exception as e: st.error(f"Error: {e}"); st.exception(e)
