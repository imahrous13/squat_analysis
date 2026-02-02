import os
# CRITICAL: Set environment variables BEFORE any other imports
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MP_GPU_MODE'] = '0' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import cv2
import tempfile
import time
import mediapipe as mp
import numpy as np
import gc

# Lazy Loaders
def get_analyzer(mode):
    try:
        if mode == "Squat": from src.analyzers.squat_analyzer import SquatAnalyzer; return SquatAnalyzer()
        if mode == "Push-Up": from src.analyzers.pushup_analyzer import PushUpAnalyzer; return PushUpAnalyzer()
        if mode == "Bench Press": from src.analyzers.bench_press_analyzer import BenchPressAnalyzer; return BenchPressAnalyzer(variant="standard")
        if mode == "Seated Bench Press": from src.analyzers.bench_press_analyzer import BenchPressAnalyzer; return BenchPressAnalyzer(variant="seated")
        if mode == "Deadlift": from src.analyzers.deadlift_analyzer import DeadliftAnalyzer; return DeadliftAnalyzer()
        if mode == "Lunge": from src.analyzers.lunge_analyzer import LungeAnalyzer; return LungeAnalyzer()
        if mode == "Jumping Jacks": from src.analyzers.jumping_jacks_analyzer import JumpingJacksAnalyzer; return JumpingJacksAnalyzer()
        if mode == "Plank": from src.analyzers.plank_analyzer import PlankAnalyzer; return PlankAnalyzer()
        if mode == "Chest Fly": from src.analyzers.chest_fly_analyzer import ChestFlyAnalyzer; return ChestFlyAnalyzer(variant="standing")
        if mode == "Seated Chest Fly": from src.analyzers.chest_fly_analyzer import ChestFlyAnalyzer; return ChestFlyAnalyzer(variant="seated")
        if mode == "Dips": from src.analyzers.dips_analyzer import DipsAnalyzer; return DipsAnalyzer(variant="normal")
        if mode == "Seated Dips": from src.analyzers.dips_analyzer import DipsAnalyzer; return DipsAnalyzer(variant="seated")
    except: pass
    return None

def get_detector(use_hybrid=False):
    try:
        if use_hybrid:
            from src.core.hybrid_pose import HybridPoseEstimator
            return HybridPoseEstimator(model_path='yolov8n.pt')
        else:
            from src.core.movenet_pose import MoveNetEstimator
            return MoveNetEstimator()
    except: pass
    return None

def reencode_video_for_browser(input_path):
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    try:
        import av
        input_container = av.open(input_path)
        output_container = av.open(output_path, mode='w', format='mp4')
        in_stream = input_container.streams.video[0]
        out_stream = output_container.add_stream('libx264', rate=in_stream.base_rate)
        out_stream.width, out_stream.height = in_stream.width, in_stream.height
        out_stream.pix_fmt = 'yuv420p'
        out_stream.options = {'preset': 'ultrafast', 'crf': '28'}
        for frame in input_container.decode(video=0):
            for packet in out_stream.encode(frame): output_container.mux(packet)
        for packet in out_stream.encode(): output_container.mux(packet)
        input_container.close(); output_container.close()
        return output_path, None
    except Exception as e: return input_path, str(e)

def process_video(input_path, output_path, mode="Squat", use_hybrid=False):
    from src.core.utils import draw_text_with_background
    from src.core.exercise_detector import ExerciseDetector
    
    cap = cv2.VideoCapture(input_path)
    # FOR CLOUD: Downscale to keep memory usage low
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 640 / max(orig_w, orig_h) if max(orig_w, orig_h) > 640 else 1.0
    width, height = int(orig_w * scale), int(orig_h * scale)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    
    detector = get_detector(use_hybrid)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    analyzer = get_analyzer(mode)
    ex_detector = ExerciseDetector() if mode == "Auto-Detect" else None
    exercise_name = mode
    
    pbar = st.progress(0)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        f_count += 1
        # CLOUD: Skip every other frame to save CPU
        if f_count % 2 != 0: continue
        
        frame = cv2.resize(frame, (width, height))
        frame, _ = detector.find_pose(frame, draw=True)
        lm = detector.get_landmarks()
        
        analysis_data = {"state": "N/A", "feedback": "Processing..."}
        if lm:
            if analyzer is None and ex_detector:
                ex_detector.add_frame(lm)
                detected = ex_detector.detect()
                if detected: analyzer = get_analyzer(detected); exercise_name = detected
            if analyzer: analysis_data = analyzer.analyze(lm, width, height)
            
        draw_text_with_background(frame, f"Gym AI: {exercise_name}", (10, 30), font_scale=0.5)
        draw_text_with_background(frame, f"Reps: {analysis_data.get('rep_count', 0)}", (10, 60), font_scale=0.5)
        
        if out.isOpened(): out.write(frame)
        if f_count % 10 == 0: pbar.progress(min(1.0, f_count/total_f))
        
    cap.release(); out.release(); pbar.empty(); gc.collect()

def main():
    st.set_page_config(page_title="Gym AI Coach", layout="centered")
    st.title("🏋️ AI Fitness Analysis")
    
    exercise_type = st.radio("Exercise:", ["Auto-Detect", "Squat", "Lunge", "Push-Up", "Deadlift", "Plank", "Bench Press", "Dips"], horizontal=True)
    u_file = st.file_uploader("Upload Workout", type=["mp4", "mov"])
    
    if u_file:
        t_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        t_in.write(u_file.read()); t_in.close()
        st.video(t_in.name)
        if st.button("Start Analysis"):
            t_out = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
            with st.spinner('Analyzing (Server Managed)...'): 
                process_video(t_in.name, t_out, exercise_type, False) # Force non-hybrid for cloud
            with st.spinner('Optimizing...'): 
                final, err = reencode_video_for_browser(t_out)
            st.video(final)

if __name__ == '__main__':
    main()
