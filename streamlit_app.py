import streamlit as st
import cv2
import tempfile
import os
import time
from pose_detector import PoseDetector
from squat_analyzer import SquatAnalyzer
from utils import draw_text_with_background, get_landmark_pixel
import mediapipe as mp

def process_video(input_path, output_path):
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
    analyzer = SquatAnalyzer()
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Pose Detection
        frame, results = detector.find_pose(frame, draw=True)
        landmarks = detector.get_landmarks()
        
        # Analysis
        analysis_data = {}
        if landmarks:
             analysis_data = analyzer.analyze(landmarks, width, height)
             
             # Draw Angles
             if analysis_data.get('l_knee_angle', 0) > 0:
                l_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
                lk_pos = get_landmark_pixel(l_knee, width, height)
                cv2.putText(frame, f"{int(analysis_data['l_knee_angle'])}", lk_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                r_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
                rk_pos = get_landmark_pixel(r_knee, width, height)
                cv2.putText(frame, f"{int(analysis_data['r_knee_angle'])}", rk_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Overlay Info (Same as app.py but adapted)
        # Dashboard Background
        cv2.rectangle(frame, (0, 0), (400, 350), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(frame[0:350, 0:400], 0.7, frame[0:350, 0:400], 0.3, 0, frame[0:350, 0:400])
        
        if analysis_data:
            state_color = (0, 255, 255)
            if analysis_data['state'] == "BOTTOM": state_color = (0, 255, 0)
            draw_text_with_background(frame, f"State: {analysis_data['state']}", (10, 70), text_color=state_color)
            
            draw_text_with_background(frame, f"Reps: {analysis_data['rep_count']}", (10, 110), font_scale=1, thickness=2)
            
            c_reps = analysis_data.get('correct_reps', 0)
            i_reps = analysis_data.get('incorrect_reps', 0)
            cv2.putText(frame, f"Correct: {c_reps}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Incorrect: {i_reps}", (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            feedback = analysis_data.get('feedback', '')
            draw_text_with_background(frame, feedback, (10, 180), text_color=(0, 100, 255))
            
            advice = analysis_data.get('advice', '')
            if advice:
                draw_text_with_background(frame, f"Advice: {advice}", (10, 220), text_color=(255, 255, 0), bg_color=(0,0,0))

            if analysis_data.get('valgus_detected', False):
                draw_text_with_background(frame, "KNEE VALGUS!", (10, 280), text_color=(0, 0, 255), bg_color=(255, 255, 255))
            
            view_mode = analysis_data.get('view', 'Unknown')
            draw_text_with_background(frame, f"View: {view_mode}", (10, 320), text_color=(200, 200, 200))
            
            score = analysis_data.get('last_rep_score', 0)
            draw_text_with_background(frame, f"Last Score: {score}", (10, 250), 
                                      text_color=(0, 255, 0) if score > 80 else (0, 0, 255))

        out.write(frame)
        
        frame_count += 1
        if total_frames > 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.text("Processing complete!")

def main():
    st.set_page_config(page_title="AI Squat Coach", layout="wide")
    
    st.title("AI Squat Analysis Coach")
    st.markdown("""
    Upload a video of yourself performing squats. The AI will analyze your form, count reps, 
    and provide specific coaching feedback on depth, knee alignment, and more.
    """)
    
    uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        # Save uploaded file to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        tfile.close()
        
        st.video(tfile.name)
        
        if st.button('Analyze Squats'):
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            output_path = output_file.name
            output_file.close() # Close so opencv can write to it
            
            with st.spinner('Analyzing form... (this may take a moment)'):
                try:
                    process_video(tfile.name, output_path)
                    
                    st.success("Analysis Complete!")
                    st.header("Analyzed Video")
                    
                    # Re-open the file to display it
                    # Note: Convert to H.264 if needed for browser support, but mp4v often works locally
                    st.video(output_path)
                    
                    # Add Download Button
                    with open(output_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Analyzed Video",
                            data=file,
                            file_name="analyzed_squat.mp4",
                            mime="video/mp4"
                        )
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    # Cleanup
                    # os.unlink(tfile.name)
                    # os.unlink(output_path) 
                    pass

if __name__ == '__main__':
    main()
