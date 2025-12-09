import cv2
import time
from pose_detector import PoseDetector
from squat_analyzer import SquatAnalyzer
from utils import draw_text_with_background, get_landmark_pixel
import mediapipe as mp

def main():
    # 1. Setup Video Capture
    # Try index 0 first, then 1 if failed. Or provide a video path.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set resolution (optional, for performance)
    cap.set(3, 1280)
    cap.set(4, 720)

    # 2. Initialize Modules
    detector = PoseDetector()
    analyzer = SquatAnalyzer()
    
    p_time = 0

    print("Starting Squat Analysis System...")
    print("Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame.")
            break

        # 3. Pose Detection
        frame, results = detector.find_pose(frame, draw=True)
        landmarks = detector.get_landmarks()
        
        # 4. Analysis
        analysis_data = {}
        if landmarks:
            h, w, c = frame.shape
            analysis_data = analyzer.analyze(landmarks, w, h)
            
            # Draw Angles on joints ONLY if we have valid data (check if feedback is not error)
            if analysis_data.get('l_knee_angle', 0) > 0:
                # Left Knee
                l_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
                lk_pos = get_landmark_pixel(l_knee, w, h)
                cv2.putText(frame, f"{int(analysis_data['l_knee_angle'])}", lk_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Right Knee
                r_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value]
                rk_pos = get_landmark_pixel(r_knee, w, h)
                cv2.putText(frame, f"{int(analysis_data['r_knee_angle'])}", rk_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
             # Handle no landmarks case explicitly if needed, though analyzer handles it now too
             pass

        # 5. UI / Overlay
        # FPS Calculation
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        
        # Dashboard Background
        cv2.rectangle(frame, (0, 0), (400, 350), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(frame[0:350, 0:400], 0.7, frame[0:350, 0:400], 0.3, 0, frame[0:350, 0:400]) # Semi-transparent
        
        # Display Data
        draw_text_with_background(frame, f"FPS: {int(fps)}", (10, 30), text_color=(0, 255, 0))
        
        if analysis_data:
            # State
            state_color = (0, 255, 255) # Yellow
            if analysis_data['state'] == "BOTTOM": state_color = (0, 255, 0) # Green
            draw_text_with_background(frame, f"State: {analysis_data['state']}", (10, 70), text_color=state_color)
            
            # Rep Count
            draw_text_with_background(frame, f"Reps: {analysis_data['rep_count']}", (10, 110), font_scale=1, thickness=2)
            
            # Correct/Incorrect Breakdown
            c_reps = analysis_data.get('correct_reps', 0)
            i_reps = analysis_data.get('incorrect_reps', 0)
            cv2.putText(frame, f"Correct: {c_reps}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Incorrect: {i_reps}", (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Feedback
            feedback = analysis_data.get('feedback', '')
            draw_text_with_background(frame, feedback, (10, 180), text_color=(0, 100, 255))
            
            # Advice (New)
            advice = analysis_data.get('advice', '')
            if advice:
                draw_text_with_background(frame, f"Advice: {advice}", (10, 220), text_color=(255, 255, 0), bg_color=(0,0,0))

            # Valgus Warning
            if analysis_data.get('valgus_detected', False):
                draw_text_with_background(frame, "KNEE VALGUS!", (10, 280), text_color=(0, 0, 255), bg_color=(255, 255, 255))
            
            # View Info
            view_mode = analysis_data.get('view', 'Unknown')
            draw_text_with_background(frame, f"View: {view_mode}", (10, 320), text_color=(200, 200, 200))

            # Last Rep Score
            score = analysis_data.get('last_rep_score', 0)
            draw_text_with_background(frame, f"Last Score: {score}", (10, 250), 
                                      text_color=(0, 255, 0) if score > 80 else (0, 0, 255))

        cv2.imshow("Squat Analysis System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
