import cv2
import time
import argparse
import mediapipe as mp

# Internal imports
from src.core.movenet_pose import MoveNetEstimator as PoseDetector
from src.analyzers.squat_analyzer import SquatAnalyzer
from src.analyzers.pushup_analyzer import PushUpAnalyzer
from src.analyzers.deadlift_analyzer import DeadliftAnalyzer
from src.analyzers.lunge_analyzer import LungeAnalyzer
from src.analyzers.jumping_jacks_analyzer import JumpingJacksAnalyzer
from src.analyzers.plank_analyzer import PlankAnalyzer
from src.core.utils import draw_text_with_background, get_landmark_pixel

def run_app(exercise_type="Squat", source=0):
    """
    Runs the fitness analyzer as a standard OpenCV application.
    
    Args:
        exercise_type (str): Type of exercise to analyze.
        source: Webcam index (int) or path to video file (str).
    """
    cap = cv2.VideoCapture(source)
    detector = PoseDetector()
    
    analyzers = {
        "Squat": SquatAnalyzer,
        "Lunge": LungeAnalyzer,
        "Push-Up": PushUpAnalyzer,
        "Deadlift": DeadliftAnalyzer,
        "Jumping Jacks": JumpingJacksAnalyzer,
        "Plank": PlankAnalyzer
    }
    
    if exercise_type not in analyzers:
        print(f"Error: Exercise '{exercise_type}' not supported.")
        return
        
    analyzer = analyzers[exercise_type]()
    p_time = 0
    
    print(f"Starting {exercise_type} analysis. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        
        # Pose Detection
        frame, _ = detector.find_pose(frame, draw=True)
        landmarks = detector.get_landmarks()
        
        if landmarks:
            analysis_data = analyzer.analyze(landmarks, w, h)
            
            # Draw Overlay
            state_color = (0, 255, 0) if analysis_data.get('state') in ["BOTTOM", "LOCKOUT", "TOP_PLANK"] else (0, 255, 255)
            
            y_pos = 30
            draw_text_with_background(frame, f"Exercise: {exercise_type}", (10, y_pos))
            
            y_pos += 35
            draw_text_with_background(frame, f"Muscles: {analysis_data.get('target_muscles', 'N/A')}", (10, y_pos), font_scale=0.5, text_color=(255, 150, 0))
            
            y_pos += 40
            draw_text_with_background(frame, f"State: {analysis_data['state']}", (10, y_pos), text_color=state_color)
            
            y_pos += 40
            draw_text_with_background(frame, f"Total Reps: {analysis_data['rep_count']}", (10, y_pos), font_scale=0.8, thickness=2)
            
            y_pos += 35
            c_reps = analysis_data.get('correct_reps', 0)
            i_reps = analysis_data.get('incorrect_reps', 0)
            draw_text_with_background(frame, f"Correct: {c_reps}", (10, y_pos), font_scale=0.6, text_color=(0, 255, 0))
            draw_text_with_background(frame, f"Incorrect: {i_reps}", (160, y_pos), font_scale=0.6, text_color=(0, 0, 255))
            
            y_pos += 40
            draw_text_with_background(frame, f"Feedback: {analysis_data.get('feedback', '')}", (10, y_pos), text_color=(0, 100, 255))
            
            y_pos += 40
            advice = analysis_data.get('advice', '')
            if advice:
                draw_text_with_background(frame, f"Advice: {advice}", (10, y_pos), font_scale=0.5, text_color=(255, 200, 0))
                y_pos += 30
            
            # Display reasons if incorrect
            score = analysis_data.get('last_rep_score', 0)
            if score > 0 and score < 75 and analysis_data.get('reasons'):
                for r in analysis_data['reasons'][:2]:
                    draw_text_with_background(frame, f"Fault: {r}", (10, y_pos), text_color=(0, 0, 255), font_scale=0.5)
                    y_pos += 25

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time) if p_time > 0 else 0
        p_time = c_time
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("AI Fitness Coach (Local)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Fitness Coach Local App")
    parser.add_argument("--type", type=str, default="Squat", choices=["Squat", "Lunge", "Push-Up", "Deadlift", "Jumping Jacks", "Plank"], help="Exercise type")
    parser.add_argument("--source", type=str, default="0", help="Webcam index or video file path")
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit (webcam index)
    source = int(args.source) if args.source.isdigit() else args.source
    
    run_app(args.type, source)
