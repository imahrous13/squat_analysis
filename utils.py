import numpy as np
import cv2

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points a, b, and c.
    The angle is calculated at point b.
    
    Args:
        a (tuple): (x, y) coordinates of the first point.
        b (tuple): (x, y) coordinates of the second point (vertex).
        c (tuple): (x, y) coordinates of the third point.
        
    Returns:
        float: Angle in degrees.
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def get_landmark_pixel(landmark, frame_width, frame_height):
    """
    Converts normalized landmark coordinates to pixel coordinates.
    
    Args:
        landmark: MediaPipe landmark object with .x and .y attributes.
        frame_width (int): Width of the image frame.
        frame_height (int): Height of the image frame.
        
    Returns:
        tuple: (x, y) pixel coordinates as integers.
    """
    return int(landmark.x * frame_width), int(landmark.y * frame_height)

def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale=0.6, text_color=(255, 255, 255), 
                            bg_color=(0, 0, 0), thickness=1, padding=5):
    """
    Draws text with a background rectangle for better visibility.
    """
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  bg_color, 
                  cv2.FILLED)
    
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
