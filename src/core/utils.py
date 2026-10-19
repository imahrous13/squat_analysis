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

def lm_xyzv(landmarks, idx):
    """
    Extracts x, y, z, and visibility from landmarks.
    Handles:
    - MediaPipe Results object (has .landmark)
    - List of MediaPipe landmark objects
    - Numpy array (33, 4)
    """
    if hasattr(landmarks, "landmark"):
        lm = landmarks.landmark[idx]
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32), float(lm.visibility)
    
    # Check if landmarks is a list/tuple of objects
    try:
        lm = landmarks[idx]
        if hasattr(lm, "x"): # It's an object with .x, .y, .z, .visibility
            return np.array([lm.x, lm.y, lm.z], dtype=np.float32), float(lm.visibility)
        # Otherwise assume it's a numpy array access
        return np.array(landmarks[idx, 0:3], dtype=np.float32), float(landmarks[idx, 3])
    except:
        return np.zeros(3, dtype=np.float32), 0.0

def safe_norm(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / (n + eps), n

def angle_3pts(a, b, c):
    # angle ABC in degrees
    ba = a - b
    bc = c - b
    ba_u, _ = safe_norm(ba)
    bc_u, _ = safe_norm(bc)
    cosang = np.clip(np.dot(ba_u, bc_u), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def center_point(p1, p2):
    return (p1 + p2) / 2.0

def angle_with_vertical(vec):
    # vec: 3D vector in image-normalized space; y is down in MediaPipe normalized coords
    # vertical direction pointing "up": (0,-1,0)
    v, _ = safe_norm(vec)
    vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    cosang = np.clip(np.dot(v, vertical), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))  # 0 = vertical up, 90 = horizontal

def choose_side_by_visibility(landmarks, left_ids, right_ids):
    # returns "left" or "right" based on average visibility
    def get_vis(lms, idx):
        if hasattr(lms, "landmark"):
            return float(lms.landmark[idx].visibility)
        lm = lms[idx]
        if hasattr(lm, "visibility"):
            return float(lm.visibility)
        return float(lms[idx, 3])

    left_vis = np.mean([get_vis(landmarks, i) for i in left_ids])
    right_vis = np.mean([get_vis(landmarks, i) for i in right_ids])
    return "left" if left_vis >= right_vis else "right"


def ema_smooth(prev, new, alpha):
    """Exponential moving average. prev=None returns new unchanged."""
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev


def debounced_transition(counter, condition, threshold):
    """
    Increment counter while condition holds; reset when false.
    Returns (new_counter, triggered).
    """
    if condition:
        new_counter = counter + 1
        return new_counter, new_counter >= threshold
    return 0, False


class SustainedFaultTracker:
    """Fault must appear in enough of the last N frames before surfacing to UI."""

    def __init__(self, frames_required=5, window=10):
        self.frames_required = frames_required
        self.window = window
        self._history = []

    def update(self, fault_active):
        self._history.append(bool(fault_active))
        if len(self._history) > self.window:
            self._history.pop(0)
        return sum(self._history) >= self.frames_required

    def reset(self):
        self._history = []
