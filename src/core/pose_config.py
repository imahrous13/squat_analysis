"""
Centralized MediaPipe Pose configuration for live video stability.

Defaults target ~15-30 FPS on mid hardware with smooth_landmarks=True.
Use mode='hybrid' when YOLO ROI cropping needs the same baseline; raise
min_detection_confidence to 0.65 only if ROI jitter causes false drops.
"""

DEFAULT_POSE_CONFIG = {
    "static_image_mode": False,
    "model_complexity": 1,
    "smooth_landmarks": True,
    "enable_segmentation": False,
    "min_detection_confidence": 0.6,
    "min_tracking_confidence": 0.7,
}

HYBRID_POSE_CONFIG = {
    **DEFAULT_POSE_CONFIG,
    # Hybrid runs pose on a cropped ROI; same smoothing avoids double-lag.
}


def get_pose_config(mode="default"):
    """Return a copy of MediaPipe Pose kwargs for the given mode."""
    if mode == "hybrid":
        return dict(HYBRID_POSE_CONFIG)
    return dict(DEFAULT_POSE_CONFIG)


def create_mediapipe_pose(mp_pose_module, mode="default", **overrides):
    """Build a MediaPipe Pose instance from centralized config."""
    config = get_pose_config(mode)
    config.update(overrides)
    return mp_pose_module.Pose(**config)
