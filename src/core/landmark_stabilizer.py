"""
Brief landmark hold on pose-detection misses to avoid FSM resets from dropout.
Does not EMA-smooth coordinates — analyzers apply their own angle smoothing.
"""


class LandmarkStabilizer:
    def __init__(self, hold_frames=8):
        self.hold_frames = hold_frames
        self.last_valid = None
        self.miss_count = 0

    def update(self, landmarks):
        if landmarks:
            self.last_valid = landmarks
            self.miss_count = 0
            return landmarks
        if self.last_valid is not None and self.miss_count < self.hold_frames:
            self.miss_count += 1
            return self.last_valid
        return None

    def reset(self):
        self.last_valid = None
        self.miss_count = 0
