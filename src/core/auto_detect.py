import numpy as np


class AutoDetectResolver:
    """
    Stabilizes auto-detect by requiring sustained SigLIP consensus and
    expanded squat-vs-lunge guards before locking an exercise.
    """

    def __init__(self, siglip_consensus=2):
        self.siglip_consensus = siglip_consensus
        self.last_siglip_label = None
        self.consecutive_siglip_count = 0

    def reset(self):
        self.last_siglip_label = None
        self.consecutive_siglip_count = 0

    def record_siglip_hint(self, label):
        if not label:
            return
        if label == self.last_siglip_label:
            self.consecutive_siglip_count += 1
        else:
            self.last_siglip_label = label
            self.consecutive_siglip_count = 1

    def sustained_siglip(self):
        return self.consecutive_siglip_count >= self.siglip_consensus

    @staticmethod
    def _buffer_has_movement(buffer, min_rom=0.05):
        if len(buffer) < 10:
            return False
        hip_y = [l[23].y for l in buffer]
        return (max(hip_y) - min(hip_y)) > min_rom

    @staticmethod
    def is_lunge_pattern(buffer):
        if len(buffer) < 10:
            return False
        knee_diffs = [abs(l[25].y - l[26].y) for l in buffer]
        ankle_spread = [abs(l[27].x - l[28].x) for l in buffer]
        avg_knee_diff = float(np.mean(knee_diffs))
        max_knee_diff = float(max(knee_diffs))
        knee_diff_rom = float(max(knee_diffs) - min(knee_diffs))
        avg_ankle_spread = float(np.mean(ankle_spread))
        return (
            (avg_knee_diff > 0.04 or max_knee_diff > 0.08 or knee_diff_rom > 0.05)
            and avg_ankle_spread > 0.08
        )

    def resolve(self, exercise_detector, heuristic_detected, ai_hint):
        buffer = exercise_detector.buffer
        has_movement = self._buffer_has_movement(buffer)

        if heuristic_detected:
            if heuristic_detected == "Squat" and self.is_lunge_pattern(buffer):
                return "Lunge"
            return heuristic_detected

        if ai_hint:
            self.record_siglip_hint(ai_hint)
        else:
            return None

        if not self.sustained_siglip() or not has_movement:
            return None

        if ai_hint == "Squat" and self.is_lunge_pattern(buffer):
            return "Lunge"

        return ai_hint
