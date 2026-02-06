import unittest
import sys
import os
from unittest.mock import MagicMock

# Mock mediapipe before importing SquatAnalyzer
mock_mp = MagicMock()
sys.modules["mediapipe"] = mock_mp

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analyzers.squat_analyzer import SquatAnalyzer
from src.core.utils import calculate_angle

class MockLandmark:
    def __init__(self, x, y, visibility=0.9):
        self.x = x
        self.y = y
        self.visibility = visibility

class TestSquatAnalyzer(unittest.TestCase):
    def setUp(self):
        # We need to setup the mock solutions because SquatAnalyzer uses mp.solutions.pose in __init__
        mock_mp.solutions.pose = MagicMock()
        # Mock enum values
        mock_mp.solutions.pose.PoseLandmark.LEFT_HIP.value = 0
        mock_mp.solutions.pose.PoseLandmark.RIGHT_HIP.value = 1
        mock_mp.solutions.pose.PoseLandmark.LEFT_KNEE.value = 2
        mock_mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value = 3
        mock_mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value = 4
        mock_mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value = 5
        mock_mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value = 6
        mock_mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value = 7

        self.analyzer = SquatAnalyzer()

        # Create a basic set of landmarks for standing
        self.landmarks = {}
        for i in range(33):
            self.landmarks[i] = MockLandmark(0, 0, visibility=0.0)

    def set_pose(self, hip_y, knee_y, ankle_y, hip_x=0.5, knee_x=0.5, ankle_x=0.5):
        """Helper to set leg landmarks"""
        # Hips
        self.landmarks[0] = MockLandmark(hip_x, hip_y)
        self.landmarks[1] = MockLandmark(hip_x, hip_y)
        # Knees
        self.landmarks[2] = MockLandmark(knee_x, knee_y)
        self.landmarks[3] = MockLandmark(knee_x, knee_y)
        # Ankles
        self.landmarks[4] = MockLandmark(ankle_x, ankle_y)
        self.landmarks[5] = MockLandmark(ankle_x, ankle_y)
        # Shoulders (for view detection)
        self.landmarks[6] = MockLandmark(hip_x, hip_y - 0.2)
        self.landmarks[7] = MockLandmark(hip_x, hip_y - 0.2)

    def test_standing_state(self):
        # Standing straight: Hip (0.5, 0.4), Knee (0.5, 0.6), Ankle (0.5, 0.8)
        # Vertical alignment
        self.set_pose(hip_y=0.4, knee_y=0.6, ankle_y=0.8)

        result = self.analyzer.analyze(self.landmarks, 100, 100)
        # Should transition to STANDING
        self.assertEqual(result['state'], "STANDING")
        self.assertEqual(self.analyzer.state, "STANDING")

    def test_full_rep_flow(self):
        # 1. Start Standing
        self.set_pose(hip_y=0.4, knee_y=0.6, ankle_y=0.8)
        self.analyzer.analyze(self.landmarks, 100, 100)
        self.assertEqual(self.analyzer.state, "STANDING")

        # 2. Descend (Squatting)
        # Frame 1: Standing
        self.set_pose(hip_y=0.4, knee_y=0.6, ankle_y=0.8)
        self.analyzer.analyze(self.landmarks, 100, 100)

        # Frame 2: Descending
        # Hip(0.5, 0.4), Knee(0.7, 0.6), Ankle(0.5, 0.8)
        self.set_pose(hip_y=0.4, knee_y=0.6, ankle_y=0.8, knee_x=0.7)
        for _ in range(5):
            res = self.analyzer.analyze(self.landmarks, 100, 100)

        self.assertEqual(self.analyzer.state, "DESCENDING", f"State is {self.analyzer.state} but expected DESCENDING. Knee angle: {res['l_knee_angle']}")

        # Frame 3: Bottom
        # Increase knee x offset to simulate deeper squat
        self.set_pose(hip_y=0.5, knee_y=0.6, ankle_y=0.8, knee_x=0.8)
        for _ in range(5):
             res = self.analyzer.analyze(self.landmarks, 100, 100)

        self.assertEqual(self.analyzer.state, "BOTTOM", f"State is {self.analyzer.state} but expected BOTTOM. Knee angle: {res['l_knee_angle']}")

        # Frame 4: Ascending
        # Go back to descending pose but slightly up.
        # knee_x=0.7 gave 90 deg. We need > 140 deg.
        # knee_x=0.55 gives ~152 deg.
        self.set_pose(hip_y=0.4, knee_y=0.6, ankle_y=0.8, knee_x=0.55)
        res = self.analyzer.analyze(self.landmarks, 100, 100)
        self.assertEqual(self.analyzer.state, "ASCENDING", f"State is {self.analyzer.state} but expected ASCENDING. Knee angle: {res['l_knee_angle']}")

        # Frame 5: Standing (Complete Rep)

        # Need to simulate time passing for duration check in ASCENDING -> STANDING transition
        # The analyzer uses time.time(). We should mock it or just rely on the duration check logic.
        # "rep_duration > 1.0"
        # self.rep_start_time = current_time (set in STANDING -> DESCENDING)

        # We need to mock time.time()
        with unittest.mock.patch('time.time') as mock_time:
            # Start time
            mock_time.return_value = 1000.0

            # Reset analyzer to start fresh for this part
            self.analyzer = SquatAnalyzer()
            # Re-mock mp.solutions.pose on the new instance (it's actually class level import, so instance has mp_pose attr)
            # The instance uses self.mp_pose which is set in __init__ using mp.solutions.pose.
            # Since we mocked mp module, mp.solutions.pose is already our mock.
            # But we need to make sure the values are set.
            self.analyzer.mp_pose.PoseLandmark.LEFT_HIP.value = 0
            self.analyzer.mp_pose.PoseLandmark.RIGHT_HIP.value = 1
            self.analyzer.mp_pose.PoseLandmark.LEFT_KNEE.value = 2
            self.analyzer.mp_pose.PoseLandmark.RIGHT_KNEE.value = 3
            self.analyzer.mp_pose.PoseLandmark.LEFT_ANKLE.value = 4
            self.analyzer.mp_pose.PoseLandmark.RIGHT_ANKLE.value = 5
            self.analyzer.mp_pose.PoseLandmark.LEFT_SHOULDER.value = 6
            self.analyzer.mp_pose.PoseLandmark.RIGHT_SHOULDER.value = 7

            # Start Standing
            self.set_pose(hip_y=0.4, knee_y=0.6, ankle_y=0.8)
            self.analyzer.analyze(self.landmarks, 100, 100)

            # Start Descending
            mock_time.return_value = 1001.0
            self.set_pose(hip_y=0.4, knee_y=0.6, ankle_y=0.8, knee_x=0.7)
            for _ in range(5):
                self.analyzer.analyze(self.landmarks, 100, 100)

            # Bottom
            mock_time.return_value = 1002.0
            self.set_pose(hip_y=0.5, knee_y=0.6, ankle_y=0.8, knee_x=0.8)
            for _ in range(5):
                self.analyzer.analyze(self.landmarks, 100, 100)

            # Ascending
            mock_time.return_value = 1003.0
            self.set_pose(hip_y=0.4, knee_y=0.6, ankle_y=0.8, knee_x=0.55)
            self.analyzer.analyze(self.landmarks, 100, 100)

            # Standing again (Finish Rep)
            mock_time.return_value = 1004.0 # > 1s duration
            self.set_pose(hip_y=0.4, knee_y=0.6, ankle_y=0.8, knee_x=0.5)
            for _ in range(5):
                res = self.analyzer.analyze(self.landmarks, 100, 100)

            self.assertEqual(self.analyzer.state, "STANDING")
            self.assertEqual(self.analyzer.rep_count, 1)

    def test_resolution_independence(self):
        # This test checks that the analyzer is resolution independent.
        # It uses relative tolerance instead of hardcoded pixels.

        knee_x = 0.55 # Should be around 152 deg

        # Setup DESCENDING state
        self.analyzer.state = "DESCENDING"

        # 1. Low Resolution (100x100)
        # Relative coords: Hip Y 0.5, Knee Y 0.6. Diff 0.1.
        # Tolerance 0.2 * 100 = 20px.
        # 50 > 60 - 20 (40). True.
        # Should detect BOTTOM.
        self.set_pose(hip_y=0.5, knee_y=0.6, ankle_y=0.8, knee_x=knee_x)

        # Reset state counter
        self.analyzer.state_counter = 0

        # Run analyze
        for _ in range(5):
            res_low = self.analyzer.analyze(self.landmarks, 100, 100)

        state_low = res_low['state']

        # 2. High Resolution (2000x2000)
        # Reset analyzer
        self.analyzer = SquatAnalyzer()
        # Remock
        self.analyzer.mp_pose.PoseLandmark.LEFT_HIP.value = 0
        self.analyzer.mp_pose.PoseLandmark.RIGHT_HIP.value = 1
        self.analyzer.mp_pose.PoseLandmark.LEFT_KNEE.value = 2
        self.analyzer.mp_pose.PoseLandmark.RIGHT_KNEE.value = 3
        self.analyzer.mp_pose.PoseLandmark.LEFT_ANKLE.value = 4
        self.analyzer.mp_pose.PoseLandmark.RIGHT_ANKLE.value = 5
        self.analyzer.mp_pose.PoseLandmark.LEFT_SHOULDER.value = 6
        self.analyzer.mp_pose.PoseLandmark.RIGHT_SHOULDER.value = 7

        self.analyzer.state = "DESCENDING"
        self.set_pose(hip_y=0.5, knee_y=0.6, ankle_y=0.8, knee_x=knee_x)

        # In 2000px: Diff is 200px.
        # Tolerance 0.2 * 2000 = 400px.
        # 1000 > 1200 - 400 (800). True.
        # Should detect BOTTOM.

        for _ in range(5):
            res_high = self.analyzer.analyze(self.landmarks, 2000, 2000)

        state_high = res_high['state']

        # They should be the same (both BOTTOM)
        self.assertEqual(state_low, state_high, "States should be same regardless of resolution")
        self.assertEqual(state_low, "BOTTOM")

if __name__ == '__main__':
    unittest.main()
