import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.utils import calculate_angle

class TestUtils(unittest.TestCase):
    def test_calculate_angle_90_deg(self):
        # A(0, -1), B(0, 0), C(1, 0)
        # Vector BA = (0, -1) -> -90 deg
        # Vector BC = (1, 0) -> 0 deg
        # Diff = 90.
        angle = calculate_angle((0, -1), (0, 0), (1, 0))
        self.assertAlmostEqual(angle, 90.0)

    def test_calculate_angle_180_deg(self):
        # A(0, -1), B(0, 0), C(0, 1)
        # BA -> -90
        # BC -> 90
        # Diff -> 180
        angle = calculate_angle((0, -1), (0, 0), (0, 1))
        self.assertAlmostEqual(angle, 180.0)

    def test_calculate_angle_straight_line(self):
        # A(-1, 0), B(0, 0), C(1, 0)
        # BA -> 180
        # BC -> 0
        # Diff -> 180
        angle = calculate_angle((-1, 0), (0, 0), (1, 0))
        self.assertAlmostEqual(angle, 180.0)

    def test_calculate_angle_acute(self):
        # A(1, 1), B(0, 0), C(1, 0)
        # BA -> 45
        # BC -> 0
        # Diff -> 45
        angle = calculate_angle((1, 1), (0, 0), (1, 0))
        self.assertAlmostEqual(angle, 45.0)

    def test_calculate_angle_reflex(self):
        # Does it handle reflex angles?
        # The function seems to return value <= 180.
        # if angle > 180: angle = 360 - angle

        # A(1, 1), B(0, 0), C(1, -1)
        # BA -> 45
        # BC -> -45 (315)
        # Diff -> -90 (abs 90)
        angle = calculate_angle((1, 1), (0, 0), (1, -1))
        self.assertAlmostEqual(angle, 90.0)

if __name__ == '__main__':
    unittest.main()
