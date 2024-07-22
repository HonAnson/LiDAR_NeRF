import unittest
import numpy as np

# Import the cart2sph function
from preprocess import cart2sph

class TestCart2Sph(unittest.TestCase):
    def setUp(self):
        # Common setup for test cases
        self.pose_position = np.array([1.0, 2.0, 3.0])
        
    def test_single_point(self):
        pcd_array = np.array([[4.0, 5.0, 6.0]])
        expected_output = np.array([[5.196152423, 0.615479708, 0.78539816]])
        
        result = cart2sph(pcd_array, self.pose_position)
        
        np.testing.assert_almost_equal(result, expected_output, decimal=6)
    
    def test_multiple_points(self):
        pcd_array = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        expected_output = np.array([
            [5.196152423, 0.615479708, 0.78539816],
            [10.39230485, 0.615479708, 0.78539816]
        ])
        
        result = cart2sph(pcd_array, self.pose_position)
        
        np.testing.assert_almost_equal(result, expected_output, decimal=6)
    
    def test_zero_pose_position(self):
        pcd_array = np.array([[1.0, 2.0, 3.0]])
        expected_output = np.array([[3.74165739, 0.93027402, 1.10714872]])
        
        result = cart2sph(pcd_array, np.array([0.0, 0.0, 0.0]))
        
        np.testing.assert_almost_equal(result, expected_output, decimal=6)
    
    def test_origin_point(self):
        pcd_array = np.array([[1.0, 2.0, 3.0]])
        expected_output = np.array([[0.0, 0.0, 0.0]])
        result = cart2sph(pcd_array, np.array([1.0, 2.0, 3.0]))
        
        np.testing.assert_almost_equal(result, expected_output, decimal=6)

if __name__ == '__main__':
    unittest.main()
