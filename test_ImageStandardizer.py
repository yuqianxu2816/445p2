"""
EECS 445 - Introduction to Machine Learning
Fall 2024  - Project 2
Test Image Standardizer
    Test the ImageStandardizer class's fit and transform functions.
    Usage: python test_ImageStandardizer.py
"""
import numpy as np
from dataset import ImageStandardizer
    
def test_image_standardizer():
    standardizer = ImageStandardizer()
    
    # Test Case 1
    X1 = np.array([[[1, 2, 3], [4, 5, 6]],
                   [[7, 8, 9], [10, 11, 12]]], dtype=np.float64)
    expected_mean1 = 6.5
    expected_std1 = 3.452052529534663
    
    standardizer.fit(X1)
    print(standardizer.image_mean, standardizer.image_std)
    assert np.allclose(standardizer.image_mean, expected_mean1), "Test case 1 failed on mean"
    assert np.allclose(standardizer.image_std, expected_std1), "Test case 1 failed on std"

    X1_transformed_expected = np.array(
        [[[-1.59325501, -1.30357228, -1.01388955],
        [-0.72420682, -0.43452409, -0.14484136]], 
        [[ 0.14484136,  0.43452409,  0.72420682],
        [ 1.01388955,  1.30357228,  1.59325501]]])
    X1_transformed_actual = standardizer.transform(X1)
    assert np.allclose(X1_transformed_actual, X1_transformed_expected), "Test case 1 failed on transform"
    
    # Test Case 2
    X2 = np.array([[[1], [3]], [[5], [7]]], dtype=np.float64)
    expected_mean2 = 4.0
    expected_std2 = 2.23606797749979
    
    standardizer.fit(X2)
    assert np.allclose(standardizer.image_mean, expected_mean2), "Test case 2 failed on mean"
    assert np.allclose(standardizer.image_std, expected_std2), "Test case 2 failed on std"

    X2_transformed_expected = np.array(
        [[[-1.34164079],
        [-0.4472136 ]],
        [[ 0.4472136 ],
        [ 1.34164079]]])
    X2_transformed_actual = standardizer.transform(X2)
    assert np.allclose(X2_transformed_actual, X2_transformed_expected), "Test case 2 failed on transform"
    
    # Test Case 3
    X3 = np.array([[[5, 5, 5], [5, 5, 5]],
                   [[5, 5, 5], [5, 5, 5]]], dtype=np.float64)
    expected_mean3 = 5.0
    expected_std3 = 0.0
    
    standardizer.fit(X3)
    assert np.allclose(standardizer.image_mean, expected_mean3), "Test case 3 failed on mean"
    assert np.allclose(standardizer.image_std, expected_std3), "Test case 3 failed on std"

# Run the tests
test_image_standardizer()
print("All tests passed!")