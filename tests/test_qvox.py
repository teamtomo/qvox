test_array = np.zeros((2,3,4))
test_array[0,0,:] = 1
test_array[1,:,3] = 2



def test_resample():
    """
    Tests the resample function from the qvox.sampling module.
    """
    import numpy as np
    from qvox.sampling import resample
    expected_output = np.zeroes((4,6,8))
    expected_output[0:2,0:2,:] = 1
    expected_output[2:4,:,6:8] = 2
    output = resample(test_array, 2, 1)
    assert(output.shape == expected_output.shape)
    assert(np.allclose(output, expected_output))




