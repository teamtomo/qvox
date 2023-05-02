import numpy as np

test_array = np.zeros((2,3,4))
test_array[0,0,:] = 1
test_array[1,:,3] = 2


def test_rescale():
    """
    Tests the resample function from the qvox.sampling module.
    """
    from qvox.sampling import rescale
    # Generate expected output for rescale
    expected_output = np.zeros((4,6,8))
    expected_output[0:2,0,:] = 1
    expected_output[0,1,:] = 1
    expected_output[3,:,6:8] = 2
    expected_output[2,:,7] = 2
    output= rescale(test_array, 2, 1)
    assert(output.shape == expected_output.shape)
    assert(np.allclose(output, expected_output))

def test_split_recombine():
    """
    Tests the split_quantized_array and combine_binary_arrays functions from the qvox.utils module.
    """
    from qvox.utils import split_quantized_array, combine_binary_arrays
    binary_arrays = split_quantized_array(test_array)
    expected_output_2 = np.zeros((2,3,4))
    expected_output_2[1,:,3] = 1
    assert(np.allclose(binary_arrays[1], expected_output_2))
    binary_arrays_combined = combine_binary_arrays(binary_arrays)
    assert(np.allclose(binary_arrays_combined, test_array))
    return True

def test_reassign():
    """
    Tests the reorder_values function from the qvox.utils module.
    """
    from qvox.utils import reassign_values
    expected_output = np.zeros((2,3,4))
    expected_output[0,0,:] = 2
    expected_output[1,:,3] = 3
    output = reorder_values(test_array,{1:2,2:3})
    assert(output.shape == expected_output.shape)
    assert(np.allclose(output, expected_output))

