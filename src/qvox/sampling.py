"""
Functions for resampling operations on numpy arrays of integers representing different classes of semantic segmentation.

Functions:
- resample: resamples a numpy array of integer labels to a specified voxel spacing
"""

import numpy as np
from .utils import split_quantized_array, combine_binary_arrays
from scipy.ndimage import zoom



def resample(quantized_array: np.ndarray, input_spacing: float, 
                                   output_spacing: float) -> np.ndarray:
    """
    Resamples a numpy array of integer labels to a specified voxel spacing, splits it into constituent binary arrays,
    resamples each array from input spacing to new output spacing, rebinarizes, then recombines the arrays into the
    original quantized integer format with the original ordering.

    Args:
    - quantized_array: numpy array of integer labels
    - input_spacing: tuple of floats representing the input voxel spacing
    - output_spacing: tuple of floats representing the desired output voxel spacing
    
    Returns:
    - numpy array of integer labels with the same shape as the input array, but resampled to the desired output
      spacing
    """
    binary_arrays = split_quantized_array(quantized_array)
    resampled_arrays = []
    for binary_array in binary_arrays:
        resampled_array = zoom(binary_array, np.divide(input_spacing, output_spacing), order=1)
        resampled_array = np.round(resampled_array).astype(int) # simple rounding - may replace eventually
        resampled_arrays.append(resampled_array)
    recombined_array = combine_binary_arrays(resampled_arrays)
    return recombined_array
