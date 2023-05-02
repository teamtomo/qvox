"""
This module contains functions for performing morphological operations on numpy arrays of integer labels.

morphological_grow: Splits a numpy array of integer labels into constituent binary arrays, performs a morphological grow operation
using an optional user-designated binary structure, then recombines the arrays into the original quantized integer format
with the original ordering.

morphological_shrink: Splits a numpy array of integer labels into constituent binary arrays, performs a morphological shrink operation
using an optional user-designated binary structure, then recombines the arrays into the original quantized integer format
with the original ordering.
"""


import numpy as np
from .utils import split_quantized_array, combine_binary_arrays
from scipy import ndimage


def morphological_grow(quantized_array: np.ndarray, binary_structure: Optional[np.ndarray] = None, num_iterations: int = 1) -> np.ndarray:
    """
    Splits a numpy array of integer labels into constituent binary arrays, performs a morphological grow operation
    using an optional user-designated binary structure, then recombines the arrays into the original quantized integer format
    with the original ordering.

    Parameters
    ----------
    quantized_array : np.ndarray
        numpy array of integer labels
    binary_structure : Optional[np.ndarray]
        optional numpy array representing the binary structure to use for the morphological grow operation. If None, a default
        binary structure will be used.
    num_iterations : int
        optional integer representing the number of iterations to perform the morphological grow operation. Default is 1.

    Returns
    -------
    np.ndarray
        numpy array of integer labels with the same shape as the input array, but with the binary arrays grown using
        the specified binary structure
    """
    binary_arrays = split_quantized_array(quantized_array)
    grown_arrays = []
    for binary_array in binary_arrays:
        if binary_structure is None:
            binary_structure = ndimage.generate_binary_structure(2,1)
        grown_array = binary_array
        for i in range(num_iterations):
            grown_array = ndimage.binary_dilation(grown_array, structure=binary_structure).astype(int)
        grown_arrays.append(grown_array)
    recombined_array = combine_binary_arrays(grown_arrays)
    return recombined_array

def morphological_shrink(quantized_array: np.ndarray, binary_structure: Optional[np.ndarray] = None, num_iterations: int = 1) -> np.ndarray:
    """
    Splits a numpy array of integer labels into constituent binary arrays, performs a morphological shrink operation
    using an optional user-designated binary structure, then recombines the arrays into the original quantized integer format
    with the original ordering.

    Parameters
    ----------
    quantized_array : np.ndarray
        numpy array of integer labels
    binary_structure : Optional[np.ndarray]
        optional numpy array representing the binary structure to use for the morphological shrink operation. If None, a default
        binary structure will be used.
    num_iterations : int
        optional integer representing the number of iterations to perform the morphological shrink operation. Default is 1.

    Returns
    -------
    np.ndarray
        numpy array of integer labels with the same shape as the input array, but with the binary arrays shrunk using
        the specified binary structure
    """
    binary_arrays = split_quantized_array(quantized_array)
    shrunk_arrays = []
    for binary_array in binary_arrays:
        if binary_structure is None:
            binary_structure = ndimage.generate_binary_structure(2,1)
        shrunk_array = binary_array
        for i in range(num_iterations):
            shrunk_array = ndimage.binary_erosion(shrunk_array, structure=binary_structure).astype(int)
        shrunk_arrays.append(shrunk_array)
    recombined_array = combine_binary_arrays(shrunk_arrays)
    return recombined_array


