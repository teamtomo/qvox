"""
This module contains utility functions for working with quantized arrays.
"""

import numpy as np
from typing import List, Tuple

def split_quantized_array(seg: np.ndarray) -> List[np.ndarray]:
    """
    Splits a semantic segmentation into a set of binary matrices. 
    Requires N times as much memory (total N*D*H*W*4 bytes for 32 bit representation), 
    so should be avoided when not necessary.
    
    Parameters
    ----------
    seg : numpy.ndarray
        numpy array of shape (D, H, W) representing a semantic segmentation
    
    Returns
    -------
    binary_matrices : list of numpy.ndarray
        list of binary numpy arrays of shape (D, H, W) representing the binary masks of each class
    """
    binary_matrices = []
    for i in range(int(seg.max())):
        binary_matrices.append((seg == i+1).astype(np.uint8))
    return binary_matrices

def combine_binary_arrays(binary_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Combines a set of binary matrices into a semantic segmentation.
    
    Parameters
    ----------
    binary_matrices : list of numpy.ndarray
        list of binary numpy arrays of shape (D, H, W) representing the binary masks of each class
    
    Returns
    -------
    combined_matrix : numpy.ndarray
        numpy array of shape (D, H, W) representing the semantic segmentation
    """
    combined_matrix = np.zeros_like(binary_matrices[0])
    for i, binary_matrix in enumerate(binary_matrices):
        combined_matrix[binary_matrix == 1] = i+1
    return combined_matrix


def reassign_values(arr: np.ndarray, value_pairs: dict[int, int]) -> np.ndarray:
    """
    Swaps the values in an n-d array according to a set of value pairs.
    
    Parameters
    ----------
    arr : numpy.ndarray
        numpy array of shape (D, H, W) representing an n-d array with integer values
    value_pairs : dict[int, int]
        dictionary representing the pairs of values to be swapped
    
    Returns
    -------
    swapped_arr : numpy.ndarray
        numpy array of shape (D, H, W) representing the n-d array with swapped values
    """
    swapped_arr = arr.copy()
    for inval, outval in value_pairs.items():
        swapped_arr[arr == inval] = outval
    return swapped_arr

