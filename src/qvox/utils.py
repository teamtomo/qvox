"""
This module contains utility functions for working with quantized arrays.
"""

import numpy as np
from typing import List, Tuple

def split_quantized_array(seg: np.ndarray) -> List[np.ndarray]:
    """
    Splits a semantic segmentation into a set of binary matrices. 
    Requires N times as much memory (total N*X*Y*Z*4 bytes for 32 bit representation), 
    so should be avoided when not necessary.
    
    Args:
    - seg: numpy array of shape (X, Y, Z) representing a semantic segmentation
    
    Returns:
    - binary_matrices: list of binary numpy arrays of shape (X, Y, Z) representing the binary masks of each class
    """
    binary_matrices = []
    for i in range(int(seg.max())):
        binary_matrices.append((seg == i).astype(np.uint8))
    return binary_matrices

def combine_binary_arrays(binary_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Combines a set of binary matrices into a semantic segmentation.
    
    Args:
    - binary_matrices: list of binary numpy arrays of shape (X, Y, Z) representing the binary masks of each class
    
    Returns:
    - combined_matrix: numpy array of shape (X, Y, Z) representing the semantic segmentation
    """
    combined_matrix = np.zeros_like(binary_matrices[0])
    for i, binary_matrix in enumerate(binary_matrices):
        combined_matrix[binary_matrix == 1] = i
    return combined_matrix


def swap_values(arr: np.ndarray, value_pairs: List[Tuple[int, int]]) -> np.ndarray:
    """
    Swaps the values in an n-d array according to a set of value pairs.
    
    Args:
    - arr: numpy array of shape (X, Y, Z, ...) representing an n-d array with integer values
    - value_pairs: list of tuples representing the pairs of values to be swapped
    
    Returns:
    - swapped_arr: numpy array of shape (X, Y, Z, ...) representing the n-d array with swapped values
    """
    swapped_arr = arr.copy()
    for pair in value_pairs:
        swapped_arr[arr == pair[0]] = pair[1]
        swapped_arr[arr == pair[1]] = pair[0]
    return swapped_arr

