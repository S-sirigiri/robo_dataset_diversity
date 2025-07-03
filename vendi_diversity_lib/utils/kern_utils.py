import cupy as cp
import numpy as np

class KernelUtilities:
    """
    A collection of utility functions for kernel-based operations using CuPy.
    """

    def __init__(self):
        # No initialization is needed for static utility methods.
        pass

    @staticmethod
    def compute_bandwidth(X):
        """
        Compute a heuristic bandwidth for kernel operations.

        For 3D inputs of shape (N, T, d), returns a single float:
            median(X) / log(N) + 1e-3

        For 4D inputs of shape (N, I, T, d), returns a 1D CuPy array of length I,
        where each entry is the bandwidth computed on X[:, i, :, :].

        Args:
            X (np.ndarray or cp.ndarray): A 3D or 4D array.

        Returns:
            float or cp.ndarray: If input is 3D, returns a Python float.
                                 If input is 4D, returns a 1D CuPy array of length I.
        """
        # ensure we're working with a CuPy array
        X = cp.asarray(X)

        if X.ndim == 3:
            # X.shape == (N, T, d)
            N = X.shape[0]
            bw = cp.median(X) / cp.log(N) + 1e-3
            return float(bw)

        elif X.ndim == 4:
            # X.shape == (N, I, T, d)
            N, I, T, d = X.shape
            # compute median over axes 0 (samples), 2 (time), and 3 (dims) for each of the I slices
            medians = cp.median(X, axis=(0, 2, 3))  # shape (I,)
            bws = medians / cp.log(N) + 1e-3  # shape (I,)
            return bws

        else:
            raise ValueError(
                f"compute_bandwidth requires a 3D or 4D array, got {X.ndim}D."
            )