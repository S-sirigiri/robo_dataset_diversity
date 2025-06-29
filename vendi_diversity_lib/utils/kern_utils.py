import cupy as cp

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
        Compute a heuristic bandwidth for kernel operations based on the data X.

        The bandwidth is calculated as the median of the values in X divided by
        the natural logarithm of the number of samples, with a small constant
        added for numerical stability.

        Args:
            X (cp.ndarray): A 1D array of sample values for which to compute the bandwidth.

        Returns:
            float: The computed bandwidth value.
        """
        X = cp.asarray(X)
        return (
                cp.median(X) / cp.log(X.shape[0]) + 1e-3
        )