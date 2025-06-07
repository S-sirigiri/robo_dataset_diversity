import cupy as cp
import numpy as np
from scipy.linalg import logm

from kernels import KernelMatrix


class _Entropy:
    """
    Provides entropy computations over a normalized kernel matrix M = K/n.

    Methods
    -------
    shannon(K)
        Compute classical Shannon entropy: H = -sum(lambda_i log(lambda_i)).
    von_neumann(K)
        Compute Von Neumann entropy: H = -tr(M log M).
    """

    def __init__(self, kernel_matrix: KernelMatrix):
        if not isinstance(kernel_matrix, KernelMatrix):
            raise TypeError("kernel_matrix must be a KernelMatrix instance")
        self.km = kernel_matrix

    def _shannon(self, K) -> float:
        # Ensure CuPy array and normalize
        K_cp = cp.asarray(K)
        n = K_cp.shape[0]
        M = K_cp / n
        # Eigenvalues, clamped to non-negative
        evals = cp.linalg.eigvalsh(M)
        evals = cp.clip(evals, 0, None)
        mask = evals > 0
        H = -cp.sum(evals[mask] * cp.log(evals[mask]))
        return float(H)

    def _von_neumann(self, K) -> float:
        # Ensure CuPy array and normalize
        K_cp = cp.asarray(K)
        n = K_cp.shape[0]
        M = K_cp / n
        # Move to CPU for matrix log
        M_cpu = cp.asnumpy(M)
        L = logm(M_cpu)
        H = -np.trace(M_cpu @ L)
        return float(H)

class ShannonEntropy(_Entropy):
    """
    Compute the classical Shannon entropy of a normalized kernel matrix.

    The Shannon entropy is defined as:
        H = -sum_i lambda_i * log(lambda_i)
    where lambda_i are the non-negative eigenvalues of M = K/n.
    """
    def __init__(self, kernel_matrix: KernelMatrix):
        super().__init__(kernel_matrix)

    def __call__(self, X, Y=None, diag=False) -> float:
        """
        Evaluate Shannon entropy on the kernel matrix for input data.

        Parameters
        ----------
        X : array-like
            First set of data points for the KernelMatrix.
        Y : array-like, optional
            Second set of data points for cross-kernel computation.
        diag : bool, default=False
            Ignored placeholder for API consistency.

        Returns
        -------
        float
            The computed Shannon entropy.
        """
        # Retrieve raw kernel matrix (NumPy or CuPy)
        K = self.km(X, Y, diag)
        # Delegate to the protected _shannon method
        return self._shannon(K)


class VonNeumannEntropy(_Entropy):
    """
    Compute the Von Neumann entropy of a normalized kernel matrix.

    The Von Neumann entropy is defined as:
        H = -tr(M log M)
    where M = K/n and log is the matrix logarithm.
    """
    def __init__(self, kernel_matrix: KernelMatrix):
        super().__init__(kernel_matrix)

    def __call__(self, X, Y=None, diag=False) -> float:
        """
        Evaluate Von Neumann entropy on the kernel matrix for input data.

        Parameters
        ----------
        X : array-like
            First set of data points for the KernelMatrix.
        Y : array-like, optional
            Second set of data points for cross-kernel computation.
        diag : bool, default=False
            Ignored placeholder for API consistency.

        Returns
        -------
        float
            The computed Von Neumann entropy.
        """
        # Retrieve raw kernel matrix (NumPy or CuPy)
        K = self.km(X, Y, diag)
        # Delegate to the protected _von_neumann method
        return self._von_neumann(K)


class LogDeterminant:
    """
    Compute the log-determinant of a kernel matrix.

    The log-determinant is given by:
        log_det = log(det(K))
    computed on the GPU via CuPy.
    """
    def __init__(self, kernel_matrix: KernelMatrix):
        if not isinstance(kernel_matrix, KernelMatrix):
            raise TypeError("kernel_matrix must be a KernelMatrix instance")
        self.km = kernel_matrix

    def __call__(self, X, Y=None, diag=False) -> float:
        """
        Evaluate log-determinant on the kernel matrix for input data.

        Parameters
        ----------
        X : array-like
            First set of data points for the KernelMatrix.
        Y : array-like, optional
            Second set of data points for cross-kernel computation.
        diag : bool, default=False
            Ignored placeholder for API consistency.

        Returns
        -------
        float
            The computed log-determinant.
        """
        # Retrieve raw kernel matrix (NumPy or CuPy)
        K = self.km(X, Y, diag)
        K = cp.asarray(K)
        # Compute determinant on GPU, then take logarithm
        det_val = cp.linalg.det(K)
        log_det = cp.log(det_val)
        return float(log_det)


class VendiScore(_Entropy):
    """
    Compute the Vendi Score using a provided KernelMatrix and an Entropy variant.

    The score is V = exp(H), where H is the chosen entropy of M=K/n.

    Parameters
    ----------
    kernel_matrix : KernelMatrix
        A configured KernelMatrix for similarity computation.
    method : {'shannon', 'von_neumann'}, default='shannon'
        Choose the entropy variant for Vendi Score.
    """

    def __init__(self, kernel_matrix: KernelMatrix, method: str = 'shannon'):
        super().__init__(kernel_matrix)
        if method not in ('shannon', 'von_neumann'):
            raise ValueError("method must be 'shannon' or 'von_neumann'")
        self.method = method

    def _score(self, K) -> float:
        # Compute raw entropy
        if self.method == 'shannon':
            H = self._shannon(K)
        else:
            H = self._von_neumann(K)
        # Exponentiate to get Vendi Score
        return float(np.exp(H))

    def __call__(self, X, Y=None, diag=False) -> float:
        """
        Compute Vendi Score for data via the kernel matrix.

        Parameters
        ----------
        X : array-like
            First batch of data.
        Y : array-like, optional
            Second batch (for cross-kernel).
        diag : bool, default=False
            Ignored.

        Returns
        -------
        float
            The Vendi Score.
        """
        K = self.km(X, Y, diag)
        return self._score(K)