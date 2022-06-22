# Import libraries
from abc import ABC, abstractmethod
import multiprocessing
import numpy as np


class Eigen(ABC):
    """Define structure for methods that compute eigen values and eigen functions."""

    @abstractmethod
    def compute_eigen_from_kernel(self, cov_funct, size: int, int_lims: list) -> dict:
        """Compute eigen from a given covariance function."""


class IntegralMethod(Eigen):
    """Study of eigen from Fredholm integral equation by discretization."""

    def __init__(self, scheme: str = "uniform", tol: float = 1.e-12):

        self.tol = tol
        self.scheme = self._get_scheme(scheme)

    def _get_scheme(self, scheme: str):
        """Get Scheme class given an identifier string."""

        if scheme == "uniform":
            return Uniform()

        elif scheme == "trapezium":
            return Trapezium()

        else:
            raise Exception("Scheme: " + scheme + " not found.\n")

    def _evaluate_cov_funct_multiprocessing(self, cov_funct_, seq: np.ndarray) -> np.ndarray:
        """Evaluate the covariance function in an uniform grid."""

        with multiprocessing.Pool() as pool:
            return np.array(pool.starmap(cov_funct_.compute,
                                         [(s, t) for s in seq \
                                          for t in seq])).reshape(len(seq), -1)

    def _compute_eigen_numpy(self, cov_matrix: np.ndarray, seq: np.ndarray) -> np.ndarray:
        """Compute eigen from COV*Weights"""

        vaps, veps = np.linalg.eig(np.sqrt(self.scheme.get_weights(seq)).dot(\
                                   cov_matrix).dot(np.sqrt(self.scheme.get_weights(seq))))
        return np.real(vaps)[np.imag(vaps) < self.tol], np.real(veps)[:, np.imag(vaps) < self.tol]

    def compute_eigen_from_kernel(self, cov_funct, size: int, int_lims: list) -> dict:
        """Compute non-zero eigen values and eigen functions from cov-function."""

        seq = np.linspace(int_lims[0], int_lims[1], size)
        cov_matrix = self._evaluate_cov_funct_multiprocessing(cov_funct, seq)
        return self.compute_eigen_from_matrix(cov_matrix, seq)

    def compute_eigen_from_matrix(self, cov_matrix: np.ndarray, seq: np.ndarray) -> dict:
        """Compute no-zero eigen values and their corresponding eigen functions from cov-matrix."""

        vals, functs = self._compute_eigen_numpy(cov_matrix, seq)
        sel_indxs = vals > self.tol
        return {'time_line': seq,
                'eigen_values': vals[sel_indxs],
                'eigen_functs': self.scheme.get_coeffs(seq).dot(functs[:, sel_indxs])}


class SchemeIM(ABC):
    """Structure to get weights matrix and its inverse matrix."""

    @abstractmethod
    def get_weights(self, seq: np.ndarray) -> np.ndarray:
        """Return W."""

    @abstractmethod
    def get_coeffs(self, seq: np.ndarray) -> np.ndarray:
        """Return (W)^(-1./2)"""


class Uniform(SchemeIM):
    """Uniform scheme."""

    def get_weights(self, seq: np.ndarray) -> np.ndarray:
        """Compute weights matrix."""

        size, int_length = len(seq), seq[-1] - seq[0]
        return np.identity(size)*(int_length)/size

    def get_coeffs(self, seq: np.ndarray) -> np.ndarray:
        """compute coefficients for eigen functions."""

        size, int_length = len(seq), seq[-1] - seq[0]
        return np.identity(size)*np.sqrt(size/(int_length))


class Trapezium(SchemeIM):
    """Trapezium scheme."""

    def get_weights(self, seq: np.ndarray) -> np.ndarray:
        """Compute weights matrix."""

        size, int_length = len(seq), seq[-1] - seq[0]
        weights = np.identity(size)*(int_length)/(size - 1)
        weights[0, 0] = (int_length)/(2*(size - 1))
        weights[-1, -1] = (int_length)/(2*(size - 1))
        return weights

    def get_coeffs(self, seq: np.ndarray) -> np.ndarray:
        """compute coefficients for eigen functions."""

        size, int_length = len(seq), seq[-1] - seq[0]
        coeffs = np.identity(size)*(size - 1)/(int_length)
        coeffs[0, 0] = (2*(size - 1))/(int_length)
        coeffs[-1, -1] = (2*(size - 1))/(int_length)
        return np.sqrt(coeffs)


class HaarMethod(Eigen):
    """Haar expansion method applied to eigen for Fredholm integral equation."""

    def __init__(self, tol: float = 1e-12):

        self.tol = tol

    def _generic_haar_wavelet(self, j: int, k: int, x: float) -> float:
        """Evaluate the haar wavelet functions."""

        # Return wavelet function value
        if k*2**(-j) < x < k*2**(-j) + 2**(-j - 1):
            return 1.
        elif k*2**(-j) + 2**(-j - 1) <= x < k*2**(-j) + 2**(-j):
            return -1.
        else:
            return 0.

    def _get_wavelet_basis(self, size_power: int):
        """Build a discrete wavelet basis in a specific time line and
            compute the theorical inner products matrix."""

        # Retrieve size
        size = 2**size_power

        # Set Haar rescaled time line array to [0,1]
        time_line = (2*np.arange(size) + 1)/(2*size)

        # Compute raw wavelet indices
        wavelet_indices = [(j, k) for j in range(size_power) for k in range(2**j)]

        # Compute the matrix with the dot products between basis elements
        basis_norm_matrix = np.diag(2**(-np.array([(0., 0.)] + wavelet_indices)[:, 0]))

        # Compute wavelet index and evaluating point
        wavelet_input = [(j, k, x) for j, k in wavelet_indices for x in time_line]

        # Compute the discret haar basis
        with multiprocessing.Pool() as pool:
            haar_discrete_basis = np.array([1]*size +
                                           pool.starmap(self._generic_haar_wavelet,
                                                        wavelet_input)).reshape(-1, size)
        return time_line, basis_norm_matrix, haar_discrete_basis

    def _evaluate_cov_funct_multiprocessing(self, cov_funct_, seq: np.ndarray) -> np.ndarray:
        """Evaluate the covariance function in an seq grid."""

        with multiprocessing.Pool() as pool:
            return np.array(pool.starmap(cov_funct_.compute,
                                         [(s, t) for s in seq \
                                             for t in seq])).reshape(len(seq), -1)

    def _compute_eigen_numpy(self, matrix: np.ndarray) -> tuple:
        """Compute the haar expansion basis coeficients."""

        vaps, veps = np.linalg.eig(matrix)
        return np.real(vaps)[np.imag(vaps) < self.tol], np.real(veps)[:, np.imag(vaps) < self.tol]

    def compute_eigen_from_kernel(self, cov_funct, size: int, int_lims: list) -> dict:
        """Compute non-zero eigen values and eigen functions from cov-function."""

        # size argument must be a power of 2
        assert size in [2**j for j in range(size)]

        # Get exponent n: size = 2**n
        size_power = int(np.log2(size))

        # Get discrete wavelet basis
        generic_time_line, generic_inner_products, phi_basis = self._get_wavelet_basis(size_power)
        phi_inv = np.linalg.inv(phi_basis)
        # Compute the coefficient matrix for the covariance function
        time_line = generic_time_line*(int_lims[1] - int_lims[0]) + int_lims[0]
        cov_matrix = self._evaluate_cov_funct_multiprocessing(cov_funct, time_line)

        # Compute the 2D wavelet transform of cov_matrix
        cov_transform = (phi_inv.T).dot(cov_matrix.dot(phi_inv))

        # Compute eigen values and vectors of cov_transform
        inner_products = (int_lims[1] - int_lims[0])*generic_inner_products
        eigen_matrix = np.sqrt(inner_products).dot(cov_transform).dot(np.sqrt(inner_products))
        values, functs_transform = self._compute_eigen_numpy(eigen_matrix)
        functs = (phi_basis.T).dot(np.diag(1/np.sqrt(np.diag(inner_products)))).dot(functs_transform)

        # Filter basis
        sel_indxs = values > self.tol
        return {'time_line': time_line,
                'eigen_values': values[sel_indxs],
                'eigen_functs': functs[:, sel_indxs]}
