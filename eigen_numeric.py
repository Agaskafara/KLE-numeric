# Import libraries
from abc import ABC, abstractmethod
import numpy as np
import multiprocessing


class Eigen(ABC):
    """Define structure for methods that compute eigen values and eigen functions."""

    @abstractmethod
    def compute_eigen_from_kernel(self, cov_funct, size: int, int_lims: list) -> dict:
        pass

    @abstractmethod
    def compute_eigen_from_matrix(self, cov_matrix, seq: np.ndarray) -> dict:
        pass


# Integral method
class IntegralMethod(Eigen):
    """Solution to Fredholm integral equation by Integral discretization."""

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
                                         [(s,t) for s in seq \
                                             for t in seq])).reshape(len(seq), -1)
    
    def _compute_eigen_numpy(self, cov_matrix: np.ndarray, seq: np.ndarray) -> np.ndarray:
        """Compute eigen from COV*Weights"""

        return np.linalg.eig(np.sqrt(self.scheme.get_weights(seq)).dot(\
            cov_matrix).dot(np.sqrt(self.scheme.get_weights(seq))))
    
    def compute_eigen_from_kernel(self, cov_funct, size: int, int_lims: list) -> dict:
        """Compute no-zero eigen values and their corresponding eigen functions from cov-function."""

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
    """Structure to get weights matrix and its half-inversed matrix."""

    @abstractmethod
    def get_weights(self, seq: np.ndarray) -> np.ndarray:
        """Return W."""
        pass

    @abstractmethod
    def get_coeffs(self, seq: np.ndarray) -> np.ndarray:
        """Return (W)^(-1./2)"""
        pass

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
