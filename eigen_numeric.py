# Import libraries
from abc import ABC, abstractmethod
import numpy as np
import multiprocessing


class Eigen(ABC):
    """Define structure for methods that compute eigen values and eigen functions."""

    @abstractmethod
    def compute_eigen(self, cov_funct) -> dict:
        pass


# Integral method
class IntegralMethod(Eigen):
    """Solution to Fredholm integral equation by Integral discretization."""

    def __init__(self, size: int, int_lims: list, scheme: str = "unif", tol: float = 1.e-12):

        self.n = size
        self.seq = np.linspace(int_lims[0], int_lims[1], self.n)
        self.tol = tol
        self.scheme = self._get_scheme(scheme)
        self.weights_matrix = self.scheme.get_weights(self.seq)
        self.coeffs_matrix = self.scheme.get_coeffs(self.seq)
    
    @classmethod
    def from_config(cls, params: dict, int_lims: list):
        return cls(params['discret_size'], int_lims, params['scheme'], params['tol'])
    
    def _get_scheme(self, scheme: str):
        """Get Scheme class given an identifier string."""

        if scheme == "unif":
            return Unif()

        elif scheme == "trapezium":
            return Trapezium()

        else:
            raise Exception("Scheme: " + scheme + " not found.\n")

    #def _evaluate_cov_funct(self, cov_funct_):
    #
    #    return np.array([ cov_funct_.compute(s, t) for s in self.seq \
    #        for t in self.seq]).reshape(self.n, -1)

    def _evaluate_cov_funct_multiprocessing(self, cov_funct_) -> np.ndarray:
        """Evaluate the covariance function in an uniform grid."""

        with multiprocessing.Pool() as pool:
            return np.array(pool.starmap(cov_funct_.compute,
                                         [(s,t) for s in self.seq \
                                             for t in self.seq])).reshape(self.n, -1)
    
    def _compute_eigen_numpy(self, cov_funct) -> np.ndarray:
        """Compute covariance matrix and eigen from COV*Weights"""

        cov_matrix = self._evaluate_cov_funct_multiprocessing(cov_funct)
        return np.linalg.eig(np.sqrt(self.weights_matrix).dot(cov_matrix).dot(np.sqrt(self.weights_matrix)))
    
    def compute_eigen(self, cov_funct) -> dict:
        """Compute no-zero eigen values and their corresponding eigen functions."""

        vals, functs = self._compute_eigen_numpy(cov_funct)
        sel_indxs = vals > self.tol
        return {'time_line': self.seq,
                'eigen_values': vals[sel_indxs],
                'eigen_functs': self.coeffs_matrix.dot(functs[:, sel_indxs])}


class SchemeIM(ABC):
    """Structure to get weights matrix and its half-inversed matrix."""

    @abstractmethod
    def get_weights(self, seq: np.ndarray) -> np.ndarray:
        """Return W."""
        pass

    @abstractmethod
    def get_coeffs(self: np.ndarray) -> np.ndarray:
        """Return (W)^(-1./2)"""
        pass

class Unif(SchemeIM):
    """Uniform scheme."""

    def get_weights(self, seq: np.ndarray) -> np.ndarray:
        """Compute weights matrix."""

        size = len(seq)
        return np.identity(size)*(seq[-1] - seq[0])/size
    
    def get_coeffs(self, seq: np.ndarray) -> np.ndarray:
        """compute coefficients for eigen functions."""

        size = len(seq)
        return np.identity(size)*np.sqrt(size/(seq[-1] - seq[0]))


class Trapezium(SchemeIM):
    """Trapezium scheme."""

    def get_weights(self, seq: np.ndarray) -> np.ndarray:
        """Compute weights matrix."""

        size = len(seq)
        weights = np.identity(size)*(seq[-1] - seq[0])/(size - 1)
        weights[0, 0] = (seq[-1] - seq[0])/(2*(size -1))
        weights[-1, -1] = (seq[-1] - seq[0])/(2*(size -1))
        return weights
    
    def get_coeffs(self, seq: np.ndarray) -> np.ndarray:
        """compute coefficients for eigen functions."""

        size = len(seq)
        coeffs = np.identity(size)*(size - 1)/(seq[-1] - seq[0])
        coeffs[0, 0] = (2*(size -1))/(seq[-1] - seq[0])
        coeffs[-1, -1] = (2*(size -1))/(seq[-1] - seq[0])
        return np.sqrt(coeffs)
