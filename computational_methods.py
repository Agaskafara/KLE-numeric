from abc import ABC, abstractmethod

import numpy as np
import multiprocessing


class Eigen(ABC):

    @abstractmethod
    def compute_eigen(self, cov_funct):
        pass


# Integral method
class IntegralMethod(Eigen):
    """Solution to Fredholm integral equation by Integral discretization."""

    def __init__(self, size: int, int_lims: list, scheme: str = "unif", tol: float = 1.e-12):

        self.n = size
        self.seq = np.linspace(int_lims[0], int_lims[1], self.n)
        self.scheme = scheme
        self.tol = tol
        self.weights_matrix = self._get_weights()
    
    @classmethod
    def from_config(cls, params: dict, int_lims: list):
        return cls(params['discret_size'], int_lims, params['scheme'], params['tol'])

    def _get_weights(self):

        if self.scheme == "unif":
            return np.identity(self.n)*(self.seq[-1] - self.seq[0])/self.n

    def _get_discret_coeffs(self):

        if self.scheme == "unif":
            return np.identity(self.n)*np.sqrt(self.n/(self.seq[-1] - self.seq[0]))

    def _evaluate_cov_funct(self, cov_funct_):

        return np.array([ cov_funct_.compute(s, t) for s in self.seq \
            for t in self.seq]).reshape(self.n, -1)

    def _evaluate_cov_funct_multiprocessing(self, cov_funct_):

        with multiprocessing.Pool() as pool:
            return np.array(pool.starmap(cov_funct_.compute,
                                         [(s,t) for s in self.seq \
                                             for t in self.seq])).reshape(self.n, -1)
    
    def _compute_eigen_numpy(self, cov_funct):

        cov_matrix = self._evaluate_cov_funct(cov_funct)
        return np.linalg.eig(cov_matrix.dot(self.weights_matrix))
    
    def compute_eigen(self, cov_funct):

        vals, functs = self._compute_eigen_numpy(cov_funct)
        sel_indxs = vals > self.tol
        return {'time_line': self.seq,
                'eigen_values': vals[sel_indxs],
                'eigen_functs': self._get_discret_coeffs().dot(functs[:, sel_indxs])}
