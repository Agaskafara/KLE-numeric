import multiprocessing
import numpy as np


class KarhunenLoeve:
    """Given discret eigen values and eigen functions,
        the covariance matrix and the stochastic process are computed with KL expansion."""

    def __init__(self, time_line: np.ndarray, eigen_values: np.ndarray, eigen_functs: np.ndarray):

        # Time sequence
        self.time_line = time_line

        # Eigen values and functions
        self.eigen_values = eigen_values
        self.eigen_functs = eigen_functs

    def get_covariance(self) -> np.ndarray:
        """Compute the covariance matrix from eigen (Mercer Theorem)."""

        # Compute truncated Mercer expansion sum
        cov_list = ((self.eigen_functs*self.eigen_values).dot(self.eigen_functs.T)).reshape(1, -1)

        # Generate time grid for function evaluation
        s_list = np.repeat(self.time_line, len(self.time_line))[None]
        t_list = np.tile(self.time_line, len(self.time_line))[None]
        return np.concatenate([s_list, t_list, cov_list], axis=0).T

    def get_process_sample(self, randoms: str = "Gaussian") -> np.ndarray:
        """Compute a realization of the stochastic process with KLE."""

        if isinstance(randoms, str):
            if randoms == "Gaussian":
                # Compute random Gaussian realizations of expansion random variables
                random_variables = np.random.normal(0, 1, len(self.eigen_values))

                # Compute Karhunen-Loeve expansion
                return np.concatenate([self.time_line[None],
                                       self.eigen_functs.dot(\
                                        np.sqrt(self.eigen_values)*random_variables)[None]]).T
            else:
                raise Exception("Distribution:" + randoms + " not known.\n")

        if  isinstance(randoms, np.ndarray):
            # Check shape
            assert randoms.shape == self.eigen_values.shape

            # Compute Karhunen-Loeve expansion
            return np.concatenate([self.time_line[None],
                                   self.eigen_functs.dot(\
                                    np.sqrt(self.eigen_values)*randoms)[None]]).T

    def get_process_samples(self,
                            randoms="Gaussian",
                            sample_size: int = -1) -> list:
        """Compute multiple realizations of the stochastic process with KLE."""

        if isinstance(randoms, str):
            assert sample_size > 0
            with multiprocessing.Pool() as pool:
                return pool.map(self.get_process_sample, [randoms]*sample_size)

        if isinstance(randoms, list):
            with multiprocessing.Pool() as pool:
                return pool.map(self.get_process_sample, randoms)
