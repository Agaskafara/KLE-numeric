from abc import ABC, abstractmethod


class CovFunct(ABC):
    """Interface for covariance function."""

    @abstractmethod
    def compute(self, s: float, t: float):
        pass

class Brownian(CovFunct):
    """Covariance function for a Brownian process."""

    def compute(self, s: float, t: float):
        assert t >= 0 and s >= 0
        return min(s, t)