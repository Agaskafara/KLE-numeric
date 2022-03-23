from abc import ABC, abstractmethod


class CovFunct(ABC):
    """Interface for covariance function."""

    @abstractmethod
    def get_prefix(self) -> str:
        """Get covariance function identifier."""
        pass

    @abstractmethod
    def compute(self, s: float, t: float) -> float:
        """Evaluate covariance function at given points."""
        pass

class Brownian(CovFunct):
    """Covariance function for a Brownian process."""

    def get_prefix(self) -> str:
        """Get Brownian identifier."""

        return 'brownian_'

    def compute(self, s: float, t: float) -> float:
        """Evaluate Brownian covariance function at given points."""

        assert t >= 0 and s >= 0
        return min(s, t)