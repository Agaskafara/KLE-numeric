from abc import ABC, abstractmethod
import numpy as np


class CovFunct(ABC):
    """Interface for covariance function."""

    @abstractmethod
    def get_prefix(self) -> str:
        """Get covariance function identifier."""

    @abstractmethod
    def compute(self, s: float, t: float) -> float:
        """Evaluate covariance function at given points."""


class WhiteNoise(CovFunct):
    """Covariance function for a WhiteNoise process."""

    def __init__(self, sigma: float = 1):
        self.sigma = sigma

    def get_prefix(self) -> str:
        """Get WhiteNoise identifier."""

        return 'white-noise_'

    def compute(self, s: float, t: float) -> float:
        """Evaluate white-noise covariance function at given points."""

        assert t >= 0 and s >= 0
        return (self.sigma**2 if s == t else 0)


class Brownian(CovFunct):
    """Covariance function for a Brownian process."""

    def __init__(self, sigma: float = 1):
        self.sigma = sigma

    def get_prefix(self) -> str:
        """Get Brownian identifier."""

        return 'brownian_'

    def compute(self, s: float, t: float) -> float:
        """Evaluate Brownian covariance function at given points."""

        assert t >= 0 and s >= 0
        return min(s, t)*self.sigma**2


class BrownianBridge(CovFunct):
    """Covariance function for a Brownian Bridge process."""

    def __init__(self, sigma: float = 1, length: float = 1):
        self.sigma = sigma
        self.length = length

    def get_prefix(self) -> str:
        """Get Brownian identifier."""

        return 'brownian_bridge_'

    def compute(self, s: float, t: float) -> float:
        """Evaluate Brownian Bridge covariance function at given points."""

        assert t >= 0 and s >= 0
        return (min(s, t) - s*t/self.length)*self.sigma**2


class OrnsteinUhlenbeck(CovFunct):
    """Covariance function for an Ornstein-Uhlenbeck process."""

    def __init__(self, beta: float, rho: float):

        assert beta > 0 and rho > 0
        self.beta = beta
        self.rho = rho

    def get_prefix(self) -> str:
        """Get Ornstein-Uhlenbeck identifier."""

        return 'Ornstein-Uhlenbeck_'

    def compute(self, s: float, t: float) -> float:
        """Evaluate Ornstein-Uhlenbeck covariance function at given points."""

        assert t >= 0 and s >= 0
        return np.exp(-self.beta*(abs(t - s)))*self.rho**2/(2*self.beta)


class Exponential(CovFunct):
    """Covariance function for an exponential process."""

    def __init__(self, sigma: float, length_scale: float):
        self.sigma = sigma
        self.length = length_scale

    def get_prefix(self) -> str:
        """Get exponential identifier."""

        return 'exponential_'

    def compute(self, s: float, t: float) -> float:
        """Evaluate exponential covariance function at given points."""

        assert t >= 0 and s >= 0
        return np.exp(-abs(s-t)/(2*self.length**2))*self.sigma**2


class SquaredExponential(CovFunct):
    """Covariance function for a squared exponential process."""

    def __init__(self, sigma: float, length_scale: float):
        self.sigma = sigma
        self.length = length_scale

    def get_prefix(self) -> str:
        """Get squared exponential identifier."""

        return 'squared-exponential_'

    def compute(self, s: float, t: float) -> float:
        """Evaluate squared exponential covariance function at given points."""

        assert t >= 0 and s >= 0
        return np.exp(-(s-t)**2/(2*self.length**2))*self.sigma**2
