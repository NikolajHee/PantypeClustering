import torch
from torch.distributions import Distribution


class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with
    the reparameterization trick given `epsilon ~ N(0, 1)`.
    """

    def __init__(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        assert mu.shape == log_sigma.shape
        f"torch.Tensors `mu` : {mu.shape} and"
        f"`log_sigma` : {log_sigma.shape} must be of the same shape"

        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> torch.Tensor:
        """`eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()

    def sample(
        self,
    ) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick)"""
        return self.mu + self.sigma * self.sample_epsilon()  # <- your code

    def log_prob(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """return the log probability: log `p(z)`"""

        return torch.distributions.Normal(self.mu, self.sigma).log_prob(z)
