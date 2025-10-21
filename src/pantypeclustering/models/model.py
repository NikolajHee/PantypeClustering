from typing import Any, Mapping, Union

import lightning
import torch
from pantypeclustering.models.distributions import ReparameterizedDiagonalGaussian
from torch import Tensor, nn
from torch.distributions import Bernoulli, Distribution


def reduce(x: Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)


class ModelVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Sequential,
        decoder: nn.Sequential,
        input_shape: tuple[int, int, int],  # (C,W,H)
        latent_features: int,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape

        self.register_buffer("prior_params", torch.zeros(torch.Size([1, 2 * latent_features])))

    def posterior(self, x: Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | mu(x), sigma(x))`"""

        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma = h_x.chunk(2, dim=-1)

        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size: int = 1) -> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])  # type: ignore
        mu, log_sigma = prior_params.chunk(2, dim=-1)  # type: ignore

        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)  # type: ignore

    def observation_model(self, z: Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_logits = self.decoder(z)
        px_logits = px_logits.view(-1, *self.input_shape)  # reshape the output
        return Bernoulli(logits=px_logits, validate_args=False)

    def forward(self, x: Tensor) -> dict[str, Tensor | Distribution]:
        """
        compute the posterior q(z|x) (encoder),
        sample z~q(z|x),
        and return the distribution p(x|z) (decoder)
        """

        # flatten the input
        x = x.view(x.size(0), -1)

        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)

        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))

        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {"px": px, "pz": pz, "qz": qz, "z": z}

    def sample_from_prior(self, batch_size: int = 100) -> dict[str, Tensor | Distribution]:
        """sample z~p(z) and return p(x|z)"""

        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)

        # sample the prior
        z = pz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return {"px": px, "pz": pz, "z": z}


class VariationalAutoencoder(lightning.LightningModule):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_theta(x | z) = B(x | g_theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_phi(z|x) = N(z | mu(x), sigma(x))`
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, int, int],
        beta: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_shape = input_shape
        self.beta = beta

    def training_step(
        self,
        train_batch: Tensor,
        batch_idx: int,
    ) -> Union[Tensor, Mapping[str, Any]]:
        # forward pass through the model
        outputs = self.model.forward(train_batch[0])

        # unpack outputs
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]

        log_px = reduce(px.log_prob(train_batch[0]))
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        kl = log_qz - log_pz

        beta_elbo = (log_px) - (self.beta * kl)

        # loss
        loss = -beta_elbo.mean()

        self.log("loss", loss, prog_bar=True)
        if batch_idx == 0:
            reconstructed_images = px.sample()
            reconstructed_images = reconstructed_images.view(-1, *self.input_shape)

            original_images = train_batch[0]
            original_images = original_images.view(-1, *self.input_shape)

            self.logger.experiment.add_images(  # type: ignore
                "original_images",
                original_images,
                self.global_step,
            )
            self.logger.experiment.add_images(  # type: ignore
                "reconstructed_images",
                reconstructed_images,
                self.global_step,
            )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        outputs = self.model.forward(batch[0])

        # unpack outputs
        px, _, _, _ = [outputs[k] for k in ["px", "pz", "qz", "z"]]

        reconstructed_images = px.sample()
        reconstructed_images = reconstructed_images.view(-1, *self.input_shape)

        original_images = batch[0]
        original_images = original_images.view(-1, *self.input_shape)

        self.logger.experiment.add_images(  # type: ignore
            f"val_original_images_{batch_idx}",
            original_images,
            self.global_step,
        )
        self.logger.experiment.add_images(  # type: ignore
            f"val_reconstructed_images_{batch_idx}",
            reconstructed_images,
            self.global_step,
        )
