from abc import abstractmethod
from typing import Any, Mapping, Union

import lightning
import torch
from models.distributions import ReparameterizedDiagonalGaussian
from models.priors import BasePrior, MixtureOfGaussian
from models.utils import fig_to_image, tsne, tsne_plot
from torch import Tensor, nn
from torch.distributions import Bernoulli, Distribution


def reduce(x: Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)


class BaseModel(lightning.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        prior: BasePrior,
        input_shape: tuple[int, int, int],  # (C,W,H)
        latent_features: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.prior = prior

    @abstractmethod
    def posterior(self, x: Tensor) -> Distribution: ...

    @abstractmethod
    def observation_model(self, z: Tensor) -> Distribution: ...

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Distribution, Distribution, Tensor]: ...


class ModelVAE(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        prior: BasePrior,
        input_shape: tuple[int, int, int],  # (C,W,H)
        latent_features: int,
        batch_size: int,
    ) -> None:
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            input_shape=input_shape,
            latent_features=latent_features,
            prior=prior,
            batch_size=batch_size,
        )
        self.register_buffer("prior_params", torch.zeros(torch.Size([1, 2 * latent_features])))

    def posterior(self, x: Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | mu(x), sigma(x))`"""

        # compute the parameters of the posterior
        h_x = self.encoder(x)
        mu, log_sigma = h_x.chunk(2, dim=-1)

        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    # def prior(self, batch_size: int = 1) -> Distribution:
    #     """return the distribution `p(z)`"""
    #     prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:]
    #  # type: ignore
    #     mu, log_sigma = prior_params.chunk(2, dim=-1)  # type: ignore

    #     # return the distribution `p(z)`
    #     return ReparameterizedDiagonalGaussian(mu, log_sigma)  # type: ignore

    def observation_model(self, z: Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        px_logits = self.decoder(z)
        px_logits = px_logits.view(-1, *self.input_shape)  # reshape the output
        return Bernoulli(logits=px_logits, validate_args=False)

    def forward(self, x: Tensor) -> tuple[Distribution, Distribution, Tensor]:
        """
        compute the posterior q(z|x) (encoder),
        sample z~q(z|x),
        and return the distribution p(x|z) (decoder)
        """

        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)

        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)

        return (px, qz, z)


class VariationalAutoencoder(lightning.LightningModule):
    """A Variational Autoencoder with
    * a Bernoulli observation model `p_theta(x | z) = B(x | g_theta(z))`
    * a Gaussian prior `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_phi(z|x) = N(z | mu(x), sigma(x))`
    """

    def __init__(
        self,
        model: BaseModel,
        input_shape: tuple[int, int, int],
        beta: float,
        val_num_images: int = 5,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_shape = input_shape
        self.beta = beta
        self.val_num_images = val_num_images

        self.reset_save()

    def training_step(
        self,
        train_batch: Tensor,
        batch_idx: int,
    ) -> Union[Tensor, Mapping[str, Any]]:
        # forward pass through the model
        x, _ = train_batch

        # unpack outputs
        px, qz, z = self.model.forward(x)

        log_px = reduce(px.log_prob(x))
        log_pz = reduce(self.model.prior.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        kl = log_qz - log_pz

        beta_elbo = (log_px) - (self.beta * kl)

        # loss
        loss = -beta_elbo.mean()

        self.log("train_loss", loss, prog_bar=True)
        if batch_idx == 0:
            reconstructed_images = px.sample()
            reconstructed_images = reconstructed_images.view(-1, *self.input_shape)

            original_images = x.view(-1, *self.input_shape)

            train_images = torch.vstack((original_images, reconstructed_images))

            self.logger.experiment.add_images(  # type: ignore
                "train_images",
                train_images,
                self.global_step,
            )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        x, _ = batch

        px, qz, z = self.model.forward(x)

        log_px = reduce(px.log_prob(x))
        log_pz = reduce(self.model.prior.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        kl = log_qz - log_pz

        beta_elbo = (log_px) - (self.beta * kl)

        loss = -beta_elbo.mean()

        self.log("test_loss", loss, prog_bar=True)

        # Store latents and labels for TSNE visualization
        self.val_labels.append(batch[1].cpu())
        self.val_latents.append(z.detach().cpu())

        if batch_idx <= self.val_num_images:
            reconstructed_images = px.sample()
            reconstructed_images = reconstructed_images.view(-1, *self.input_shape)

            original_images = batch[0]
            original_images = original_images.view(-1, *self.input_shape)

            test_images = torch.vstack((original_images, reconstructed_images))

            self.logger.experiment.add_images(  # type: ignore
                f"test_images_{batch_idx}",
                test_images,
                self.global_step,
            )

        if type(self.model.prior) is MixtureOfGaussian:
            prototypes = self.model.prior.means
            means: list[Tensor] = []
            for i in range(prototypes.size(0)):
                px = self.model.observation_model(prototypes[i])
                means_image = px.sample()
                means.append(means_image)

            test_images = torch.vstack(means)
            self.logger.experiment.add_images(  # type: ignore
                "GaussianMixtureMeans",
                test_images,
                self.global_step,
            )

    def reset_save(self) -> None:
        self.val_latents: list[Tensor] = []
        self.val_labels: list[Tensor] = []

    def on_validation_epoch_start(self) -> None:
        self.reset_save()

    def on_validation_epoch_end(self) -> None:
        if len(self.val_latents) > 30:
            latents_np = torch.cat(self.val_latents, dim=0).numpy()
            labels_np = torch.cat(self.val_labels, dim=0).numpy()

            if type(self.model.prior) is MixtureOfGaussian:
                latents_np = torch.cat(
                    (
                        *self.val_latents,
                        self.model.prior.means.cpu(),
                    ),
                    dim=0,
                ).numpy()
                labels_np = torch.cat(
                    (
                        *self.val_labels,
                        *[torch.tensor([10]) for _ in range(10)],
                    ),
                    dim=0,
                ).numpy()

            tsne_results = tsne(latents_np)

            fig = tsne_plot(tsne_results, labels_np)

            tsne_result = fig_to_image(fig)

            self.logger.experiment.add_images(  # type: ignore
                "tsne",
                tsne_result[None,],
                self.global_step,
            )
