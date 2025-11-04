import os
import time

import lightning
from lightning.pytorch.loggers import TensorBoardLogger
from models.model import ModelVAE, VariationalAutoencoder
from models.priors import MixtureOfGaussian
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

curr_time = time.strftime("%H:%M:%S", time.localtime())


# define the models, evaluator and optimizer
class CNNEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(in_features=64 * 7 * 7, out_features=2 * latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z = self.fc(z)
        return z


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.to_decoder = nn.Linear(in_features=latent_dim, out_features=64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        x = self.to_decoder(z)
        x = x.reshape(-1, 64, 7, 7)  # TODO: remove hardcoding?
        x = self.decoder(x)
        return x


def main() -> None:
    latent_dim = 12
    beta = 1
    batch_size = 16
    max_epochs = 20
    input_shape = (1, 28, 28)
    version = "MixtureGaussian"
    num_workers = 9

    num_train_batches = 3750
    num_test_batches = 1000

    train_dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    test_dataset = MNIST(os.getcwd(), download=True, transform=ToTensor(), train=False)

    train_dataloader = DataLoader(  # type: ignore
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(  # type: ignore
        test_dataset,
        num_workers=num_workers,
        persistent_workers=True,
    )

    print(f"len of train: {len(train_dataset)}")
    print(f"len of test: {len(test_dataset)}")

    prior = MixtureOfGaussian(
        latent_dim=latent_dim,
        num_clusters=10,
        batch_size=batch_size,
    )

    model_vae = ModelVAE(
        CNNEncoder(latent_dim),
        CNNDecoder(latent_dim),
        prior=prior,
        input_shape=input_shape,
        latent_features=latent_dim,
        batch_size=batch_size,
    )

    vae = VariationalAutoencoder(model_vae, input_shape=input_shape, beta=beta)

    logger = TensorBoardLogger("", version=version + curr_time)

    trainer = lightning.Trainer(
        limit_train_batches=num_train_batches,
        limit_val_batches=num_test_batches,
        max_epochs=max_epochs,
        logger=logger,
    )
    trainer.fit(
        model=vae,
        train_dataloaders=train_dataloader,  # pyright: ignore[reportUnknownArgumentType]
        val_dataloaders=test_dataloader,  # pyright: ignore[reportUnknownArgumentType]
    )


if __name__ == "__main__":
    main()
