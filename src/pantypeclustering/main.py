import os
import time

import lightning
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from pantypeclustering.models.model import ModelVAE, VariationalAutoencoder

curr_time = time.strftime("%H:%M:%S", time.localtime())

# define the models, evaluator and optimizer


def main() -> None:
    latent_dim = 10
    input_dim = 28 * 28
    beta = 1

    encoder = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=256),
        nn.LeakyReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.LeakyReLU(),
        nn.Linear(in_features=128, out_features=latent_dim * 2),
    )

    decoder = nn.Sequential(
        nn.Linear(in_features=latent_dim, out_features=128),
        nn.LeakyReLU(),
        nn.Linear(in_features=128, out_features=256),
        nn.LeakyReLU(),
        nn.Linear(in_features=256, out_features=input_dim),
    )

    train_dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    test_dataset = MNIST(os.getcwd(), download=True, transform=ToTensor(), train=False)

    train_dataloader = DataLoader(  # type: ignore
        train_dataset,
        batch_size=16,
        num_workers=9,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(  # type: ignore
        test_dataset,
        num_workers=9,
        persistent_workers=True,
    )

    print(f"len of train: {len(train_dataset)}")
    print(f"len of test: {len(test_dataset)}")

    input_shape = (1, 28, 28)

    model_vae = ModelVAE(encoder, decoder, input_shape=input_shape, latent_features=latent_dim)

    vae = VariationalAutoencoder(model_vae, input_shape=input_shape, beta=beta)

    max_epochs = 10

    logger = TensorBoardLogger("", version="Univariate" + curr_time)

    trainer = lightning.Trainer(
        limit_train_batches=500,
        limit_val_batches=10,
        max_epochs=max_epochs,
        logger=logger,
    )
    trainer.fit(
        model=vae,
        train_dataloaders=train_dataloader,  # pyright: ignore[reportUnknownArgumentType]
        val_dataloaders=test_dataloader,  # pyright: ignore[reportUnknownArgumentType]
    )
