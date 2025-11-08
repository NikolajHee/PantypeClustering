# define the models, evaluator and optimizer

from torch import Tensor, nn


class CNNEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(in_features=64 * 7 * 7, out_features=2 * latent_dim)

        # Initialize the final layer with smaller weights for stability
        # This helps prevent log_sigma from starting too large
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z = self.fc(z)
        return z


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.to_decoder = nn.Linear(in_features=latent_dim, out_features=64 * 7 * 7)
        final_conv = nn.ConvTranspose2d(32, 2 * 1, 4, 2, 1)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            final_conv,
        )

        # Initialize the final conv layer with smaller weights for stability
        nn.init.xavier_uniform_(final_conv.weight, gain=0.1)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def forward(self, z: Tensor) -> Tensor:
        x = self.to_decoder(z)
        x = x.reshape(-1, 64, 7, 7)  # TODO: remove hardcoding?
        x = self.decoder(x)
        return x
