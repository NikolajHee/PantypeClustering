from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.manifold import TSNE


def tsne(z: NDArray[Any]) -> NDArray[Any]:
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(z)


def tsne_plot(tsne_results: NDArray[Any], labels: NDArray[Any]) -> Figure:
    fig = plt.figure(figsize=(10, 8))  # pyright: ignore[reportUnknownMemberType]
    scatter = plt.scatter(  # pyright: ignore[reportUnknownMemberType]
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
    )
    plt.colorbar(scatter)  # pyright: ignore[reportUnknownMemberType]
    plt.title("t-SNE Visualization of Latent Space")  # pyright: ignore[reportUnknownMemberType]
    plt.xlabel("t-SNE 1")  # pyright: ignore[reportUnknownMemberType]
    plt.ylabel("t-SNE 2")  # pyright: ignore[reportUnknownMemberType]

    plt.tight_layout()
    return fig


def fig_to_image(fig: Figure) -> NDArray[Any]:
    fig.canvas.draw()  # pyright: ignore[reportUnknownMemberType]
    image_flat = np.frombuffer(fig.canvas.tostring_argb(), dtype="uint8")  # type: ignore
    image = image_flat.reshape(*reversed([i * 2 for i in fig.canvas.get_width_height()]), 4)
    image = image[:, :, 1:]
    plt.close(fig)
    return image.transpose((2, 0, 1))
