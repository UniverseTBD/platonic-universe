"""A self-contained, CPU-cheap galaxy morphology classifier for the stretch goal.

The Concept-LDA notebook's stretch claim is that the method is *model-agnostic*:
the same pipeline that recovered planted concepts should recover morphology
classes from the embeddings of a real galaxy-image classifier. To keep the
notebook runnable end-to-end on a laptop with no downloads, this module

1. procedurally renders three galaxy morphologies (elliptical, spiral,
   edge-on disk) as small grayscale images, and
2. trains a compact CNN classifier on them, exposing its penultimate layer as an
   embedding.

This mirrors the tiny ConvNet used by galaxy-morphology tutorials such as
`iamsh4shank/gal-classifier` (Galaxy10). To run the *real* thing instead, swap
``embed_images`` for any astronomical foundation model — e.g.
``pu.models.get_adapter("dinov2")`` / ``"astropt"`` on Galaxy10 thumbnails; the
Concept-LDA code downstream does not change.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Procedural galaxy images                                                    #
# --------------------------------------------------------------------------- #

def _grid(size: int):
    ax = np.linspace(-1.0, 1.0, size)
    return np.meshgrid(ax, ax)


def render_galaxy(kind: str, rng: np.random.Generator, size: int = 48) -> np.ndarray:
    """Render one (size, size) grayscale galaxy of the given morphology in [0, 1]."""
    xx, yy = _grid(size)
    theta = rng.uniform(0, np.pi)
    ct, st = np.cos(theta), np.sin(theta)
    xr, yr = xx * ct + yy * st, -xx * st + yy * ct
    if kind == "elliptical":                      # smooth, round-ish Sersic blob
        q = rng.uniform(0.7, 1.0)
        r = np.sqrt(xr ** 2 + (yr / q) ** 2)
        img = np.exp(-(r / rng.uniform(0.35, 0.5)) ** 1.0)
    elif kind == "spiral":                         # bulge + logarithmic arms
        r = np.sqrt(xr ** 2 + yr ** 2) + 1e-3
        phi = np.arctan2(yr, xr)
        arms = np.cos(2 * (phi - 3.0 * np.log(r)))
        bulge = np.exp(-(r / 0.28) ** 2)
        disk = np.exp(-(r / 0.6)) * (0.5 + 0.5 * arms) * (r < 0.95)
        img = np.clip(0.7 * bulge + 0.7 * disk, 0, None)
    elif kind == "edge_on":                        # thin, elongated disk
        q = rng.uniform(0.12, 0.22)
        r = np.sqrt((xr / rng.uniform(0.55, 0.7)) ** 2 + (yr / q) ** 2)
        img = np.exp(-r ** 2)
    else:
        raise ValueError(kind)
    img = img / (img.max() + 1e-8)
    img = np.clip(img + rng.normal(0, 0.04, img.shape), 0, 1)
    return img.astype(np.float32)


def make_galaxy_dataset(n_per_class: int = 250, size: int = 48,
                        random_state: int = 0) -> Tuple[np.ndarray, np.ndarray, list]:
    """Return ``(images (N,1,size,size), labels (N,), class_names)``."""
    rng = np.random.default_rng(random_state)
    names = ["elliptical", "spiral", "edge_on"]
    imgs, ys = [], []
    for c, kind in enumerate(names):
        for _ in range(n_per_class):
            imgs.append(render_galaxy(kind, rng, size))
            ys.append(c)
    X = np.stack(imgs)[:, None, :, :].astype(np.float32)
    y = np.asarray(ys)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], names


# --------------------------------------------------------------------------- #
# Compact CNN classifier + embedding                                          #
# --------------------------------------------------------------------------- #

class TinyGalaxyCNN(nn.Module):
    """A small ConvNet: 3 conv blocks -> 64-d embedding -> linear classifier."""

    def __init__(self, n_classes: int = 3, embed_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.embed = nn.Linear(64, embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x, return_embedding: bool = False):
        h = self.features(x).flatten(1)
        z = torch.relu(self.embed(h))
        if return_embedding:
            return z
        return self.head(z)


def train_classifier(X: np.ndarray, y: np.ndarray, epochs: int = 6,
                     batch_size: int = 64, lr: float = 1e-3,
                     seed: int = 0) -> Tuple[TinyGalaxyCNN, float]:
    """Train the CNN on CPU for a few epochs; return ``(model, train_accuracy)``."""
    torch.manual_seed(seed)
    model = TinyGalaxyCNN(n_classes=int(y.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.as_tensor(X)
    yt = torch.as_tensor(y, dtype=torch.long)
    n = len(Xt)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(n, generator=g).split(batch_size):
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = float((model(Xt).argmax(1) == yt).float().mean())
    return model, acc


def embed_images(model: TinyGalaxyCNN, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """Extract the penultimate-layer ``(N, embed_dim)`` embedding on CPU."""
    model.eval()
    Xt = torch.as_tensor(X)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            out.append(model(Xt[i:i + batch_size], return_embedding=True).numpy())
    return np.concatenate(out, axis=0)
