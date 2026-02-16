from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_paths: list[Path]
    test_paths: list[Path]


def _list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    paths.sort()
    return paths


def load_cougar_paths(root: Path) -> tuple[list[Path], list[int]]:
    face_dir = root / "cougar_face"
    body_dir = root / "cougar_body"
    if not face_dir.exists() or not body_dir.exists():
        raise FileNotFoundError(
            f"Expected {face_dir} and {body_dir} to exist. Got: {root}" 
        )

    face_paths = _list_images(face_dir)
    body_paths = _list_images(body_dir)

    paths = face_paths + body_paths
    # Label convention: +1 = face, -1 = body
    labels = ([1] * len(face_paths)) + ([-1] * len(body_paths))
    return paths, labels


def _preprocess_image(
    path: Path,
    size: tuple[int, int] = (32, 32),
    grayscale: bool = True,
) -> np.ndarray:
    img = Image.open(path)
    img = img.convert("L" if grayscale else "RGB")
    img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    # Normalize pixel range to [0,1]
    arr = arr / 255.0
    return arr


def extract_features(
    paths: list[Path],
    size: tuple[int, int] = (32, 32),
    grayscale: bool = True,
    add_edge_hist: bool = True,
    edge_bins: int = 16,
) -> np.ndarray:
    """Simple fixed-length features.

    Base: flattened resized grayscale pixels.
    Optional: histogram of gradient magnitudes (captures edges/structure).

    This stays within "construct your own image features" without using a pre-built
    feature extractor.
    """

    feats: list[np.ndarray] = []
    for p in paths:
        img = _preprocess_image(p, size=size, grayscale=grayscale)

        base = img.reshape(-1)
        if add_edge_hist:
            # Sobel-ish gradients via simple finite differences.
            gx = np.zeros_like(img)
            gy = np.zeros_like(img)
            gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
            gy[1:-1, :] = img[2:, :] - img[:-2, :]
            mag = np.sqrt(gx * gx + gy * gy)
            # Histogram of magnitudes (normalized)
            hist, _ = np.histogram(mag, bins=edge_bins, range=(0.0, float(mag.max() + 1e-6)))
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-12)
            feat = np.concatenate([base, hist], axis=0)
        else:
            feat = base

        feats.append(feat)

    X = np.stack(feats, axis=0)
    return X


def extract_features_hoglike(
    paths: list[Path],
    size: tuple[int, int] = (64, 64),
    pixels_downsample: tuple[int, int] = (16, 16),
    cell_size: int = 8,
    n_orient_bins: int = 9,
) -> np.ndarray:
    """Handcrafted HOG-like descriptor (no external feature libs).

    - Resize to `size` and convert to grayscale.
    - Compute gradient magnitude + orientation.
    - Build orientation histograms per cell.
    - Concatenate with coarse downsampled pixel intensities for global context.
    """

    feats: list[np.ndarray] = []
    for p in paths:
        img = _preprocess_image(p, size=size, grayscale=True)

        # coarse pixels
        coarse = Image.fromarray((img * 255.0).astype(np.uint8), mode="L").resize(
            pixels_downsample, Image.BILINEAR
        )
        coarse = (np.asarray(coarse, dtype=np.float32) / 255.0).reshape(-1)

        # gradients
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
        gy[1:-1, :] = img[2:, :] - img[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy) + 1e-12

        # orientation in [0, 180)
        ori = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0

        h, w = img.shape
        if h % cell_size != 0 or w % cell_size != 0:
            raise ValueError("size must be divisible by cell_size")

        n_cells_y = h // cell_size
        n_cells_x = w // cell_size
        bin_width = 180.0 / n_orient_bins

        hog = np.zeros((n_cells_y, n_cells_x, n_orient_bins), dtype=np.float32)
        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                y0, y1 = cy * cell_size, (cy + 1) * cell_size
                x0, x1 = cx * cell_size, (cx + 1) * cell_size
                cell_ori = ori[y0:y1, x0:x1].reshape(-1)
                cell_mag = mag[y0:y1, x0:x1].reshape(-1)

                # soft binning to nearest bin (simple)
                bins = np.floor(cell_ori / bin_width).astype(int)
                bins = np.clip(bins, 0, n_orient_bins - 1)
                for b, m in zip(bins, cell_mag, strict=False):
                    hog[cy, cx, b] += float(m)

        hog_vec = hog.reshape(-1)
        hog_vec = hog_vec / (np.linalg.norm(hog_vec) + 1e-12)

        feat = np.concatenate([coarse, hog_vec], axis=0)
        feats.append(feat)

    return np.stack(feats, axis=0)


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def make_split(
    root: Path,
    test_ratio: float = 0.3,
    seed: int = 0,
    feature_mode: str = "hoglike",
) -> DatasetSplit:
    paths, labels = load_cougar_paths(root)
    paths = list(paths)
    labels = np.asarray(labels, dtype=np.int64)

    rng = np.random.default_rng(seed)

    idx_face = np.where(labels == 1)[0]
    idx_body = np.where(labels == -1)[0]

    rng.shuffle(idx_face)
    rng.shuffle(idx_body)

    n_face_test = int(round(len(idx_face) * test_ratio))
    n_body_test = int(round(len(idx_body) * test_ratio))

    test_idx = np.concatenate([idx_face[:n_face_test], idx_body[:n_body_test]])
    train_idx = np.concatenate([idx_face[n_face_test:], idx_body[n_body_test:]])

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    train_paths = [paths[i] for i in train_idx]
    test_paths = [paths[i] for i in test_idx]

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    if feature_mode == "pixel_edge":
        X_train = extract_features(train_paths)
        X_test = extract_features(test_paths)
    elif feature_mode == "hoglike":
        X_train = extract_features_hoglike(train_paths)
        X_test = extract_features_hoglike(test_paths)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_paths=train_paths,
        test_paths=test_paths,
    )
