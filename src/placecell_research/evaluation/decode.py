"""Linear decoding utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from placecell_research.numerics.error_metrics import root_mean_squared_error


@dataclass(slots=True)
class DecodeResult:
    """Linear decode summary."""

    rmse: float
    mae: float
    r2: float
    shuffle_rmse: float | None
    train_size: int
    validation_size: int
    predictions: np.ndarray
    targets: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["predictions"] = self.predictions.tolist()
        payload["targets"] = self.targets.tolist()
        return payload


@dataclass(slots=True)
class RidgeDecoderState:
    """Centered ridge parameters fitted without materializing the full design matrix."""

    feature_mean: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray


@dataclass(slots=True)
class FittedPositionDecoder:
    """One position decoder plus its deterministic episode-held-out split."""

    ridge: RidgeDecoderState
    train_indices: np.ndarray
    validation_indices: np.ndarray


def episode_level_decode_skip_reason(episode_ids: np.ndarray) -> str | None:
    """Return why episode-held-out XY decoding cannot produce a valid split."""
    if len(episode_ids) < 4:
        return "Need at least 4 samples for XY decoding."
    if len(np.unique(episode_ids)) < 2:
        return "Need at least 2 episodes for episode-level XY decoding."
    return None


def supports_episode_level_decode(episode_ids: np.ndarray) -> bool:
    """Return whether episode-held-out XY decoding can produce a valid split."""
    return episode_level_decode_skip_reason(episode_ids) is None


def _split_indices(
    num_samples: int,
    train_fraction: float,
    episode_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if num_samples < 4:
        raise ValueError("Need at least 4 samples for XY decoding.")
    rng = np.random.default_rng(0)
    if episode_ids is not None:
        episode_ids = np.asarray(episode_ids)
        if episode_ids.shape != (num_samples,):
            raise ValueError(
                f"Expected episode_ids shape ({num_samples},), got {episode_ids.shape}."
            )
        skip_reason = episode_level_decode_skip_reason(episode_ids)
        if skip_reason is not None:
            raise ValueError(skip_reason)
        unique_episode_ids = np.unique(episode_ids)
        shuffled_episode_ids = rng.permutation(unique_episode_ids)
        split = int(
            max(
                1,
                min(
                    len(shuffled_episode_ids) - 1,
                    round(len(shuffled_episode_ids) * train_fraction),
                ),
            )
        )
        train_mask = np.isin(episode_ids, shuffled_episode_ids[:split])
        indices = np.arange(num_samples, dtype=np.int64)
        return indices[train_mask], indices[~train_mask]

    indices = rng.permutation(np.arange(num_samples, dtype=np.int64))
    split = int(max(1, min(num_samples - 1, round(num_samples * train_fraction))))
    return indices[:split], indices[split:]


def chunked_ridge_fit_predict(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    alpha: float,
    chunk_size: int = 262_144,
) -> np.ndarray:
    """Ridge fit + validation prediction via chunked normal equations."""
    decoder = chunked_ridge_fit(
        features,
        targets,
        train_indices,
        alpha,
        chunk_size=chunk_size,
    )
    return chunked_ridge_predict(
        decoder,
        features,
        validation_indices,
        chunk_size=chunk_size,
    )


def _ridge_means(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_count = int(len(train_indices))
    feature_mean = np.zeros(int(features.shape[1]), dtype=np.float64)
    target_mean = np.zeros(int(targets.shape[1]), dtype=np.float64)
    for start in range(0, train_count, chunk):
        rows = train_indices[start : start + chunk]
        feature_mean += features[rows].sum(axis=0, dtype=np.float64)
        target_mean += targets[rows].sum(axis=0, dtype=np.float64)
    feature_mean /= max(train_count, 1)
    target_mean /= max(train_count, 1)
    return feature_mean, target_mean


def _ridge_gram_and_cross(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    feature_mean: np.ndarray,
    target_mean: np.ndarray,
    chunk: int,
    *,
    gram: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Accumulate the centered normal equations; pass gram=None to skip the O(N*D^2) term."""
    feature_dim = int(features.shape[1])
    target_dim = int(targets.shape[1])
    train_count = int(len(train_indices))
    cross = np.zeros((feature_dim, target_dim), dtype=np.float64)
    for start in range(0, train_count, chunk):
        rows = train_indices[start : start + chunk]
        centered_features = features[rows].astype(np.float64) - feature_mean
        centered_targets = targets[rows].astype(np.float64) - target_mean
        if gram is not None:
            gram += centered_features.T @ centered_features
        cross += centered_features.T @ centered_targets
    return gram, cross


def chunked_ridge_fit(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    alpha: float,
    chunk_size: int = 262_144,
) -> RidgeDecoderState:
    """Fit centered ridge parameters in bounded row memory."""
    feature_dim = int(features.shape[1])
    chunk = max(int(chunk_size), 1)
    feature_mean, target_mean = _ridge_means(features, targets, train_indices, chunk)
    gram, cross = _ridge_gram_and_cross(
        features,
        targets,
        train_indices,
        feature_mean,
        target_mean,
        chunk,
        gram=np.zeros((feature_dim, feature_dim), dtype=np.float64),
    )
    gram[np.diag_indices(feature_dim)] += float(alpha)
    return RidgeDecoderState(
        feature_mean=feature_mean,
        target_mean=target_mean,
        weights=np.linalg.solve(gram, cross),
    )


def chunked_ridge_predict(
    decoder: RidgeDecoderState,
    features: np.ndarray,
    indices: np.ndarray,
    chunk_size: int = 262_144,
) -> np.ndarray:
    """Apply one fitted decoder to arbitrary aligned feature rows."""
    chunk = max(int(chunk_size), 1)
    predictions = np.empty((len(indices), len(decoder.target_mean)), dtype=np.float64)
    for start in range(0, len(indices), chunk):
        rows = indices[start : start + chunk]
        centered_features = features[rows].astype(np.float64) - decoder.feature_mean
        predictions[start : start + len(rows)] = (
            centered_features @ decoder.weights + decoder.target_mean
        )
    return predictions


def fit_position_ridge_decoder(
    codes: np.ndarray,
    positions: np.ndarray,
    *,
    train_fraction: float = 0.8,
    alpha: float = 1.0e-3,
    episode_ids: np.ndarray | None = None,
    chunk_size: int = 262_144,
) -> FittedPositionDecoder:
    """Fit one deterministic position decoder for reuse across paired conditions."""
    if codes.ndim != 2:
        raise ValueError(f"Expected codes [N, D], got {codes.shape}.")
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError(f"Expected positions [N, 2], got {positions.shape}.")
    if len(codes) != len(positions):
        raise ValueError("Codes and positions must have the same first dimension.")
    train_indices, validation_indices = _split_indices(
        len(codes),
        train_fraction,
        episode_ids=episode_ids,
    )
    return FittedPositionDecoder(
        ridge=chunked_ridge_fit(
            codes,
            positions,
            train_indices,
            alpha,
            chunk_size=chunk_size,
        ),
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def linear_decode_position(
    codes: np.ndarray,
    positions: np.ndarray,
    train_fraction: float = 0.8,
    alpha: float = 1.0e-3,
    include_shuffle: bool = True,
    episode_ids: np.ndarray | None = None,
    chunk_size: int = 262_144,
) -> DecodeResult:
    """Fit a deterministic ridge decoder from representations to XY position."""
    if codes.ndim != 2:
        raise ValueError(f"Expected codes [N, D], got {codes.shape}.")
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError(f"Expected positions [N, 2], got {positions.shape}.")
    if len(codes) != len(positions):
        raise ValueError("Codes and positions must have the same first dimension.")

    train_indices, validation_indices = _split_indices(
        len(codes),
        train_fraction,
        episode_ids=episode_ids,
    )
    chunk = max(int(chunk_size), 1)
    feature_dim = int(codes.shape[1])
    feature_mean, target_mean = _ridge_means(codes, positions, train_indices, chunk)
    gram, cross = _ridge_gram_and_cross(
        codes,
        positions,
        train_indices,
        feature_mean,
        target_mean,
        chunk,
        gram=np.zeros((feature_dim, feature_dim), dtype=np.float64),
    )
    gram[np.diag_indices(feature_dim)] += float(alpha)
    decoder = RidgeDecoderState(
        feature_mean=feature_mean,
        target_mean=target_mean,
        weights=np.linalg.solve(gram, cross),
    )
    predictions = chunked_ridge_predict(decoder, codes, validation_indices, chunk_size=chunk_size)
    targets = positions[validation_indices]
    rmse = root_mean_squared_error(targets, predictions)
    mae = float(mean_absolute_error(targets, predictions))
    r2 = float(r2_score(targets, predictions))

    shuffle_rmse: float | None = None
    if include_shuffle:
        shuffled = positions.copy()
        rng = np.random.default_rng(0)
        rng.shuffle(shuffled, axis=0)
        shuffle_target_mean = np.zeros(int(shuffled.shape[1]), dtype=np.float64)
        for start in range(0, len(train_indices), chunk):
            rows = train_indices[start : start + chunk]
            shuffle_target_mean += shuffled[rows].sum(axis=0, dtype=np.float64)
        shuffle_target_mean /= max(int(len(train_indices)), 1)
        _, shuffle_cross = _ridge_gram_and_cross(
            codes,
            shuffled,
            train_indices,
            feature_mean,
            shuffle_target_mean,
            chunk,
            gram=None,
        )
        shuffle_decoder = RidgeDecoderState(
            feature_mean=feature_mean,
            target_mean=shuffle_target_mean,
            weights=np.linalg.solve(gram, shuffle_cross),
        )
        shuffle_predictions = chunked_ridge_predict(
            shuffle_decoder, codes, validation_indices, chunk_size=chunk_size
        )
        shuffle_rmse = root_mean_squared_error(shuffled[validation_indices], shuffle_predictions)

    return DecodeResult(
        rmse=rmse,
        mae=mae,
        r2=r2,
        shuffle_rmse=shuffle_rmse,
        train_size=int(len(train_indices)),
        validation_size=int(len(validation_indices)),
        predictions=predictions.astype(np.float32, copy=False),
        targets=targets.astype(np.float32, copy=False),
    )


def fit_ridge_position_decoder(
    codes: np.ndarray,
    positions: np.ndarray,
    *,
    alpha: float = 1.0e-3,
    chunk_size: int = 262_144,
) -> RidgeDecoderState:
    """Fit the centered ridge decoder on ALL provided samples, no internal split."""
    if codes.ndim != 2:
        raise ValueError(f"Expected codes [N, D], got {codes.shape}.")
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError(f"Expected positions [N, 2], got {positions.shape}.")
    if len(codes) != len(positions):
        raise ValueError("Codes and positions must have the same first dimension.")
    fit_indices = np.arange(len(codes))
    chunk = max(int(chunk_size), 1)
    feature_dim = int(codes.shape[1])
    feature_mean, target_mean = _ridge_means(codes, positions, fit_indices, chunk)
    gram, cross = _ridge_gram_and_cross(
        codes,
        positions,
        fit_indices,
        feature_mean,
        target_mean,
        chunk,
        gram=np.zeros((feature_dim, feature_dim), dtype=np.float64),
    )
    gram[np.diag_indices(feature_dim)] += float(alpha)
    return RidgeDecoderState(
        feature_mean=feature_mean,
        target_mean=target_mean,
        weights=np.linalg.solve(gram, cross),
    )


def score_ridge_position_decoder(
    decoder: RidgeDecoderState,
    codes: np.ndarray,
    positions: np.ndarray,
    *,
    chunk_size: int = 262_144,
) -> tuple[float, float]:
    """Return (rmse, r2) of a previously fitted decoder applied without refitting."""
    if codes.ndim != 2:
        raise ValueError(f"Expected codes [N, D], got {codes.shape}.")
    if len(codes) != len(positions):
        raise ValueError("Codes and positions must have the same first dimension.")
    evaluation_indices = np.arange(len(codes))
    predictions = chunked_ridge_predict(decoder, codes, evaluation_indices, chunk_size=chunk_size)
    return (
        root_mean_squared_error(positions, predictions),
        float(r2_score(positions, predictions)),
    )


def subsample_index_array(indices: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    sampled_positions = rng.choice(len(indices), size=max_samples, replace=False)
    return indices[np.sort(sampled_positions)]


def nonlinear_decode_position(
    codes: np.ndarray,
    positions: np.ndarray,
    train_fraction: float = 0.8,
    hidden_sizes: tuple[int, ...] = (128, 128),
    max_epochs: int = 200,
    batch_size: int = 1024,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    max_train_samples: int = 65_536,
    max_validation_samples: int = 16_384,
    random_seed: int = 0,
    include_shuffle: bool = False,
    episode_ids: np.ndarray | None = None,
) -> DecodeResult:
    """Fit a deterministic MLP decoder from representations to XY position."""
    if codes.ndim != 2:
        raise ValueError(f"Expected codes [N, D], got {codes.shape}.")
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError(f"Expected positions [N, 2], got {positions.shape}.")
    if len(codes) != len(positions):
        raise ValueError("Codes and positions must have the same first dimension.")

    import torch
    from torch import nn

    train_indices, validation_indices = _split_indices(
        len(codes),
        train_fraction,
        episode_ids=episode_ids,
    )
    train_indices = subsample_index_array(train_indices, max_train_samples, random_seed)
    validation_indices = subsample_index_array(
        validation_indices,
        max_validation_samples,
        random_seed + 1,
    )

    train_features = codes[train_indices].astype(np.float32, copy=False)
    train_targets = positions[train_indices].astype(np.float32, copy=False)
    validation_features = codes[validation_indices].astype(np.float32, copy=False)
    targets = positions[validation_indices].astype(np.float32, copy=False)

    feature_mean = train_features.mean(axis=0, keepdims=True)
    raw_feature_std = train_features.std(axis=0, keepdims=True)
    feature_std = np.where(raw_feature_std < 1e-6, 1.0, np.maximum(raw_feature_std, 1e-6))
    target_mean = train_targets.mean(axis=0, keepdims=True)
    target_std = np.maximum(train_targets.std(axis=0, keepdims=True), 1e-6)

    train_features = (train_features - feature_mean) / feature_std
    train_targets = (train_targets - target_mean) / target_std
    validation_features = (validation_features - feature_mean) / feature_std

    torch.manual_seed(int(random_seed))
    layers: list[nn.Module] = []
    input_dim = int(codes.shape[1])
    for hidden_size in hidden_sizes:
        hidden_dim = int(hidden_size)
        if hidden_dim <= 0:
            raise ValueError("hidden_sizes must contain positive integers.")
        layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, 2))
    model = nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_fn = nn.MSELoss()

    x_train = torch.from_numpy(train_features)
    y_train = torch.from_numpy(train_targets)
    generator = torch.Generator().manual_seed(int(random_seed))
    effective_batch_size = max(1, min(int(batch_size), len(train_indices)))
    for _epoch in range(max(1, int(max_epochs))):
        permutation = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(train_indices), effective_batch_size):
            batch_indices = permutation[start : start + effective_batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train[batch_indices])
            loss = loss_fn(prediction, y_train[batch_indices])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        normalized_predictions = model(torch.from_numpy(validation_features)).numpy()
    predictions = normalized_predictions * target_std + target_mean
    rmse = root_mean_squared_error(targets, predictions)
    mae = float(mean_absolute_error(targets, predictions))
    r2 = float(r2_score(targets, predictions))

    shuffle_rmse: float | None = None
    if include_shuffle:
        shuffled = positions.copy()
        rng = np.random.default_rng(random_seed)
        rng.shuffle(shuffled, axis=0)
        shuffled_decode = nonlinear_decode_position(
            codes,
            shuffled,
            train_fraction=train_fraction,
            hidden_sizes=hidden_sizes,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_train_samples=max_train_samples,
            max_validation_samples=max_validation_samples,
            random_seed=random_seed + 2,
            include_shuffle=False,
            episode_ids=episode_ids,
        )
        shuffle_rmse = shuffled_decode.rmse

    return DecodeResult(
        rmse=rmse,
        mae=mae,
        r2=r2,
        shuffle_rmse=shuffle_rmse,
        train_size=int(len(train_indices)),
        validation_size=int(len(validation_indices)),
        predictions=predictions.astype(np.float32, copy=False),
        targets=targets.astype(np.float32, copy=False),
    )
