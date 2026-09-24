"""Position and heading decoders fitted on train, selected on validation, scored on test."""

from __future__ import annotations

import copy

import numpy as np
import torch

from placecell_research.evaluation.matched_decode import MatchedPositionDecoder

FIRST_STEP = 15
BATCH = 4096
MLP_HIDDEN = 128
MLP_LEARNING_RATE = 1.0e-3
MLP_WEIGHT_DECAY = 1.0e-4
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 20
SEED = 0


def stack_features(values: np.ndarray, frames: int) -> np.ndarray:
    """The current and frames-1 preceding steps, from FIRST_STEP on."""
    if values.ndim != 3 or values.shape[1] <= FIRST_STEP or not 1 <= frames <= FIRST_STEP + 1:
        raise ValueError("Expected episode/time/features and 1..16 causal frames")
    return np.concatenate(
        [values[:, FIRST_STEP - lag : values.shape[1] - lag] for lag in range(frames)], axis=-1
    )


def supported_rows(
    features: np.ndarray, position_xy: np.ndarray, heading: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Features and [x, y, sin, cos] targets of every step whose 16-step window is valid."""
    window = np.lib.stride_tricks.sliding_window_view(valid, FIRST_STEP + 1, axis=1).all(axis=-1)
    step_heading = heading[:, FIRST_STEP:][window]
    targets = np.column_stack(
        (position_xy[:, FIRST_STEP:][window], np.sin(step_heading), np.cos(step_heading))
    ).astype(np.float32)
    return features[window].astype(np.float32), targets, window.sum(1).tolist()


def angular_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.arctan2(prediction[:, 0], prediction[:, 1])
    actual = np.arctan2(target[:, 0], target[:, 1])
    return np.abs(np.angle(np.exp(1j * (predicted - actual)))) * 180 / np.pi


def scores(prediction: np.ndarray, target: np.ndarray, task: str) -> dict[str, float]:
    prediction = prediction.astype(np.float64)
    target = target.astype(np.float64)
    if task == "heading":
        errors = angular_error(prediction, target)
        return {"median_error_degrees": float(np.median(errors))}
    mse = np.mean((prediction - target) ** 2, axis=0)
    return {"rmse": float(np.sqrt(mse.mean())), "r2": float(np.mean(1 - mse / target.var(0)))}


def normalize_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(x.shape[1])
    squares = np.zeros_like(mean)
    for start in range(0, len(x), BATCH):
        block = x[start : start + BATCH].astype(np.float64)
        mean += block.sum(0)
        squares += (block * block).sum(0)
    mean /= len(x)
    scale = np.sqrt(np.maximum(squares / len(x) - mean * mean, 0))
    scale[scale < 1e-6] = 1
    return mean.astype(np.float32), scale.astype(np.float32)


def predict_mlp(model, x, mean, scale, target_mean, target_scale) -> np.ndarray:
    out = np.empty((len(x), 2), np.float32)
    with torch.no_grad():
        for start in range(0, len(x), BATCH):
            block = torch.from_numpy((np.asarray(x[start : start + BATCH]) - mean) / scale)
            out[start : start + BATCH] = model(block).numpy() * target_scale + target_mean
    return out


def fit_mlp(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Two-layer MLP with early stopping on validation; returns test predictions."""
    torch.manual_seed(SEED)
    x, y = data["train"]
    validation_x, validation_y = data["validation"]
    mean, scale = normalize_stats(x)
    target_mean = y.mean(0)
    target_scale = np.maximum(y.std(0), 1e-6)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], MLP_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(MLP_HIDDEN, 2),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY
    )
    rng = np.random.default_rng(SEED)
    best, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(MLP_MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(x), BATCH):
            ids = order[start : start + BATCH]
            batch_x = torch.from_numpy((np.asarray(x[ids]) - mean) / scale)
            batch_y = torch.from_numpy((y[ids] - target_mean) / target_scale)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        prediction = predict_mlp(model, validation_x, mean, scale, target_mean, target_scale)
        score = float(np.mean(((prediction - validation_y) / target_scale) ** 2))
        if not np.isfinite(score):
            raise ValueError("Non-finite MLP validation score")
        if score < best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= MLP_PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    return predict_mlp(model, data["test"][0], mean, scale, target_mean, target_scale)


def fit_ridge(data: dict[str, tuple[np.ndarray, np.ndarray]]) -> MatchedPositionDecoder:
    """Ridge fitted on train, its regularization selected on validation."""
    decoder = MatchedPositionDecoder.fit(*data["train"])
    decoder.score(*data["validation"], select=True)
    return decoder


def ridge_predict(decoder: MatchedPositionDecoder, x: np.ndarray) -> np.ndarray:
    prediction = np.empty((len(x), 2), np.float32)
    for start in range(0, len(x), BATCH):
        prediction[start : start + BATCH] = (
            x[start : start + BATCH] - decoder.mean
        ) @ decoder.weights[decoder.selected] + decoder.target_mean
    return prediction


def shuffled_code(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Train and validation rows paired with the targets of random other rows: the chance level."""
    rng = np.random.default_rng(SEED)
    return {
        split: (x, y if split == "test" else y[rng.permutation(len(y))])
        for split, (x, y) in data.items()
    }


def within_episode_error(
    decoder: MatchedPositionDecoder, values: np.ndarray, position_xy: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Mean over episodes of the position error (RMSE over x and y) at each step."""
    prediction = ridge_predict(decoder, values.reshape(-1, values.shape[-1]))
    squared = (prediction.reshape(position_xy.shape).astype(np.float64) - position_xy) ** 2
    error = np.where(valid, np.sqrt(squared.mean(-1)), np.nan)
    return np.nanmean(error, axis=0)


def shift_control(prediction: np.ndarray, row_counts: list[int]) -> np.ndarray:
    """Test predictions rolled by one random offset within each episode."""
    rng = np.random.default_rng(SEED)
    shifted = []
    offset = 0
    for length in row_counts:
        block = prediction[offset : offset + length]
        if length > 1:
            block = np.roll(
                block,
                int(rng.integers(max(1, length // 20), max(2, length - length // 20))),
                axis=0,
            )
        shifted.append(block)
        offset += length
    return np.concatenate(shifted)


def decode(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    test_row_counts: list[int],
    *,
    nonlinear: bool = True,
) -> tuple[dict[str, float], MatchedPositionDecoder]:
    """Ridge and MLP scores of position and heading for one feature set, and the position ridge."""
    result = {}
    for task, columns in (("position", slice(0, 2)), ("heading", slice(2, 4))):
        data = {split: (features[split], targets[split][:, columns]) for split in features}
        truth = data["test"][1]
        decoder = fit_ridge(data)
        if task == "position":
            position_decoder = decoder
        prediction = ridge_predict(decoder, data["test"][0])
        result |= {f"{task}_ridge_{k}": v for k, v in scores(prediction, truth, task).items()}
        shifted = shift_control(prediction, test_row_counts)
        result |= {
            f"{task}_ridge_shift_control_{k}": v for k, v in scores(shifted, truth, task).items()
        }
        chance = ridge_predict(fit_ridge(shuffled_code(data)), data["test"][0])
        result |= {
            f"{task}_ridge_shuffled_code_{k}": v for k, v in scores(chance, truth, task).items()
        }
        if nonlinear:
            result |= {f"{task}_mlp_{k}": v for k, v in scores(fit_mlp(data), truth, task).items()}
    return result, position_decoder
