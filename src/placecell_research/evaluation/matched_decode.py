"""Train-standardized ridge selection on validation, frozen for confirmation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from placecell_research.numerics.error_metrics import rmse_from_coordinate_mse


@dataclass
class MatchedPositionDecoder:
    mean: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray
    alphas: tuple[float, ...]
    selected: int | None = None

    @classmethod
    def fit(cls, codes: np.ndarray, positions: np.ndarray) -> MatchedPositionDecoder:
        if codes.ndim != 2 or positions.shape != (len(codes), 2) or len(codes) < 2:
            raise ValueError("Matched decoding requires codes [N,D] and positions [N,2].")
        mean = np.mean(codes, axis=0, dtype=np.float64)
        target_mean = np.mean(positions, axis=0, dtype=np.float64)
        width = codes.shape[1]
        gram, cross = np.zeros((width, width)), np.zeros((width, 2))
        for start in range(0, len(codes), 65536):
            block = codes[start : start + 65536].astype(np.float64) - mean
            targets = positions[start : start + 65536].astype(np.float64) - target_mean
            gram += block.T @ block
            cross += block.T @ targets
        scale = np.sqrt(np.maximum(gram.diagonal(), 0) / len(codes))
        scale[scale == 0] = 1.0
        gram /= scale[:, None] * scale[None, :]
        cross /= scale[:, None]
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        alphas = tuple(10.0**exponent for exponent in range(-6, 4))
        projected = eigenvectors.T @ cross
        weights = np.stack(
            [
                (eigenvectors @ (projected / (np.maximum(eigenvalues, 0) + alpha)[:, None]))
                / scale[:, None]
                for alpha in alphas
            ]
        )
        return cls(mean, target_mean, weights, alphas)

    def score(self, codes: np.ndarray, positions: np.ndarray, *, select: bool) -> dict[str, float]:
        if select and self.selected is not None:
            raise ValueError("The matched decoder was already selected; do not retune it.")
        if not select and self.selected is None:
            raise ValueError("Score validation before confirmation to select ridge regularization.")
        choices = range(len(self.alphas)) if select else [self.selected]
        errors = []
        for index in choices:
            squared_error = np.zeros(2)
            for start in range(0, len(codes), 65536):
                predictions = (codes[start : start + 65536] - self.mean) @ self.weights[
                    index
                ] + self.target_mean
                squared_error += ((positions[start : start + 65536] - predictions) ** 2).sum(0)
            errors.append(squared_error)
        if select:
            scores = [rmse_from_coordinate_mse(error / len(codes)) for error in errors]
            self.selected = int(np.argmin(scores))
            error = errors[self.selected]
        else:
            error = errors[0]
        total = ((positions - positions.mean(0)) ** 2).sum(0)
        r2 = np.where(total > 0, 1 - error / np.where(total > 0, total, 1), np.nan)
        return {
            "matched_decode_rmse": float(rmse_from_coordinate_mse(error / len(codes))),
            "matched_decode_r2": float(r2.mean()),
            "matched_decode_alpha": self.alphas[self.selected],
        }


def matched_decode_metrics(
    representations: dict[str, np.ndarray],
    positions: np.ndarray,
    valid: np.ndarray,
    split: str,
    decoders: dict[str, MatchedPositionDecoder],
) -> dict[str, float]:
    mask = valid.reshape(-1).astype(bool)
    targets = positions.reshape(-1, 2)[mask]
    metrics = {}
    for source, values in representations.items():
        codes = values.reshape(-1, values.shape[-1])[mask]
        if split == "train":
            decoders[source] = MatchedPositionDecoder.fit(codes, targets)
        else:
            if source not in decoders:
                raise ValueError(f"Matched decoder has no training data for {source}.")
            scores = decoders[source].score(codes, targets, select=split == "validation")
            metrics.update({f"{source}.{key}": value for key, value in scores.items()})
    return metrics
