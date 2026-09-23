"""The analytic successor representation for an environment, from its own trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import AnalysisInput, AnalysisResult


@dataclass(frozen=True)
class SuccessorOracle:
    """The analytic SR over discretised XY, plus what is needed to map codes onto it."""

    successor_matrix: np.ndarray
    transition_matrix: np.ndarray
    occupancy: np.ndarray
    state_of_bin: np.ndarray
    bin_edges_x: np.ndarray
    bin_edges_y: np.ndarray
    discount_gamma: float
    dropped_bins: int

    @property
    def num_states(self) -> int:
        return int(self.successor_matrix.shape[0])

    def states_for_positions(self, positions_xy: np.ndarray) -> np.ndarray:
        """Map (..., 2) world positions onto state indices; -1 where the bin was dropped."""
        x_index = np.clip(
            np.digitize(positions_xy[..., 0], self.bin_edges_x) - 1,
            0,
            self.state_of_bin.shape[0] - 1,
        )
        y_index = np.clip(
            np.digitize(positions_xy[..., 1], self.bin_edges_y) - 1,
            0,
            self.state_of_bin.shape[1] - 1,
        )
        return self.state_of_bin[x_index, y_index]


def build_successor_oracle(
    positions_xy: np.ndarray,
    valid_steps: np.ndarray,
    *,
    discount_gamma: float,
    num_bins_x: int = 20,
    num_bins_y: int = 20,
) -> SuccessorOracle:
    """Count T from real trajectories and invert it."""
    if not 0.0 <= discount_gamma < 1.0:
        raise ValueError(f"discount_gamma must be in [0, 1); got {discount_gamma}.")
    if positions_xy.ndim != 3 or positions_xy.shape[-1] != 2:
        raise ValueError(f"positions_xy must be (episodes, time, 2); got {positions_xy.shape}.")

    finite = np.isfinite(positions_xy).all(axis=-1) & valid_steps.astype(bool)
    if not finite.any():
        raise ValueError("No valid positions to build an oracle from.")
    flat = positions_xy[finite]
    edges_x = np.linspace(flat[:, 0].min(), flat[:, 0].max(), num_bins_x + 1)
    edges_y = np.linspace(flat[:, 1].min(), flat[:, 1].max(), num_bins_y + 1)

    x_index = np.clip(np.digitize(positions_xy[..., 0], edges_x) - 1, 0, num_bins_x - 1)
    y_index = np.clip(np.digitize(positions_xy[..., 1], edges_y) - 1, 0, num_bins_y - 1)
    bin_index = x_index * num_bins_y + y_index
    num_bins = num_bins_x * num_bins_y

    usable = finite[:, :-1] & finite[:, 1:]
    source = bin_index[:, :-1][usable]
    destination = bin_index[:, 1:][usable]
    counts = np.zeros((num_bins, num_bins), dtype=np.float64)
    np.add.at(counts, (source, destination), 1.0)

    occupancy_all = np.bincount(bin_index[finite], minlength=num_bins).astype(np.float64)
    outgoing = counts.sum(axis=1)
    reachable = outgoing > 0
    dropped = int((~reachable).sum())
    if not reachable.any():
        raise ValueError("No bin has an outgoing transition; cannot build a transition matrix.")

    counts = counts[np.ix_(reachable, reachable)]
    outgoing = counts.sum(axis=1, keepdims=True)
    empty_rows = outgoing[:, 0] == 0
    transition = np.divide(counts, np.maximum(outgoing, 1.0))
    transition[empty_rows, empty_rows.nonzero()[0]] = 1.0

    num_states = transition.shape[0]
    successor = (1.0 - discount_gamma) * np.linalg.inv(
        np.eye(num_states) - discount_gamma * transition
    )

    state_of_bin = np.full(num_bins, -1, dtype=np.int64)
    state_of_bin[reachable.nonzero()[0]] = np.arange(num_states)

    return SuccessorOracle(
        successor_matrix=successor,
        transition_matrix=transition,
        occupancy=occupancy_all[reachable],
        state_of_bin=state_of_bin.reshape(num_bins_x, num_bins_y),
        bin_edges_x=edges_x,
        bin_edges_y=edges_y,
        discount_gamma=discount_gamma,
        dropped_bins=dropped,
    )


def oracle_features(oracle: SuccessorOracle, features_by_state: np.ndarray) -> np.ndarray:
    """Psi = (1-gamma) * M @ Phi: the successor FEATURE map a learned psi should approximate."""
    if features_by_state.shape[0] != oracle.num_states:
        raise ValueError(
            f"features_by_state has {features_by_state.shape[0]} rows but the oracle has "
            f"{oracle.num_states} states."
        )
    return oracle.successor_matrix @ features_by_state


def compare_to_oracle(
    learned: np.ndarray,
    states: np.ndarray,
    oracle: SuccessorOracle,
    features_by_state: np.ndarray,
) -> dict[str, float]:
    """Does a learned code have the GEOMETRY of a successor representation?"""
    target = oracle_features(oracle, features_by_state)
    usable = states >= 0
    if usable.sum() < 2:
        raise ValueError("Fewer than two usable timesteps map onto oracle states.")

    def _state_means(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assignment = states[usable]
        sums = np.zeros((oracle.num_states, values.shape[-1]))
        np.add.at(sums, assignment, values[usable])
        counts = np.bincount(assignment, minlength=oracle.num_states).astype(np.float64)
        seen = counts > 0
        return sums[seen] / counts[seen, None], seen

    learned_means, seen = _state_means(learned)
    if seen.sum() < 3:
        raise ValueError("Fewer than three states were visited; RSA is meaningless.")

    def _similarity(matrix: np.ndarray) -> np.ndarray:
        normed = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        return normed @ normed.T

    def _upper(matrix: np.ndarray) -> np.ndarray:
        return matrix[np.triu_indices_from(matrix, k=1)]

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    oracle_similarity = _upper(_similarity(target[seen]))
    learned_similarity = _upper(_similarity(learned_means))
    position_similarity = _upper(_similarity(np.eye(int(seen.sum()))))
    centres_x = 0.5 * (oracle.bin_edges_x[:-1] + oracle.bin_edges_x[1:])
    centres_y = 0.5 * (oracle.bin_edges_y[:-1] + oracle.bin_edges_y[1:])
    grid_x, grid_y = np.meshgrid(centres_x, centres_y, indexing="ij")
    bin_of_state = np.full(oracle.num_states, -1, dtype=np.int64)
    flat_state_of_bin = oracle.state_of_bin.reshape(-1)
    for bin_index, state in enumerate(flat_state_of_bin):
        if state >= 0:
            bin_of_state[state] = bin_index
    coordinates = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)[bin_of_state][seen]
    euclidean_similarity = _upper(
        -np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    )

    return {
        "oracle_rsa": _corr(learned_similarity, oracle_similarity),
        "position_rsa_baseline": _corr(position_similarity, oracle_similarity),
        "euclidean_rsa_baseline": _corr(euclidean_similarity, oracle_similarity),
        "num_states_compared": float(seen.sum()),
        "dropped_bins": float(oracle.dropped_bins),
        "occupancy_max_over_median": float(
            oracle.occupancy.max() / max(np.median(oracle.occupancy), 1.0)
        ),
    }


def _orthonormal_span(matrix: np.ndarray) -> np.ndarray:
    """Orthonormal basis for a set of column vectors."""
    basis, _ = np.linalg.qr(matrix)
    return basis


def eigen_comparison(
    oracle: SuccessorOracle,
    learned_by_state: np.ndarray,
    *,
    num_modes: int = 8,
) -> dict[str, float]:
    """Stachenfeld et al. (2017) grid-cell prediction: grid cells are the SR eigenvectors."""
    effective_num_modes = min(
        int(num_modes),
        max(oracle.num_states - 1, 0),
        learned_by_state.shape[1],
    )
    if effective_num_modes < 1:
        raise ValueError(
            "eigen_comparison requires at least two oracle states and one learned feature."
        )
    values, vectors = np.linalg.eig(oracle.successor_matrix)
    order = np.argsort(-values.real)[1 : effective_num_modes + 1]
    oracle_modes = np.real(vectors[:, order])

    state_of_bin = oracle.state_of_bin
    num_bins_x, num_bins_y = state_of_bin.shape
    spatial_pairs: list[tuple[int, int]] = []
    for bin_x in range(num_bins_x):
        for bin_y in range(num_bins_y):
            state = state_of_bin[bin_x, bin_y]
            if state < 0:
                continue
            for step_x, step_y in ((1, 0), (0, 1)):
                other_x, other_y = bin_x + step_x, bin_y + step_y
                if other_x >= num_bins_x or other_y >= num_bins_y:
                    continue
                neighbour = state_of_bin[other_x, other_y]
                if neighbour >= 0:
                    spatial_pairs.append((int(state), int(neighbour)))
    pair_index = np.array(spatial_pairs, dtype=np.int64)
    transition = oracle.transition_matrix
    connectivity = (
        transition[pair_index[:, 0], pair_index[:, 1]]
        + transition[pair_index[:, 1], pair_index[:, 0]]
        if pair_index.size
        else np.empty(0)
    )

    def _discontinuity(mode: np.ndarray) -> float:
        if mode.std() < 1e-12 or connectivity.size < 8:
            return float("nan")
        jump = np.abs(mode[pair_index[:, 0]] - mode[pair_index[:, 1]])
        weak = connectivity <= np.quantile(connectivity, 0.25)
        strong = connectivity >= np.quantile(connectivity, 0.75)
        if not weak.any() or not strong.any():
            return float("nan")
        return float(jump[weak].mean() / max(jump[strong].mean(), 1e-12))

    leading = oracle_modes[:, 0]
    fragmentation = [_discontinuity(oracle_modes[:, i]) for i in range(oracle_modes.shape[1])]

    centered = learned_by_state - learned_by_state.mean(axis=0, keepdims=True)
    learned_modes = np.linalg.svd(centered, full_matrices=False)[0][:, :effective_num_modes]
    centered_oracle_modes = oracle_modes - oracle_modes.mean(axis=0, keepdims=True)
    overlap = _orthonormal_span(centered_oracle_modes).T @ _orthonormal_span(learned_modes)
    alignment = float((overlap**2).sum() / effective_num_modes)
    finite_fragmentation = [
        value for value in fragmentation if np.isfinite(value)
    ]

    return {
        "eigen_leading_fragmentation": _discontinuity(leading),
        "eigen_mean_fragmentation": (
            float(np.mean(finite_fragmentation))
            if finite_fragmentation
            else float("nan")
        ),
        "eigen_alignment": alignment,
        "eigen_alignment_chance": float(effective_num_modes / oracle.num_states),
        "eigen_num_modes": float(effective_num_modes),
    }


@dataclass(slots=True)
class SuccessorOracleComparisonModule:
    """Compare a learned successor feature with the aligned analytic target M @ Phi."""

    name: str = "sr_oracle"
    cost_tier: str = "heavy"

    def run(
        self,
        inputs: list[AnalysisInput],
        labels: list[str],
        output_dir: Path,
        config: dict[str, Any],
    ) -> AnalysisResult:
        del output_dir
        if len(inputs) != 2:
            raise ValueError(f"sr_oracle requires exactly two inputs, got {len(inputs)}.")
        for metadata_key in ("dataset_artifact_id", "split_artifact_id"):
            successor_reference = inputs[0].metadata.get(metadata_key)
            feature_reference = inputs[1].metadata.get(metadata_key)
            if (
                successor_reference is not None
                and feature_reference is not None
                and successor_reference != feature_reference
            ):
                raise ValueError(
                    "sr_oracle inputs must be sample-aligned; "
                    f"{metadata_key} differs ({successor_reference!r} vs {feature_reference!r})."
                )

        successor = np.asarray(inputs[0].representation)
        feature = np.asarray(inputs[1].representation)
        positions = np.asarray(inputs[0].position_xy)
        if (
            successor.ndim != 3
            or feature.ndim != 3
            or successor.shape[:2] != feature.shape[:2]
            or successor.shape[:2] != positions.shape[:2]
        ):
            raise ValueError(
                "sr_oracle requires successor, feature, and position inputs with matching "
                f"[episode, time] axes, got {successor.shape}, {feature.shape}, and "
                f"{positions.shape}."
            )
        if inputs[1].position_xy.shape != positions.shape or not np.allclose(
            inputs[1].position_xy,
            positions,
            atol=1e-5,
            equal_nan=True,
        ):
            raise ValueError("sr_oracle inputs must share aligned positions.")

        successor_valid = np.asarray(inputs[0].valid_mask, dtype=bool)
        feature_valid = np.asarray(inputs[1].valid_mask, dtype=bool)
        if (
            successor_valid.shape != positions.shape[:2]
            or feature_valid.shape != positions.shape[:2]
        ):
            raise ValueError(
                "sr_oracle valid masks must match the shared position [episode, time] axes."
            )
        valid = successor_valid & feature_valid

        oracle = build_successor_oracle(
            positions,
            valid,
            discount_gamma=float(config.get("sr_oracle_discount_gamma", 0.95)),
            num_bins_x=int(config.get("sr_oracle_num_bins_x", 20)),
            num_bins_y=int(config.get("sr_oracle_num_bins_y", 20)),
        )
        states = oracle.states_for_positions(positions).reshape(-1)
        states[~valid.reshape(-1)] = -1
        usable = states >= 0
        flat_feature = feature.reshape(-1, feature.shape[-1])
        feature_sums = np.zeros((oracle.num_states, feature.shape[-1]), dtype=np.float64)
        np.add.at(feature_sums, states[usable], flat_feature[usable])
        feature_counts = np.bincount(states[usable], minlength=oracle.num_states).astype(np.float64)
        features_by_state = feature_sums / feature_counts[:, None]
        flat_successor = successor.reshape(-1, successor.shape[-1])
        metrics = compare_to_oracle(flat_successor, states, oracle, features_by_state)

        successor_sums = np.zeros((oracle.num_states, successor.shape[-1]), dtype=np.float64)
        np.add.at(successor_sums, states[usable], flat_successor[usable])
        successor_counts = np.bincount(states[usable], minlength=oracle.num_states).astype(
            np.float64
        )
        learned_by_state = successor_sums / np.maximum(successor_counts, 1.0)[:, None]
        metrics.update(
            eigen_comparison(
                oracle,
                learned_by_state,
                num_modes=int(config.get("sr_oracle_num_eigen_modes", 8)),
            )
        )
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={},
            figures={},
            tables={},
            metadata={
                "sr_oracle_discount_gamma": oracle.discount_gamma,
                "sr_oracle_num_states": oracle.num_states,
                "successor_label": labels[0] if len(labels) > 0 else inputs[0].label,
                "feature_label": labels[1] if len(labels) > 1 else inputs[1].label,
            },
        )
