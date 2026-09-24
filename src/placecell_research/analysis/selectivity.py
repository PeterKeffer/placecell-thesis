"""Mixed-selectivity analysis: are the units really spatial, and which spatial code?"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from placecell_research.evaluation.decode import chunked_ridge_fit_predict

from .base import AnalysisInput, AnalysisResult
from .helpers import subsample_indices, write_csv
from .probing import distance_to_points, distance_to_segments, episode_split_indices
from .world_overlay import overlay_bounds, resolve_world_overlay

_GROUP_LABELS = {
    "position": "place_like",
    "heading": "hd_like",
    "speed": "speed_like",
    "angular_velocity": "angular_velocity_like",
    "time": "time_like",
    "visual": "visual_like",
}


def _build_position_basis(
    positions: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    centers_x: int,
    centers_y: int,
    width_scale: float,
) -> np.ndarray:
    """Gaussian RBF basis over the arena: a smooth, moderate-dim position code."""
    (x_low, x_high), (y_low, y_high) = bounds
    center_x = np.linspace(x_low, x_high, centers_x, dtype=np.float64)
    center_y = np.linspace(y_low, y_high, centers_y, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(center_x, center_y, indexing="ij")
    centers = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    spacing_x = (x_high - x_low) / max(centers_x - 1, 1)
    spacing_y = (y_high - y_low) / max(centers_y - 1, 1)
    width = max(width_scale * 0.5 * (spacing_x + spacing_y), 1e-6)
    points = positions.astype(np.float64)
    squared_distance = (
        np.sum(points * points, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * points @ centers.T
    )
    return np.exp(-np.maximum(squared_distance, 0.0) / (2.0 * width * width)).astype(np.float32)


def _resolve_position_bounds(
    env_id: str, positions: np.ndarray, env_kwargs=None
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Arena bounds from the world overlay when known, else padded data extent."""
    overlay = resolve_world_overlay(env_id, env_kwargs)
    if overlay is not None:
        return overlay_bounds(overlay)
    margin_x = 0.05 * max(float(positions[:, 0].max() - positions[:, 0].min()), 1e-6)
    margin_y = 0.05 * max(float(positions[:, 1].max() - positions[:, 1].min()), 1e-6)
    return (
        (float(positions[:, 0].min() - margin_x), float(positions[:, 0].max() + margin_x)),
        (float(positions[:, 1].min() - margin_y), float(positions[:, 1].max() + margin_y)),
    )


def _heading_features(heading: np.ndarray, harmonics: int) -> np.ndarray:
    """Circular heading as sin/cos harmonics (captures HD and axis-tuned cells)."""
    columns = []
    for order in range(1, max(harmonics, 1) + 1):
        columns.append(np.sin(order * heading))
        columns.append(np.cos(order * heading))
    return np.stack(columns, axis=1).astype(np.float32)


def _standardize(values: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance columns so ridge penalises groups comparably."""
    centered = values - values.mean(axis=0, keepdims=True)
    scale = np.maximum(centered.std(axis=0, keepdims=True), 1e-8)
    return (centered / scale).astype(np.float32)


def _visual_pca_features(latent: np.ndarray, num_components: int) -> np.ndarray:
    """Top principal components of the frozen AE latent (the visual input), standardized."""
    centered = latent.astype(np.float64) - latent.mean(axis=0, dtype=np.float64)
    covariance = centered.T @ centered
    _eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    top_components = eigenvectors[:, ::-1][:, : max(1, min(num_components, eigenvectors.shape[1]))]
    return _standardize((centered @ top_components).astype(np.float32))


def _per_output_r2(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Per-unit coefficient of determination on held-out samples."""
    residual = targets - predictions
    residual_sum = np.sum(residual * residual, axis=0)
    centered = targets - targets.mean(axis=0, keepdims=True)
    total_sum = np.sum(centered * centered, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - residual_sum / total_sum
    return np.where(total_sum > 1e-12, r2, 0.0).astype(np.float64)


def _flatten_valid_rows(
    analysis_input: AnalysisInput,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray,
    np.ndarray | None,
]:
    """Flatten to valid, finite per-step rows shared across every feature group."""
    episodes, steps = analysis_input.position_xy.shape[:2]
    num_units = int(analysis_input.representation.shape[-1])
    representation = analysis_input.representation.reshape(-1, num_units)
    positions = analysis_input.position_xy.reshape(-1, 2)
    time_fraction = np.broadcast_to(
        (np.arange(steps, dtype=np.float32) / float(max(steps - 1, 1)))[None, :],
        (episodes, steps),
    ).reshape(-1)
    episode_ids = np.repeat(np.arange(episodes, dtype=np.int64), steps)
    valid = np.asarray(analysis_input.valid_mask, dtype=bool).reshape(-1)

    heading = None if analysis_input.heading is None else analysis_input.heading.reshape(-1)
    kinematics = (
        None
        if analysis_input.kinematics is None
        else analysis_input.kinematics.reshape(-1, analysis_input.kinematics.shape[-1])
    )
    latent = (
        None
        if analysis_input.latent is None
        else analysis_input.latent.reshape(-1, analysis_input.latent.shape[-1])
    )

    keep = (
        valid
        & np.all(np.isfinite(representation), axis=1)
        & np.all(np.isfinite(positions), axis=1)
    )
    if heading is not None:
        keep &= np.isfinite(heading)
    if kinematics is not None:
        keep &= np.all(np.isfinite(kinematics), axis=1)
    if latent is not None:
        keep &= np.all(np.isfinite(latent), axis=1)

    representation = representation[keep]
    positions = positions[keep]
    time_fraction = time_fraction[keep]
    episode_ids = episode_ids[keep]
    heading = None if heading is None else heading[keep]
    kinematics = None if kinematics is None else kinematics[keep]
    latent = None if latent is None else latent[keep]
    return representation, positions, heading, kinematics, time_fraction, episode_ids, latent


def _subsample_selection(sample_count: int, max_samples: int, seed: int) -> np.ndarray | None:
    """Row indices to keep so the RBF design matrix and ridge solves stay bounded."""
    if max_samples <= 0 or sample_count <= max_samples:
        return None
    return subsample_indices(sample_count, max_samples, seed)


@dataclass(slots=True)
class SelectivityPartitionModule:
    """Per-unit cross-validated variance partition: position vs non-positional drivers."""

    name: str = "selectivity_partition"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def required_batch_keys(self) -> set[str]:
        return {"latent"}

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        ridge_alpha = float(config.get("selectivity_ridge_alpha", 1e-2))
        train_fraction = float(config.get("selectivity_train_fraction", 0.8))
        split_seed = int(config.get("selectivity_split_seed", 0))
        min_r2 = float(config.get("selectivity_min_r2", 0.02))
        mixed_dominance = float(config.get("selectivity_mixed_dominance", 0.6))
        max_samples = int(config.get("selectivity_max_samples", 150_000))

        representation, positions, heading, kinematics, time_fraction, episode_ids, latent = (
            _flatten_valid_rows(analysis_input)
        )
        if representation.shape[0] < 8 or len(np.unique(episode_ids)) < 2:
            raise ValueError(
                "Selectivity partition needs >=8 valid steps across >=2 episodes."
            )
        selection = _subsample_selection(representation.shape[0], max_samples, split_seed)
        if selection is not None:
            representation = representation[selection]
            positions = positions[selection]
            time_fraction = time_fraction[selection]
            episode_ids = episode_ids[selection]
            heading = None if heading is None else heading[selection]
            kinematics = None if kinematics is None else kinematics[selection]
            latent = None if latent is None else latent[selection]

        bounds = _resolve_position_bounds(
            str(analysis_input.metadata.get("env_id", "")), positions,
            analysis_input.metadata.get("env_kwargs"),
        )
        groups = self._build_groups(
            positions, heading, kinematics, time_fraction, latent, bounds, config
        )

        train_indices, validation_indices = episode_split_indices(
            episode_ids, train_fraction=train_fraction, random_seed=split_seed
        )
        targets_validation = representation[validation_indices]

        full_matrix, group_columns = _assemble_feature_matrix(groups)
        full_predictions = chunked_ridge_fit_predict(
            full_matrix, representation, train_indices, validation_indices, ridge_alpha
        )
        full_r2 = _per_output_r2(targets_validation, full_predictions)

        unique_r2: dict[str, np.ndarray] = {}
        for group_name, columns in group_columns.items():
            keep_columns = np.ones(full_matrix.shape[1], dtype=bool)
            keep_columns[columns] = False
            reduced_predictions = chunked_ridge_fit_predict(
                full_matrix[:, keep_columns],
                representation,
                train_indices,
                validation_indices,
                ridge_alpha,
            )
            reduced_r2 = _per_output_r2(targets_validation, reduced_predictions)
            unique_r2[group_name] = full_r2 - reduced_r2

        categories = _categorize_units(full_r2, unique_r2, min_r2=min_r2, dominance=mixed_dominance)
        metrics = _partition_metrics(full_r2, unique_r2, categories)
        figures = self._render_figure(output_dir, analysis_input, full_r2, unique_r2, categories)
        figures.update(
            self._render_decomposition_figure(output_dir, analysis_input, full_r2, unique_r2)
        )
        figures.update(
            self._render_unit_scatter_figure(output_dir, analysis_input, unique_r2, categories)
        )
        tables = {
            "selectivity_partition": _write_partition_table(
                output_dir, analysis_input, full_r2, unique_r2, categories
            )
        }
        per_unit_metrics: dict[str, np.ndarray] = {"full_r2": full_r2.astype(np.float32)}
        for group_name, values in unique_r2.items():
            per_unit_metrics[f"unique_r2_{group_name}"] = values.astype(np.float32)

        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics=per_unit_metrics,
            figures=figures,
            tables=tables,
            metadata={
                "selectivity_groups": list(group_columns.keys()),
                "selectivity_train_samples": int(len(train_indices)),
                "selectivity_validation_samples": int(len(validation_indices)),
                "selectivity_feature_dim": int(full_matrix.shape[1]),
            },
        )

    def _build_groups(self, positions, heading, kinematics, time_fraction, latent, bounds, config):
        groups: list[tuple[str, np.ndarray]] = [
            (
                "position",
                _build_position_basis(
                    positions,
                    bounds,
                    centers_x=int(config.get("selectivity_position_centers_x", 8)),
                    centers_y=int(config.get("selectivity_position_centers_y", 8)),
                    width_scale=float(config.get("selectivity_position_rbf_width_scale", 1.25)),
                ),
            )
        ]
        if heading is not None:
            harmonics = int(config.get("selectivity_heading_harmonics", 2))
            groups.append(("heading", _heading_features(heading, harmonics)))
        if kinematics is not None and kinematics.shape[1] >= 1:
            groups.append(("speed", _standardize(kinematics[:, 0:1])))
        if kinematics is not None and kinematics.shape[1] >= 2:
            groups.append(("angular_velocity", _standardize(kinematics[:, 1:2])))
        groups.append(("time", _standardize(time_fraction[:, None])))
        if latent is not None and latent.shape[1] >= 1:
            components = int(config.get("selectivity_visual_pca_components", 64))
            groups.append(("visual", _visual_pca_features(latent, components)))
        return groups

    def _render_figure(self, output_dir, analysis_input, full_r2, unique_r2, categories):
        module_dir = output_dir / self.name
        module_dir.mkdir(parents=True, exist_ok=True)
        figure_path = module_dir / (
            f"selectivity_partition__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        figure, (variance_axis, category_axis) = plt.subplots(1, 2, figsize=(11.0, 4.4))

        group_names = list(unique_r2.keys())
        mean_unique = [float(np.clip(unique_r2[name], 0.0, None).mean()) for name in group_names]
        bar_labels = [*group_names, "full_model"]
        bar_values = [*mean_unique, float(full_r2.mean())]
        colors = ["#1F77B4"] * len(group_names) + ["#2CA02C"]
        variance_axis.bar(np.arange(len(bar_labels)), bar_values, color=colors)
        variance_axis.set_xticks(
            np.arange(len(bar_labels)), labels=bar_labels, rotation=30, ha="right"
        )
        variance_axis.set_ylabel("mean cross-validated R^2")
        variance_axis.set_title("Unique variance explained per source")
        variance_axis.grid(axis="y", alpha=0.3)

        order = [
            "place_like",
            "hd_like",
            "speed_like",
            "angular_velocity_like",
            "time_like",
            "mixed",
            "untuned",
        ]
        present = [label for label in order if label in set(categories)]
        counts = [int(np.count_nonzero(categories == label)) for label in present]
        category_axis.bar(np.arange(len(present)), counts, color="#7F7F7F")
        category_axis.set_xticks(np.arange(len(present)), labels=present, rotation=30, ha="right")
        category_axis.set_ylabel("unit count")
        category_axis.set_title("Unit selectivity categories")
        category_axis.grid(axis="y", alpha=0.3)

        figure.suptitle(
            f"{analysis_input.source_name} mixed selectivity ({len(full_r2)} units)", fontsize=12
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)
        return {"selectivity_partition": figure_path}

    def _render_decomposition_figure(self, output_dir, analysis_input, full_r2, unique_r2):
        """Population variance decomposition: unique per source + shared + unexplained ~ 100%."""
        module_dir = output_dir / self.name
        module_dir.mkdir(parents=True, exist_ok=True)
        figure_path = module_dir / (
            f"selectivity_decomposition__{analysis_input.source_name}"
            f"__{analysis_input.split_name}.png"
        )
        label_map = {
            "position": "Position (x, y)",
            "heading": "Head direction",
            "speed": "Speed",
            "angular_velocity": "Angular velocity",
            "time": "Time",
            "visual": "Visual input (linear, lower bound)",
        }
        group_names = list(unique_r2.keys())
        mean_full = float(np.clip(full_r2.mean(), 0.0, 1.0))
        unique_means = {
            name: float(np.clip(unique_r2[name], 0.0, None).mean()) for name in group_names
        }
        sum_unique = sum(unique_means.values())
        rescaled = sum_unique > mean_full and sum_unique > 1e-12
        if rescaled:
            scale = mean_full / sum_unique
            unique_means = {name: value * scale for name, value in unique_means.items()}
            shared = 0.0
        else:
            shared = mean_full - sum_unique
        unexplained = 1.0 - mean_full
        labels = [label_map.get(name, name) for name in group_names] + ["Shared", "Unexplained"]
        values = np.array([*unique_means.values(), shared, unexplained], dtype=np.float64)
        colors = ["#1F77B4"] * len(group_names) + ["#9467BD", "#C7C7C7"]

        figure, axis = plt.subplots(figsize=(7.8, 0.5 * len(labels) + 1.8))
        y_positions = np.arange(len(labels))
        axis.barh(y_positions, values * 100.0, color=colors)
        axis.set_yticks(y_positions, labels=labels)
        axis.invert_yaxis()
        axis.set_xlabel("variance explained (% of code variance, held-out)")
        for y_position, value in zip(y_positions, values, strict=False):
            axis.text(
                value * 100.0 + 0.4, y_position, f"{value * 100.0:.0f}%", va="center", fontsize=8
            )
        axis.set_title(
            f"{analysis_input.source_name}: variance decomposition\n"
            "unique-per-source + shared (correlated inputs) + unexplained",
            fontsize=10,
        )
        axis.grid(axis="x", alpha=0.3)

        caveats = []
        if "visual" in group_names:
            caveats.append(
                "Visual = linear readout of the frozen AE latent (the code's own input "
                "substrate, not a behavioral confound); a lower bound on its true contribution."
            )
        if rescaled:
            caveats.append(
                "Unique bars rescaled to the full-model R^2 (sources are partly redundant)."
            )
        bottom_margin = 0.0
        if caveats:
            figure.text(0.5, 0.01, "\n".join(caveats), ha="center", va="bottom", fontsize=7)
            bottom_margin = 0.04 + 0.03 * len(caveats)
        figure.tight_layout(rect=(0.0, bottom_margin, 1.0, 1.0))
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)
        return {"selectivity_decomposition": figure_path}

    def _render_unit_scatter_figure(self, output_dir, analysis_input, unique_r2, categories):
        """Each unit positioned by spatial vs directional unique variance, colored by category."""
        if "heading" not in unique_r2:
            return {}
        module_dir = output_dir / self.name
        module_dir.mkdir(parents=True, exist_ok=True)
        figure_path = module_dir / (
            f"selectivity_unit_scatter__{analysis_input.source_name}"
            f"__{analysis_input.split_name}.png"
        )
        spatial = np.clip(unique_r2["position"], 0.0, None)
        directional = np.clip(unique_r2["heading"], 0.0, None)
        category_colors = {
            "place_like": "#1F77B4",
            "hd_like": "#FF7F0E",
            "speed_like": "#2CA02C",
            "angular_velocity_like": "#17BECF",
            "time_like": "#8C564B",
            "visual_like": "#9467BD",
            "mixed": "#7F7F7F",
            "untuned": "#D0D0D0",
        }
        figure, axis = plt.subplots(figsize=(6.2, 5.4))
        for label, color in category_colors.items():
            mask = categories == label
            if not np.any(mask):
                continue
            axis.scatter(
                spatial[mask], directional[mask], s=14, alpha=0.7, color=color, label=label
            )
        axis.set_xlabel("spatial unique variance (position ΔR²)")
        axis.set_ylabel("directional unique variance (heading ΔR²)")
        axis.set_title(f"{analysis_input.source_name}: unit selectivity")
        axis.legend(loc="upper right", fontsize=7, framealpha=0.9)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)
        return {"selectivity_unit_scatter": figure_path}


def _assemble_feature_matrix(
    groups: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Concatenate group feature blocks and record each group's column indices."""
    blocks = [block for _name, block in groups]
    matrix = np.concatenate(blocks, axis=1).astype(np.float32)
    group_columns: dict[str, np.ndarray] = {}
    cursor = 0
    for name, block in groups:
        group_columns[name] = np.arange(cursor, cursor + block.shape[1], dtype=np.int64)
        cursor += block.shape[1]
    return matrix, group_columns


def _categorize_units(
    full_r2: np.ndarray,
    unique_r2: dict[str, np.ndarray],
    *,
    min_r2: float,
    dominance: float,
) -> np.ndarray:
    """Label each unit by its dominant unique driver, or mixed / untuned."""
    group_names = list(unique_r2.keys())
    clipped = np.stack([np.clip(unique_r2[name], 0.0, None) for name in group_names], axis=1)
    total_unique = clipped.sum(axis=1)
    top_index = clipped.argmax(axis=1)
    top_value = clipped[np.arange(len(full_r2)), top_index]
    with np.errstate(invalid="ignore", divide="ignore"):
        top_share = np.where(total_unique > 1e-12, top_value / total_unique, 0.0)

    categories = np.empty(len(full_r2), dtype=object)
    for unit in range(len(full_r2)):
        if full_r2[unit] < min_r2 or total_unique[unit] <= 1e-9:
            categories[unit] = "untuned"
        elif top_share[unit] >= dominance:
            categories[unit] = _GROUP_LABELS[group_names[top_index[unit]]]
        else:
            categories[unit] = "mixed"
    return categories


def _partition_metrics(full_r2, unique_r2, categories) -> dict[str, float]:
    metrics = {
        "mean_full_r2": float(full_r2.mean()),
        "median_full_r2": float(np.median(full_r2)),
        "num_units": float(len(full_r2)),
    }
    for group_name, values in unique_r2.items():
        metrics[f"mean_unique_r2_{group_name}"] = float(np.clip(values, 0.0, None).mean())
    unit_count = max(len(full_r2), 1)
    for label in set(_GROUP_LABELS.values()) | {"mixed", "untuned"}:
        metrics[f"fraction_{label}"] = float(np.count_nonzero(categories == label)) / unit_count
    return metrics


def _write_partition_table(output_dir, analysis_input, full_r2, unique_r2, categories) -> Path:
    group_names = list(unique_r2.keys())
    header = ["unit_index", "full_r2", *[f"unique_r2_{name}" for name in group_names], "category"]
    rows = [
        [
            unit,
            float(full_r2[unit]),
            *[float(unique_r2[name][unit]) for name in group_names],
            str(categories[unit]),
        ]
        for unit in range(len(full_r2))
    ]
    return write_csv(
        output_dir
        / "selectivity_partition"
        / f"selectivity_partition__{analysis_input.source_name}__{analysis_input.split_name}.csv",
        header,
        rows,
    )


@dataclass(slots=True)
class SpatialCodeTypeModule:
    """Sufficiency screen: does wall/goal distance match the place basis (parsimony)?"""

    name: str = "spatial_code_type"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        ridge_alpha = float(config.get("selectivity_ridge_alpha", 1e-2))
        train_fraction = float(config.get("selectivity_train_fraction", 0.8))
        split_seed = int(config.get("selectivity_split_seed", 0))
        min_r2 = float(config.get("spatial_code_min_r2", 0.02))
        sufficiency_threshold = float(config.get("spatial_code_sufficiency_threshold", 0.8))
        max_samples = int(config.get("selectivity_max_samples", 150_000))

        env_id = str(analysis_input.metadata.get("env_id", ""))
        representation, positions, _heading, _kinematics, _time, episode_ids, _latent = (
            _flatten_valid_rows(analysis_input)
        )
        if representation.shape[0] < 8 or len(np.unique(episode_ids)) < 2:
            raise ValueError("Spatial code-type comparison needs >=8 steps across >=2 episodes.")
        selection = _subsample_selection(representation.shape[0], max_samples, split_seed)
        if selection is not None:
            representation = representation[selection]
            positions = positions[selection]
            episode_ids = episode_ids[selection]

        models: dict[str, np.ndarray] = {
            "allocentric_place": _build_position_basis(
                positions,
                _resolve_position_bounds(
                    env_id, positions, analysis_input.metadata.get("env_kwargs")
                ),
                centers_x=int(config.get("selectivity_position_centers_x", 8)),
                centers_y=int(config.get("selectivity_position_centers_y", 8)),
                width_scale=float(config.get("selectivity_position_rbf_width_scale", 1.25)),
            )
        }
        overlay = resolve_world_overlay(env_id, analysis_input.metadata.get("env_kwargs"))
        if overlay is not None and overlay.segments:
            wall_distance = distance_to_segments(
                positions[:, None, :], overlay.segments
            ).reshape(-1)
            models["boundary_vector"] = _scalar_with_square(wall_distance)
            goal_points = _goal_or_landmark_points(overlay)
            if goal_points is not None:
                goal_distance = distance_to_points(positions[:, None, :], goal_points).reshape(-1)
                models["goal_vector"] = _scalar_with_square(goal_distance)

        train_indices, validation_indices = episode_split_indices(
            episode_ids, train_fraction=train_fraction, random_seed=split_seed
        )
        targets_validation = representation[validation_indices]
        model_r2 = {
            name: _per_output_r2(
                targets_validation,
                chunked_ridge_fit_predict(
                    features, representation, train_indices, validation_indices, ridge_alpha
                ),
            )
            for name, features in models.items()
        }

        place_r2 = model_r2["allocentric_place"]
        spatial = place_r2 >= min_r2
        spatial_count = max(int(np.count_nonzero(spatial)), 1)
        safe_place = np.maximum(place_r2, 1e-6)

        sufficiency = {
            name: (np.clip(values, 0.0, None) / safe_place).astype(np.float64)
            for name, values in model_r2.items()
            if name != "allocentric_place"
        }

        metrics = {"num_spatial_units": float(np.count_nonzero(spatial))}
        for name, values in model_r2.items():
            metrics[f"mean_r2_{name}"] = float(values.mean())
        sufficiency_fraction: dict[str, float] = {}
        for name, ratio in sufficiency.items():
            sufficient_count = float(np.count_nonzero(spatial & (ratio >= sufficiency_threshold)))
            sufficiency_fraction[name] = sufficient_count / spatial_count
            metrics[f"fraction_{name}_sufficient"] = sufficiency_fraction[name]

        figures = self._render_figure(
            output_dir, analysis_input, model_r2, sufficiency_fraction
        )
        tables = {
            "spatial_code_type": _write_code_type_table(
                output_dir, analysis_input, model_r2, sufficiency, spatial
            )
        }
        per_unit_metrics = {
            f"r2_{name}": values.astype(np.float32) for name, values in model_r2.items()
        }
        for name, ratio in sufficiency.items():
            per_unit_metrics[f"{name}_sufficiency"] = ratio.astype(np.float32)
        per_unit_metrics["is_spatial"] = spatial.astype(np.float32)
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics=per_unit_metrics,
            figures=figures,
            tables=tables,
            metadata={
                "spatial_code_models": list(model_r2.keys()),
                "spatial_code_sufficiency_threshold": sufficiency_threshold,
            },
        )

    def _render_figure(self, output_dir, analysis_input, model_r2, sufficiency_fraction) -> dict:
        module_dir = output_dir / self.name
        module_dir.mkdir(parents=True, exist_ok=True)
        figure_path = module_dir / (
            f"spatial_code_type__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        model_names = list(model_r2.keys())
        figure, (r2_axis, sufficiency_axis) = plt.subplots(1, 2, figsize=(10.0, 4.2))

        r2_axis.bar(
            np.arange(len(model_names)),
            [float(model_r2[name].mean()) for name in model_names],
            color="#1F77B4",
        )
        r2_axis.set_xticks(np.arange(len(model_names)), labels=model_names, rotation=20, ha="right")
        r2_axis.set_ylabel("mean cross-validated R^2")
        r2_axis.set_title("Spatial model fit")
        r2_axis.grid(axis="y", alpha=0.3)

        if sufficiency_fraction:
            names = list(sufficiency_fraction.keys())
            r2_axis.set_title("Spatial model fit (place = 64-param superset)")
            sufficiency_axis.bar(
                np.arange(len(names)),
                [sufficiency_fraction[name] for name in names],
                color="#7F7F7F",
            )
            sufficiency_axis.set_xticks(
                np.arange(len(names)), labels=names, rotation=20, ha="right"
            )
            sufficiency_axis.set_ylim(0.0, 1.0)
            sufficiency_axis.set_ylabel("fraction of spatial units sufficient")
            sufficiency_axis.set_title("Distance model ~ matches place fit")
            sufficiency_axis.grid(axis="y", alpha=0.3)
        else:
            sufficiency_axis.text(
                0.5, 0.5, "no wall/goal geometry\n(place model only)",
                ha="center", va="center", fontsize=10,
            )
            sufficiency_axis.set_axis_off()

        figure.suptitle(f"{analysis_input.source_name} spatial code type", fontsize=12)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)
        return {"spatial_code_type": figure_path}


def _goal_or_landmark_points(overlay) -> tuple[tuple[float, float], ...] | None:
    """Goal-layer points if present, else every landmark point (object proximity)."""
    goal_points: list[tuple[float, float]] = []
    all_points: list[tuple[float, float]] = []
    for layer in overlay.landmarks:
        all_points.extend(layer.positions)
        if "goal" in layer.label.lower():
            goal_points.extend(layer.positions)
    if goal_points:
        return tuple(goal_points)
    return tuple(all_points) if all_points else None


def _scalar_with_square(values: np.ndarray) -> np.ndarray:
    """Standardised scalar plus its square: a coarse distance-tuning curve."""
    standardized = _standardize(values[:, None])
    return np.concatenate([standardized, _standardize(standardized * standardized)], axis=1)


def _write_code_type_table(output_dir, analysis_input, model_r2, sufficiency, spatial) -> Path:
    model_names = list(model_r2.keys())
    sufficiency_names = list(sufficiency.keys())
    header = [
        "unit_index",
        *[f"r2_{name}" for name in model_names],
        *[f"{name}_sufficiency" for name in sufficiency_names],
        "is_spatial",
    ]
    rows = [
        [
            unit,
            *[float(model_r2[name][unit]) for name in model_names],
            *[float(sufficiency[name][unit]) for name in sufficiency_names],
            bool(spatial[unit]),
        ]
        for unit in range(len(spatial))
    ]
    return write_csv(
        output_dir
        / "spatial_code_type"
        / f"spatial_code_type__{analysis_input.source_name}__{analysis_input.split_name}.csv",
        header,
        rows,
    )
