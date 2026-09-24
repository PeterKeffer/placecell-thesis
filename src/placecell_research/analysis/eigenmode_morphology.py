"""Classify transition eigenmodes and learned rate maps as bands or checkerboards."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..numerics.rate_map_kernels import (
    compute_spatial_bin_assignments,
    flatten_positions,
)
from .base import AnalysisInput, AnalysisResult
from .helpers import get_or_compute_rate_maps
from .reanchoring_gridness import benjamini_hochberg
from .transition_geometry import (
    _build_transition_counts,
    _flatten_valid_episode_bins,
    _transition_laplacian_modes,
)
from .world_overlay import (
    apply_plot_bounds,
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    overlay_bounds,
    resolve_world_overlay,
    style_arena_axes,
)

VERTICAL_BAND = 1
HORIZONTAL_BAND = 2
CHECKERBOARD = 3

_KIND_LABELS = {
    VERTICAL_BAND: "vertical_band",
    HORIZONTAL_BAND: "horizontal_band",
    CHECKERBOARD: "checkerboard",
}


@dataclass(frozen=True, slots=True)
class CosineTemplateBank:
    values: np.ndarray
    kind_codes: np.ndarray
    frequencies_x: np.ndarray
    frequencies_y: np.ndarray


@dataclass(frozen=True, slots=True)
class ModeGroupScore:
    first_mode_rank: int
    last_mode_rank: int
    eigenvalue_min: float
    eigenvalue_max: float
    vertical_band_score: float
    horizontal_band_score: float
    checkerboard_score: float
    best_score: float
    best_kind_code: int
    best_frequency_x: int
    best_frequency_y: int
    shuffle_p_value: float
    significant_after_fdr: bool = False


def cosine_template_bank(
    shape: tuple[int, int],
    visited_mask: np.ndarray,
    *,
    max_frequency: int,
) -> CosineTemplateBank:
    """Create boundary-anchored rectangular Laplacian templates on visited bins."""
    height, width = shape
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    templates: list[np.ndarray] = []
    kind_codes: list[int] = []
    frequencies_x: list[int] = []
    frequencies_y: list[int] = []

    def append_template(frequency_x: int, frequency_y: int, kind_code: int) -> None:
        template = np.outer(
            np.cos(np.pi * frequency_y * y),
            np.cos(np.pi * frequency_x * x),
        )[visited_mask]
        template = template - template.mean()
        norm = float(np.linalg.norm(template))
        if norm <= 1e-12:
            return
        templates.append(template / norm)
        kind_codes.append(kind_code)
        frequencies_x.append(frequency_x)
        frequencies_y.append(frequency_y)

    for frequency in range(1, max_frequency + 1):
        append_template(frequency, 0, VERTICAL_BAND)
        append_template(0, frequency, HORIZONTAL_BAND)
    for frequency_x in range(1, max_frequency + 1):
        for frequency_y in range(1, max_frequency + 1):
            append_template(frequency_x, frequency_y, CHECKERBOARD)

    if not templates:
        raise ValueError("Eigenmode morphology could not build any nonconstant cosine templates.")
    return CosineTemplateBank(
        values=np.asarray(templates, dtype=np.float64),
        kind_codes=np.asarray(kind_codes, dtype=np.int32),
        frequencies_x=np.asarray(frequencies_x, dtype=np.int32),
        frequencies_y=np.asarray(frequencies_y, dtype=np.int32),
    )


def _normalized_map_values(spatial_maps: np.ndarray, visited_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(spatial_maps, dtype=np.float64)[:, visited_mask]
    values = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def score_cosine_morphology(
    spatial_maps: np.ndarray,
    visited_mask: np.ndarray | None = None,
    *,
    max_frequency: int = 4,
) -> dict[str, np.ndarray]:
    """Score fixed-phase x-bands, y-bands, and separable checkerboards."""
    maps = np.asarray(spatial_maps, dtype=np.float64)
    if maps.ndim == 2:
        maps = maps[None, ...]
    if maps.ndim != 3:
        raise ValueError(f"Expected [map, y, x] spatial maps, got shape {maps.shape}.")
    if visited_mask is None:
        visited_mask = np.all(np.isfinite(maps), axis=0)
    visited_mask = np.asarray(visited_mask, dtype=bool)
    if visited_mask.shape != maps.shape[1:]:
        raise ValueError(
            f"visited_mask shape {visited_mask.shape} does not match maps {maps.shape[1:]}."
        )
    if int(visited_mask.sum()) < 4:
        raise ValueError("Eigenmode morphology needs at least four commonly visited bins.")

    bank = cosine_template_bank(
        maps.shape[1:],
        visited_mask,
        max_frequency=max_frequency,
    )
    normalized = _normalized_map_values(maps, visited_mask)
    correlations = np.abs(normalized @ bank.values.T)

    category_scores = {}
    for kind_code, name in (
        (VERTICAL_BAND, "vertical_band_score"),
        (HORIZONTAL_BAND, "horizontal_band_score"),
        (CHECKERBOARD, "checkerboard_score"),
    ):
        category_scores[name] = correlations[:, bank.kind_codes == kind_code].max(axis=1)

    best_template_indices = np.argmax(correlations, axis=1)
    return {
        **category_scores,
        "best_score": correlations[np.arange(len(maps)), best_template_indices],
        "best_kind_code": bank.kind_codes[best_template_indices],
        "best_frequency_x": bank.frequencies_x[best_template_indices],
        "best_frequency_y": bank.frequencies_y[best_template_indices],
    }


def _group_near_degenerate_modes(
    eigenvalues: np.ndarray,
    *,
    relative_tolerance: float,
) -> list[np.ndarray]:
    groups: list[list[int]] = []
    for mode_index, eigenvalue in enumerate(np.asarray(eigenvalues, dtype=np.float64)):
        if not groups:
            groups.append([mode_index])
            continue
        previous = float(eigenvalues[groups[-1][-1]])
        scale = max(abs(previous), abs(float(eigenvalue)), 1e-12)
        if abs(float(eigenvalue) - previous) <= relative_tolerance * scale:
            groups[-1].append(mode_index)
        else:
            groups.append([mode_index])
    return [np.asarray(group, dtype=np.int32) for group in groups]


def _subspace_template_scores(
    mode_vectors: np.ndarray,
    bank: CosineTemplateBank,
) -> tuple[np.ndarray, int]:
    centered = mode_vectors - mode_vectors.mean(axis=0, keepdims=True)
    basis, _ = np.linalg.qr(centered)
    projection = basis.T @ bank.values.T
    scores = np.sqrt(np.square(projection).sum(axis=0))
    best_template_index = int(np.argmax(scores))
    return scores, best_template_index


def _shuffle_map_p_values(
    normalized_maps: np.ndarray,
    observed_scores: np.ndarray,
    candidate_indices: np.ndarray,
    bank: CosineTemplateBank,
    *,
    shuffle_count: int,
    seed: int,
) -> np.ndarray:
    p_values = np.full(len(normalized_maps), np.nan, dtype=np.float64)
    if shuffle_count <= 0 or candidate_indices.size == 0:
        return p_values
    rng = np.random.default_rng(seed)
    exceedance_counts = np.zeros(candidate_indices.size, dtype=np.int64)
    candidates = normalized_maps[candidate_indices]
    for _ in range(shuffle_count):
        shuffled = np.asarray([rng.permutation(values) for values in candidates])
        null_scores = np.abs(shuffled @ bank.values.T).max(axis=1)
        exceedance_counts += null_scores >= observed_scores[candidate_indices]
    p_values[candidate_indices] = (exceedance_counts + 1.0) / (shuffle_count + 1.0)
    return p_values


def _mode_group_scores(
    mode_maps: np.ndarray,
    eigenvalues: np.ndarray,
    visited_mask: np.ndarray,
    bank: CosineTemplateBank,
    *,
    relative_tolerance: float,
    shuffle_count: int,
    seed: int,
    fdr_alpha: float,
) -> list[ModeGroupScore]:
    mode_values = np.asarray(mode_maps, dtype=np.float64)[:, visited_mask].T
    groups = _group_near_degenerate_modes(
        eigenvalues,
        relative_tolerance=relative_tolerance,
    )
    rng = np.random.default_rng(seed)
    rows: list[ModeGroupScore] = []
    for group in groups:
        vectors = mode_values[:, group]
        template_scores, best_template_index = _subspace_template_scores(vectors, bank)
        observed = float(template_scores[best_template_index])
        exceedance_count = 0
        for _ in range(shuffle_count):
            permutation = rng.permutation(len(vectors))
            shuffled_scores, _ = _subspace_template_scores(vectors[permutation], bank)
            exceedance_count += float(shuffled_scores.max()) >= observed
        p_value = (
            float((exceedance_count + 1.0) / (shuffle_count + 1.0))
            if shuffle_count > 0
            else float("nan")
        )

        rows.append(
            ModeGroupScore(
                first_mode_rank=int(group[0]) + 1,
                last_mode_rank=int(group[-1]) + 1,
                eigenvalue_min=float(eigenvalues[group].min()),
                eigenvalue_max=float(eigenvalues[group].max()),
                vertical_band_score=float(template_scores[bank.kind_codes == VERTICAL_BAND].max()),
                horizontal_band_score=float(
                    template_scores[bank.kind_codes == HORIZONTAL_BAND].max()
                ),
                checkerboard_score=float(template_scores[bank.kind_codes == CHECKERBOARD].max()),
                best_score=observed,
                best_kind_code=int(bank.kind_codes[best_template_index]),
                best_frequency_x=int(bank.frequencies_x[best_template_index]),
                best_frequency_y=int(bank.frequencies_y[best_template_index]),
                shuffle_p_value=p_value,
            )
        )
    if shuffle_count <= 0:
        return rows
    significant = benjamini_hochberg(
        np.asarray([row.shuffle_p_value for row in rows]),
        alpha=fdr_alpha,
    )
    return [
        ModeGroupScore(
            **{
                field_name: getattr(row, field_name)
                for field_name in ModeGroupScore.__dataclass_fields__
                if field_name != "significant_after_fdr"
            },
            significant_after_fdr=bool(significant[row_index]),
        )
        for row_index, row in enumerate(rows)
    ]


def _write_mode_group_table(
    output_dir: Path,
    analysis_input: AnalysisInput,
    rows: list[ModeGroupScore],
) -> Path:
    table_path = (
        output_dir
        / "eigenmode_morphology"
        / f"mode_groups__{analysis_input.source_name}__{analysis_input.split_name}.csv"
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(ModeGroupScore.__dataclass_fields__)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names + ["best_kind"])
        writer.writeheader()
        for row in rows:
            values = {field_name: getattr(row, field_name) for field_name in field_names}
            values["best_kind"] = _KIND_LABELS[row.best_kind_code]
            writer.writerow(values)
    return table_path


def _render_panel(
    output_dir: Path,
    analysis_input: AnalysisInput,
    mode_maps: np.ndarray,
    mode_scores: dict[str, np.ndarray],
    learned_maps: np.ndarray,
    learned_scores: dict[str, np.ndarray],
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    render_top_k: int,
) -> Path:
    world_overlay = resolve_world_overlay(
        str(analysis_input.metadata.get("env_id", "")),
        analysis_input.metadata.get("env_kwargs"),
    )
    top_indices = np.argsort(learned_scores["best_score"])[::-1][
        : min(render_top_k, len(learned_maps))
    ]
    columns = max(1, min(render_top_k, max(len(mode_maps), len(top_indices))))
    figure, axes = plt.subplots(2, columns, figsize=(3.4 * columns, 6.8), squeeze=False)
    valid_positions = flatten_positions(analysis_input.position_xy, analysis_input.valid_mask)
    x_bounds, y_bounds = bounds

    def render_map(axis: plt.Axes, spatial_map: np.ndarray, title: str) -> None:
        finite = np.isfinite(spatial_map)
        values = spatial_map[finite]
        scale = float(np.max(np.abs(values))) if values.size else 1.0
        axis.imshow(
            np.ma.masked_where(~finite, spatial_map),
            origin="lower",
            cmap="coolwarm",
            vmin=-max(scale, 1e-6),
            vmax=max(scale, 1e-6),
            extent=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]),
            aspect="equal",
        )
        if world_overlay is not None:
            draw_world_segments_on_axis(axis, world_overlay.segments, line_color="#202020")
            draw_landmarks_on_axis(axis, world_overlay)
        apply_plot_bounds(
            axis,
            env_id=str(analysis_input.metadata.get("env_id", "")),
            position_xy=valid_positions,
        )
        axis.set_title(title, fontsize=9)
        style_arena_axes(axis)

    for column in range(columns):
        if column < len(mode_maps):
            kind_code = int(mode_scores["best_kind_code"][column])
            render_map(
                axes[0, column],
                mode_maps[column],
                f"mode {column + 1}: {_KIND_LABELS[kind_code]}\n"
                f"score={mode_scores['best_score'][column]:.2f}, "
                f"k=({int(mode_scores['best_frequency_x'][column])},"
                f"{int(mode_scores['best_frequency_y'][column])})",
            )
        else:
            axes[0, column].axis("off")
        if column < len(top_indices):
            unit_index = int(top_indices[column])
            kind_code = int(learned_scores["best_kind_code"][unit_index])
            render_map(
                axes[1, column],
                learned_maps[unit_index],
                f"unit {unit_index}: {_KIND_LABELS[kind_code]}\n"
                f"score={learned_scores['best_score'][unit_index]:.2f}, "
                f"k=({int(learned_scores['best_frequency_x'][unit_index])},"
                f"{int(learned_scores['best_frequency_y'][unit_index])})",
            )
        else:
            axes[1, column].axis("off")
    axes[0, 0].set_ylabel("transition modes")
    axes[1, 0].set_ylabel("learned units")
    figure.suptitle(f"{analysis_input.source_name}: eigenmode morphology", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure_path = (
        output_dir
        / "eigenmode_morphology"
        / f"eigenmode_morphology__{analysis_input.source_name}__"
        f"{analysis_input.split_name}.png"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


@dataclass(slots=True)
class EigenmodeMorphologyModule:
    """Score transition modes and learned units against rectangular cosine families."""

    name: str = "eigenmode_morphology"
    cost_tier: str = "heavy"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        num_bins_x = int(config.get("transition_geometry_num_bins_x", 24))
        num_bins_y = int(config.get("transition_geometry_num_bins_y", 24))
        num_modes = int(config.get("transition_geometry_num_modes", 6))
        include_self_transitions = bool(
            config.get("transition_geometry_include_self_transitions", False)
        )
        max_frequency = int(config.get("eigenmode_morphology_max_frequency", 4))
        shuffle_count = int(config.get("eigenmode_morphology_shuffle_count", 999))
        shuffle_top_k = int(config.get("eigenmode_morphology_shuffle_top_k", 32))
        fdr_alpha = float(config.get("eigenmode_morphology_fdr_alpha", 0.05))
        seed = int(config.get("eigenmode_morphology_random_seed", 0))
        relative_tolerance = float(config.get("eigenmode_morphology_degeneracy_tolerance", 0.05))
        render_top_k = int(config.get("eigenmode_morphology_render_top_k", 6))

        valid_positions = flatten_positions(analysis_input.position_xy, analysis_input.valid_mask)
        if valid_positions.size == 0:
            raise ValueError("Eigenmode morphology needs at least one valid position.")
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
        if bounds is None:
            _bins, _x_edges, _y_edges, bounds = compute_spatial_bin_assignments(
                valid_positions,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
            )
        episode_bins, occupancy_counts = _flatten_valid_episode_bins(
            analysis_input.position_xy,
            analysis_input.valid_mask,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            bounds=bounds,
        )
        transition_counts, transitions_used = _build_transition_counts(
            episode_bins,
            num_bins=num_bins_x * num_bins_y,
            include_self_transitions=include_self_transitions,
        )
        if transitions_used <= 0:
            raise ValueError("Eigenmode morphology found no valid spatial transitions.")
        eigenvalues, _mode_indices, mode_maps = _transition_laplacian_modes(
            transition_counts,
            occupancy_counts=occupancy_counts,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            num_modes=num_modes,
        )
        rate_maps = get_or_compute_rate_maps(
            analysis_input,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=float(config["smoothing_sigma"]),
            min_occupancy=float(config.get("min_occupancy", 1e-6)),
            bounds=bounds,
        ).rate_maps
        visited_mask = np.all(np.isfinite(mode_maps), axis=0)
        mode_scores = score_cosine_morphology(
            mode_maps,
            visited_mask,
            max_frequency=max_frequency,
        )
        learned_scores = score_cosine_morphology(
            rate_maps,
            visited_mask,
            max_frequency=max_frequency,
        )
        bank = cosine_template_bank(
            mode_maps.shape[1:],
            visited_mask,
            max_frequency=max_frequency,
        )
        normalized_learned = _normalized_map_values(rate_maps, visited_mask)
        candidate_count = min(max(shuffle_top_k, 1), len(rate_maps))
        candidate_indices = np.argsort(learned_scores["best_score"])[::-1][:candidate_count]
        p_values = _shuffle_map_p_values(
            normalized_learned,
            learned_scores["best_score"],
            candidate_indices,
            bank,
            shuffle_count=shuffle_count,
            seed=seed,
        )
        significant = np.zeros(len(rate_maps), dtype=bool)
        if shuffle_count > 0:
            significant[candidate_indices] = benjamini_hochberg(
                p_values[candidate_indices],
                alpha=fdr_alpha,
            )
        mode_groups = _mode_group_scores(
            mode_maps,
            eigenvalues,
            visited_mask,
            bank,
            relative_tolerance=relative_tolerance,
            shuffle_count=shuffle_count,
            seed=seed + 1,
            fdr_alpha=fdr_alpha,
        )
        table_path = _write_mode_group_table(output_dir, analysis_input, mode_groups)
        figure_path = _render_panel(
            output_dir,
            analysis_input,
            mode_maps,
            mode_scores,
            rate_maps,
            learned_scores,
            bounds=bounds,
            render_top_k=render_top_k,
        )

        repeated_band = np.isin(
            learned_scores["best_kind_code"],
            [VERTICAL_BAND, HORIZONTAL_BAND],
        ) & ((learned_scores["best_frequency_x"] >= 2) | (learned_scores["best_frequency_y"] >= 2))
        metrics = {
            "max_vertical_band_cosine_score": float(learned_scores["vertical_band_score"].max()),
            "max_horizontal_band_cosine_score": float(
                learned_scores["horizontal_band_score"].max()
            ),
            "max_checkerboard_cosine_score": float(learned_scores["checkerboard_score"].max()),
            "num_morphology_candidates_tested": float(candidate_count),
            "num_morphology_significant_after_fdr": float(significant.sum()),
            "num_significant_repeated_band_candidates": float(
                np.count_nonzero(significant & repeated_band)
            ),
            "transition_mode_group_count": float(len(mode_groups)),
            "transition_mode_groups_significant_after_fdr": float(
                sum(row.significant_after_fdr for row in mode_groups)
            ),
            "max_transition_mode_group_cosine_score": float(
                max((row.best_score for row in mode_groups), default=0.0)
            ),
        }
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={
                **{
                    key: np.asarray(value, dtype=np.float32)
                    for key, value in learned_scores.items()
                },
                "morphology_shuffle_p_value": p_values.astype(np.float32),
                "morphology_significant_after_fdr": significant.astype(np.float32),
                "repeated_band_candidate": repeated_band.astype(np.float32),
            },
            figures={"eigenmode_morphology": figure_path},
            tables={"eigenmode_morphology_mode_groups": table_path},
            metadata={
                "morphology_kind_codes": _KIND_LABELS,
                "eigenmode_morphology_max_frequency": max_frequency,
                "eigenmode_morphology_shuffle_count": shuffle_count,
                "eigenmode_morphology_shuffle_top_k": shuffle_top_k,
                "eigenmode_morphology_fdr_alpha": fdr_alpha,
                "eigenmode_morphology_degeneracy_tolerance": relative_tolerance,
                "transition_eigenvalues": eigenvalues.tolist(),
                "visited_bin_count": int(visited_mask.sum()),
            },
        )
