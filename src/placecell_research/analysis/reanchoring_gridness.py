"""Re-anchoring-aware grid-cell battery."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from placecell_research.config.schema import analysis_config_default

from ..numerics.rate_map_kernels import (
    autocorrelogram,
    compute_spatial_bin_assignments,
    flatten_positions,
    gridness_from_autocorrelogram,
    gridness_score,
    infer_bounds,
)
from .base import AnalysisInput, AnalysisResult
from .figures import save_figure
from .helpers import write_csv
from .world_overlay import overlay_bounds, resolve_world_overlay

_EPS = 1e-9


def average_autocorrelograms(rate_maps: np.ndarray) -> np.ndarray:
    """Mean of the per-map (already max-normalized) autocorrelograms."""
    rate_maps = np.asarray(rate_maps, dtype=np.float64)
    accumulator: np.ndarray | None = None
    count = 0
    for rate_map in rate_maps:
        rate_map = np.nan_to_num(rate_map, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.any(rate_map):
            continue
        autocorr = autocorrelogram(rate_map)
        accumulator = autocorr if accumulator is None else accumulator + autocorr
        count += 1
    if accumulator is None or count == 0:
        return np.zeros(rate_maps.shape[1:], dtype=np.float64)
    return accumulator / float(count)


def _radial_profile(autocorr, inner_radius, outer_radius, reducer):
    """Annulus-reduced profile vs integer radius."""
    center_y, center_x = np.array(autocorr.shape) // 2
    yy, xx = np.indices(autocorr.shape)
    radius_int = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2).astype(int)
    max_radius = int(min(outer_radius, radius_int.max()))
    radii: list[int] = []
    values: list[float] = []
    for r in range(int(inner_radius), max_radius + 1):
        annulus = autocorr[radius_int == r]
        if annulus.size == 0:
            continue
        radii.append(r)
        values.append(float(reducer(annulus)))
    if len(radii) < 2:
        return None
    return np.asarray(radii), np.asarray(values)


def _dominant_ring(autocorr, inner_radius, outer_radius, reducer):
    """Radius and prominence of the first periodic ring (trough then peak)."""
    result = _radial_profile(autocorr, inner_radius, outer_radius, reducer)
    if result is None:
        return None
    radii, profile = result
    trough_local = int(np.argmin(profile))
    after = profile[trough_local:]
    if after.size < 2:
        return None
    ring_local = trough_local + int(np.argmax(after))
    prominence = float(profile[ring_local] - profile[trough_local])
    return int(radii[ring_local]), prominence


def radial_periodicity_score(autocorr: np.ndarray) -> float:
    inner_radius = 3.0
    outer_radius = min(autocorr.shape) / 2.5
    ring = _dominant_ring(autocorr, inner_radius, outer_radius, reducer=np.mean)
    if ring is None:
        return 0.0
    return max(0.0, ring[1])


def angular_harmonics(autocorr: np.ndarray) -> tuple[float, float]:
    inner_radius = 3.0
    outer_radius = min(autocorr.shape) / 2.5
    ring = _dominant_ring(autocorr, inner_radius, outer_radius, reducer=np.max)
    if ring is None:
        return 0.0, 0.0
    ring_radius = float(ring[0])
    center_y, center_x = np.array(autocorr.shape) // 2
    angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    sample_y = center_y + ring_radius * np.sin(angles)
    sample_x = center_x + ring_radius * np.cos(angles)
    profile = map_coordinates(autocorr, [sample_y, sample_x], order=1, mode="nearest")
    profile = profile - profile.mean()
    spectrum = np.abs(np.fft.rfft(profile))
    total = float(spectrum[1:].sum()) + _EPS
    stripe_score = float(spectrum[2]) / total if spectrum.size > 2 else 0.0
    lattice_score = float(spectrum[6]) / total if spectrum.size > 6 else 0.0
    return stripe_score, lattice_score


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    finite = np.isfinite(pvalues)
    reject = np.zeros(pvalues.shape, dtype=bool)
    count = int(np.count_nonzero(finite))
    if count == 0:
        return reject
    indices = np.flatnonzero(finite)
    order = indices[np.argsort(pvalues[indices])]
    thresholds = (np.arange(1, count + 1) / count) * alpha
    below = pvalues[order] <= thresholds
    if not np.any(below):
        return reject
    largest = int(np.flatnonzero(below)[-1])
    reject[order[: largest + 1]] = True
    return reject


@dataclass(slots=True)
class _EpisodeBins:
    episode_index: int
    activations: np.ndarray
    linear_bins: np.ndarray
    occupancy: np.ndarray
    occupancy_mask: np.ndarray


def _collect_episode_bins(
    analysis_input: AnalysisInput,
    *,
    num_bins_x: int,
    num_bins_y: int,
    smoothing_sigma: float,
    minimum_valid_steps: int,
    minimum_visited_fraction: float,
    max_episodes: int,
    bounds,
) -> tuple[list[_EpisodeBins], dict[str, int]]:
    num_episodes = analysis_input.representation.shape[0]
    total_bins = num_bins_x * num_bins_y
    kept: list[_EpisodeBins] = []
    skipped_short = 0
    skipped_sparse = 0
    capped = 0
    for episode_index in range(num_episodes):
        if max_episodes > 0 and len(kept) >= max_episodes:
            capped = 1
            break
        valid_mask = analysis_input.valid_mask[episode_index].astype(bool, copy=False)
        valid_steps = int(np.count_nonzero(valid_mask))
        if valid_steps < minimum_valid_steps:
            skipped_short += 1
            continue
        positions = analysis_input.position_xy[episode_index][valid_mask]
        linear_bins, _, _, _ = compute_spatial_bin_assignments(
            positions, num_bins_x=num_bins_x, num_bins_y=num_bins_y, bounds=bounds
        )
        raw_occupancy = (
            np.bincount(linear_bins, minlength=total_bins)
            .reshape(num_bins_y, num_bins_x)
            .astype(np.float64)
        )
        visited_fraction = float(np.count_nonzero(raw_occupancy > 0.0)) / max(total_bins, 1)
        if visited_fraction < minimum_visited_fraction:
            skipped_sparse += 1
            continue
        occupancy = (
            gaussian_filter(raw_occupancy, sigma=smoothing_sigma)
            if smoothing_sigma > 0.0
            else raw_occupancy
        )
        activations = np.clip(
            analysis_input.representation[episode_index][valid_mask], 0.0, None
        ).astype(np.float32, copy=False)
        kept.append(
            _EpisodeBins(
                episode_index=episode_index,
                activations=activations,
                linear_bins=linear_bins.astype(np.int64, copy=False),
                occupancy=occupancy,
                occupancy_mask=raw_occupancy > 0.0,
            )
        )
    stats = {
        "episodes_used": len(kept),
        "episodes_skipped_short": skipped_short,
        "episodes_skipped_sparse": skipped_sparse,
        "episodes_capped": capped,
    }
    return kept, stats


def _unit_episode_maps(
    unit_index: int,
    episodes: list[_EpisodeBins],
    *,
    num_bins_x: int,
    num_bins_y: int,
    smoothing_sigma: float,
    min_occupancy: float,
    activation_override: list[np.ndarray] | None = None,
) -> np.ndarray:
    total_bins = num_bins_x * num_bins_y
    maps = np.empty((len(episodes), num_bins_y, num_bins_x), dtype=np.float64)
    for slot, episode in enumerate(episodes):
        activation = (
            activation_override[slot]
            if activation_override is not None
            else episode.activations[:, unit_index]
        )
        activity = (
            np.bincount(episode.linear_bins, weights=activation, minlength=total_bins)
            .reshape(num_bins_y, num_bins_x)
            .astype(np.float64)
        )
        if smoothing_sigma > 0.0:
            activity = gaussian_filter(activity, sigma=smoothing_sigma)
        rate = np.full((num_bins_y, num_bins_x), np.nan, dtype=np.float64)
        valid = episode.occupancy > min_occupancy
        rate[valid] = activity[valid] / episode.occupancy[valid]
        maps[slot] = rate
    return maps


@dataclass(slots=True)
class ReanchoringGridnessModule:
    """Re-anchoring-aware grid battery + per-episode autocorrelogram figures."""

    name: str = "reanchoring_gridness"
    cost_tier: str = "heavy"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        num_units = int(analysis_input.representation.shape[-1])
        pooled_bins_x = int(config.get("reanchoring_num_bins_x", config.get("num_bins_x", 40)))
        pooled_bins_y = int(config.get("reanchoring_num_bins_y", config.get("num_bins_y", 40)))
        episode_bins_x = int(
            config.get(
                "reanchoring_per_episode_num_bins_x",
                config.get("per_episode_num_bins_x", 20),
            )
        )
        episode_bins_y = int(
            config.get(
                "reanchoring_per_episode_num_bins_y",
                config.get("per_episode_num_bins_y", 20),
            )
        )
        smoothing_sigma = float(config.get("reanchoring_smoothing_sigma", 1.0))
        min_occupancy = float(config.get("reanchoring_min_occupancy", 1e-6))
        minimum_valid_steps = int(config.get("reanchoring_minimum_valid_steps", 50))
        minimum_visited_fraction = float(config.get("reanchoring_minimum_visited_fraction", 0.15))
        max_episodes = int(config.get("reanchoring_max_episodes", 256))
        shuffle_count = int(
            config.get(
                "reanchoring_shuffle_count",
                analysis_config_default("reanchoring_shuffle_count"),
            )
        )
        shuffle_top_k = int(config.get("reanchoring_shuffle_top_k", 32))
        fdr_alpha = float(config.get("reanchoring_fdr_alpha", 0.05))
        render_top_k = int(config.get("reanchoring_render_top_k", 8))
        render_episodes = int(config.get("reanchoring_render_example_episodes", 4))
        seed = int(config.get("reanchoring_random_seed", 0))

        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
        if world_bounds is None:
            valid_positions = flatten_positions(
                analysis_input.position_xy, analysis_input.valid_mask
            )
            if valid_positions.size == 0:
                raise ValueError("reanchoring_gridness needs at least one valid position.")
            world_bounds = infer_bounds(valid_positions)

        episodes, episode_stats = _collect_episode_bins(
            analysis_input,
            num_bins_x=episode_bins_x,
            num_bins_y=episode_bins_y,
            smoothing_sigma=smoothing_sigma,
            minimum_valid_steps=minimum_valid_steps,
            minimum_visited_fraction=minimum_visited_fraction,
            max_episodes=max_episodes,
            bounds=world_bounds,
        )

        pooled_world_gridness = self._pooled_gridness(
            analysis_input,
            num_bins_x=pooled_bins_x,
            num_bins_y=pooled_bins_y,
            smoothing_sigma=smoothing_sigma,
            min_occupancy=min_occupancy,
            bounds=world_bounds,
            num_units=num_units,
        )

        mean_per_episode = np.zeros(num_units, dtype=np.float32)
        averaged_autocorr_gridness = np.zeros(num_units, dtype=np.float32)
        radial_periodicity = np.zeros(num_units, dtype=np.float32)
        stripe = np.zeros(num_units, dtype=np.float32)
        lattice = np.zeros(num_units, dtype=np.float32)
        averaged_autocorr_by_unit: list[np.ndarray | None] = [None] * num_units

        for unit_index in range(num_units):
            if not episodes:
                break
            unit_maps = _unit_episode_maps(
                unit_index,
                episodes,
                num_bins_x=episode_bins_x,
                num_bins_y=episode_bins_y,
                smoothing_sigma=smoothing_sigma,
                min_occupancy=min_occupancy,
            )
            mean_per_episode[unit_index] = float(
                np.mean([gridness_score(np.nan_to_num(m, nan=0.0)) for m in unit_maps])
            )
            averaged_autocorr = average_autocorrelograms(unit_maps)
            averaged_autocorr_by_unit[unit_index] = averaged_autocorr
            averaged_autocorr_gridness[unit_index] = gridness_from_autocorrelogram(
                averaged_autocorr
            )
            radial_periodicity[unit_index] = radial_periodicity_score(averaged_autocorr)
            stripe[unit_index], lattice[unit_index] = angular_harmonics(averaged_autocorr)

        reanchoring_index = (averaged_autocorr_gridness - pooled_world_gridness).astype(np.float32)

        shuffle_p_value, shuffle_significant, candidates_tested = self._shuffle_null(
            episodes=episodes,
            averaged_autocorr_gridness=averaged_autocorr_gridness,
            num_units=num_units,
            episode_bins_x=episode_bins_x,
            episode_bins_y=episode_bins_y,
            smoothing_sigma=smoothing_sigma,
            min_occupancy=min_occupancy,
            shuffle_count=shuffle_count,
            shuffle_top_k=shuffle_top_k,
            fdr_alpha=fdr_alpha,
            seed=seed,
        )

        per_unit_metrics = {
            "pooled_world_gridness": pooled_world_gridness,
            "mean_per_episode_gridness": mean_per_episode,
            "averaged_autocorr_gridness": averaged_autocorr_gridness,
            "reanchoring_index": reanchoring_index,
            "radial_periodicity_score": radial_periodicity,
            "stripe_score": stripe,
            "lattice_score": lattice,
            "shuffle_p_value": shuffle_p_value,
            "shuffle_significant": shuffle_significant.astype(np.float32),
        }
        table_path = self._write_table(output_dir, analysis_input, per_unit_metrics)
        figures = self._render(
            output_dir,
            analysis_input,
            episodes=episodes,
            averaged_autocorr_by_unit=averaged_autocorr_by_unit,
            per_unit_metrics=per_unit_metrics,
            episode_bins_x=episode_bins_x,
            episode_bins_y=episode_bins_y,
            smoothing_sigma=smoothing_sigma,
            min_occupancy=min_occupancy,
            render_top_k=render_top_k,
            render_episodes=render_episodes,
        )

        def _summary(values: np.ndarray) -> dict[str, float]:
            return {
                "median": float(np.nanmedian(values)) if values.size else float("nan"),
                "max": float(np.nanmax(values)) if values.size else float("nan"),
            }

        metrics = {
            "reanchoring_episodes_used": float(episode_stats["episodes_used"]),
            "reanchoring_episodes_skipped_short": float(episode_stats["episodes_skipped_short"]),
            "reanchoring_episodes_skipped_sparse": float(episode_stats["episodes_skipped_sparse"]),
            "reanchoring_episodes_capped": float(episode_stats["episodes_capped"]),
            "median_averaged_autocorr_gridness": _summary(averaged_autocorr_gridness)["median"],
            "max_averaged_autocorr_gridness": _summary(averaged_autocorr_gridness)["max"],
            "median_pooled_world_gridness": _summary(pooled_world_gridness)["median"],
            "max_reanchoring_index": _summary(reanchoring_index)["max"],
            "median_radial_periodicity_score": _summary(radial_periodicity)["median"],
            "shuffle_candidates_tested": float(candidates_tested),
            "num_significant_after_fdr": float(int(np.count_nonzero(shuffle_significant))),
        }
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics=per_unit_metrics,
            figures=figures,
            tables={"per_unit_metrics": table_path} if table_path is not None else {},
            metadata={
                "reanchoring_pooled_num_bins": [pooled_bins_x, pooled_bins_y],
                "reanchoring_per_episode_num_bins": [episode_bins_x, episode_bins_y],
                "reanchoring_minimum_valid_steps": minimum_valid_steps,
                "reanchoring_minimum_visited_fraction": minimum_visited_fraction,
                "reanchoring_shuffle_count": shuffle_count,
                "reanchoring_shuffle_top_k": shuffle_top_k,
                "reanchoring_shuffle_candidates_tested": candidates_tested,
                "reanchoring_signed_rectified": True,
            },
        )


    def _pooled_gridness(
        self,
        analysis_input: AnalysisInput,
        *,
        num_bins_x: int,
        num_bins_y: int,
        smoothing_sigma: float,
        min_occupancy: float,
        bounds,
        num_units: int,
    ) -> np.ndarray:
        from .helpers import compute_rate_maps

        valid = analysis_input.valid_mask.astype(bool, copy=False)
        if not np.any(valid):
            return np.zeros(num_units, dtype=np.float32)
        pooled = compute_rate_maps(
            analysis_input.representation,
            analysis_input.position_xy,
            valid,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=smoothing_sigma,
            min_occupancy=min_occupancy,
            bounds=bounds,
        ).rate_maps
        return np.asarray(
            [gridness_score(np.nan_to_num(rate_map, nan=0.0)) for rate_map in pooled],
            dtype=np.float32,
        )

    def _shuffle_null(
        self,
        *,
        episodes: list[_EpisodeBins],
        averaged_autocorr_gridness: np.ndarray,
        num_units: int,
        episode_bins_x: int,
        episode_bins_y: int,
        smoothing_sigma: float,
        min_occupancy: float,
        shuffle_count: int,
        shuffle_top_k: int,
        fdr_alpha: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        p_values = np.full(num_units, np.nan, dtype=np.float32)
        significant = np.zeros(num_units, dtype=bool)
        if shuffle_count <= 0 or not episodes:
            return p_values, significant, 0
        candidate_count = min(shuffle_top_k, num_units)
        candidates = np.argsort(-averaged_autocorr_gridness)[:candidate_count]
        rng = np.random.default_rng(seed)
        for unit_index in candidates:
            observed = float(averaged_autocorr_gridness[unit_index])
            exceed = 0
            for _ in range(shuffle_count):
                shuffled = [
                    np.roll(
                        episode.activations[:, unit_index],
                        int(rng.integers(1, max(2, episode.activations.shape[0]))),
                    )
                    for episode in episodes
                ]
                shuffled_maps = _unit_episode_maps(
                    unit_index,
                    episodes,
                    num_bins_x=episode_bins_x,
                    num_bins_y=episode_bins_y,
                    smoothing_sigma=smoothing_sigma,
                    min_occupancy=min_occupancy,
                    activation_override=shuffled,
                )
                shuffled_score = gridness_from_autocorrelogram(
                    average_autocorrelograms(shuffled_maps)
                )
                if shuffled_score >= observed:
                    exceed += 1
            p_values[unit_index] = (1.0 + exceed) / (1.0 + shuffle_count)
        reject = benjamini_hochberg(p_values, alpha=fdr_alpha)
        significant[reject] = True
        return p_values, significant, int(candidate_count)

    def _write_table(self, output_dir: Path, analysis_input: AnalysisInput, per_unit):
        if output_dir is None:
            return None
        columns = list(per_unit.keys())
        rows = []
        for unit_index in range(len(per_unit[columns[0]])):
            row: list[object] = [unit_index]
            for column in columns:
                value = float(per_unit[column][unit_index])
                row.append(value if np.isfinite(value) else "nan")
            rows.append(row)
        filename = (
            f"reanchoring_gridness_per_unit__"
            f"{analysis_input.source_name}__{analysis_input.split_name}.csv"
        )
        return write_csv(
            output_dir / self.name / filename,
            ["unit_index", *columns],
            rows,
        )

    def _render(
        self,
        output_dir: Path,
        analysis_input: AnalysisInput,
        *,
        episodes: list[_EpisodeBins],
        averaged_autocorr_by_unit,
        per_unit_metrics,
        episode_bins_x: int,
        episode_bins_y: int,
        smoothing_sigma: float,
        min_occupancy: float,
        render_top_k: int,
        render_episodes: int,
    ) -> dict[str, Path]:
        if output_dir is None or not episodes:
            return {}
        scores = per_unit_metrics["averaged_autocorr_gridness"]
        interest = scores + per_unit_metrics["radial_periodicity_score"]
        ranked = np.argsort(-interest)[: max(1, render_top_k)]
        example_slots = np.linspace(0, len(episodes) - 1, min(render_episodes, len(episodes)))
        example_slots = sorted({int(round(s)) for s in example_slots})
        num_cols = len(example_slots) + 1
        figure, axes = plt.subplots(
            len(ranked),
            num_cols,
            figsize=(2.5 * num_cols, 2.6 * len(ranked)),
            squeeze=False,
        )
        figure.suptitle(
            "Re-anchoring grid battery, per-episode autocorrelograms, "
            f"{analysis_input.source_name} ({analysis_input.split_name})",
            fontsize=12,
        )
        for row, unit_index in enumerate(ranked.astype(int)):
            for column, slot in enumerate(example_slots):
                unit_maps = _unit_episode_maps(
                    unit_index,
                    [episodes[slot]],
                    num_bins_x=episode_bins_x,
                    num_bins_y=episode_bins_y,
                    smoothing_sigma=smoothing_sigma,
                    min_occupancy=min_occupancy,
                )
                episode_autocorr = autocorrelogram(np.nan_to_num(unit_maps[0], nan=0.0))
                axis = axes[row][column]
                axis.imshow(episode_autocorr, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(f"episode {episodes[slot].episode_index}", fontsize=8)
                if column == 0:
                    axis.set_ylabel(
                        f"unit {unit_index}\nR_ac {scores[unit_index]:.2f}\n"
                        f"reanchor {per_unit_metrics['reanchoring_index'][unit_index]:.2f}",
                        fontsize=8,
                    )
            averaged = averaged_autocorr_by_unit[unit_index]
            axis = axes[row][-1]
            if averaged is not None:
                axis.imshow(averaged, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title("averaged autocorrelogram", fontsize=8)
        figure.subplots_adjust(
            left=0.08, right=0.985, bottom=0.03, top=0.92, wspace=0.1, hspace=0.15
        )
        path = (
            output_dir
            / self.name
            / f"reanchoring_gridness__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(figure, path, dpi=150)
        plt.close(figure)
        return {"reanchoring_gridness_panel": path}
