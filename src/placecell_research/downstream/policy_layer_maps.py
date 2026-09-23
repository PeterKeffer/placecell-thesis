"""Q-network policy-layer spatial diagnostics for downstream eval rollouts."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from placecell_research.analysis.rate_map_export import render_rate_map_grid_pages
from placecell_research.analysis.world_overlay import (
    overlay_bounds,
    resolve_world_overlay,
)
from placecell_research.evaluation.decode import (
    episode_level_decode_skip_reason,
    linear_decode_position,
    nonlinear_decode_position,
)
from placecell_research.numerics.place_cell_quality import reliability_weighted_information
from placecell_research.numerics.rate_map_kernels import (
    compute_rate_maps,
    gridness_score,
    prepare_place_metric_rate_maps,
    skaggs_spatial_information,
)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "layer"


def _position_xy_from_info(info: dict[str, Any]) -> list[float] | None:
    raw_position = info.get("position_xy")
    if raw_position is None:
        raw_position = info.get("next_start_position_xy")
    if raw_position is None:
        return None
    position = np.asarray(raw_position, dtype=np.float32).reshape(-1)
    if position.shape[0] < 2:
        return None
    return [float(position[0]), float(position[1])]


def _flatten_output(output: Any) -> np.ndarray | None:
    tensor = output[0] if isinstance(output, tuple) else output
    detach = getattr(tensor, "detach", None)
    if not callable(detach):
        return None
    array = detach().float().cpu().numpy()
    if array.ndim == 1:
        array = array[None, :]
    return array.reshape(array.shape[0], -1).astype(np.float32, copy=False)


@dataclass(slots=True)
class _LayerBuffer:
    name: str
    value_chunks: list[np.ndarray] = field(default_factory=list)
    position_chunks: list[np.ndarray] = field(default_factory=list)
    episode_chunks: list[np.ndarray] = field(default_factory=list)

    def append(
        self, values: np.ndarray, positions: np.ndarray, episode_indices: np.ndarray
    ) -> None:
        self.value_chunks.append(values.astype(np.float32, copy=True))
        self.position_chunks.append(positions.astype(np.float32, copy=True))
        self.episode_chunks.append(episode_indices.astype(np.int64, copy=True))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.concatenate(self.value_chunks, axis=0),
            np.concatenate(self.position_chunks, axis=0),
            np.concatenate(self.episode_chunks, axis=0),
        )


class PolicyLayerActivationRecorder:
    """Capture Q-network layer activations and write spatial maps after eval."""

    def __init__(
        self,
        *,
        model: Any,
        output_dir: Path,
        env_id: str,
        max_episodes: int | None = None,
        num_bins_x: int = 60,
        num_bins_y: int = 60,
        smoothing_sigma: float = 0.4,
        min_occupancy: float = 1e-6,
        page_size: int = 64,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.env_id = str(env_id)
        self.max_episodes = None if max_episodes is None else int(max_episodes)
        self.num_bins_x = int(num_bins_x)
        self.num_bins_y = int(num_bins_y)
        self.smoothing_sigma = float(smoothing_sigma)
        self.min_occupancy = float(min_occupancy)
        self.page_size = max(1, int(page_size))
        self._pending_positions: np.ndarray | None = None
        self._pending_episode_indices: np.ndarray | None = None
        self._buffers: dict[str, _LayerBuffer] = {}
        self._handles: list[Any] = []
        self._install_hooks(model)

    @property
    def enabled(self) -> bool:
        return bool(self._handles)

    def _install_hooks(self, model: Any) -> None:
        import torch

        policy = getattr(model, "policy", None)
        features_extractor = getattr(policy, "features_extractor", None)
        if not hasattr(features_extractor, "register_forward_hook"):
            features_extractor = getattr(getattr(policy, "q_net", None), "features_extractor", None)
        if hasattr(features_extractor, "register_forward_hook"):
            self._register_layer_hook("features", features_extractor)

        q_network = getattr(policy, "q_net", None)
        q_layers = getattr(q_network, "q_net", None)
        if q_layers is not None:
            linear_modules = [
                module for module in q_layers.modules() if isinstance(module, torch.nn.Linear)
            ]
            for layer_index, module in enumerate(linear_modules):
                layer_name = (
                    "q_values"
                    if layer_index == len(linear_modules) - 1
                    else f"linear_{layer_index:02d}"
                )
                self._register_layer_hook(layer_name, module)
            return

        policy_net = getattr(getattr(policy, "mlp_extractor", None), "policy_net", None)
        if policy_net is not None:
            linear_modules = [
                module for module in policy_net.modules() if isinstance(module, torch.nn.Linear)
            ]
            for layer_index, module in enumerate(linear_modules):
                self._register_layer_hook(f"policy_linear_{layer_index:02d}", module)
        action_net = getattr(policy, "action_net", None)
        if isinstance(action_net, torch.nn.Linear):
            self._register_layer_hook("action_logits", action_net)

    def _register_layer_hook(self, layer_name: str, module: Any) -> None:
        self._buffers[layer_name] = _LayerBuffer(name=layer_name)
        self._handles.append(module.register_forward_hook(self._make_hook(layer_name)))

    def _make_hook(self, layer_name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            positions = self._pending_positions
            episode_indices = self._pending_episode_indices
            values = _flatten_output(output)
            if positions is None or episode_indices is None or values is None:
                return
            row_count = min(values.shape[0], positions.shape[0], episode_indices.shape[0])
            if row_count <= 0:
                return
            self._buffers[layer_name].append(
                values[:row_count],
                positions[:row_count],
                episode_indices[:row_count],
            )

        return hook

    def prepare_step(self, infos: list[dict[str, Any]], episode_indices: np.ndarray) -> None:
        positions: list[list[float]] = []
        retained_episode_indices: list[int] = []
        raw_episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
        for info_index, info in enumerate(infos):
            if info_index >= raw_episode_indices.shape[0]:
                break
            episode_index = int(raw_episode_indices[info_index])
            if self.max_episodes is not None and episode_index >= self.max_episodes:
                continue
            position_xy = _position_xy_from_info(info if isinstance(info, dict) else {})
            if position_xy is None:
                continue
            positions.append(position_xy)
            retained_episode_indices.append(episode_index)
        if not positions:
            self._pending_positions = None
            self._pending_episode_indices = None
            return
        self._pending_positions = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
        self._pending_episode_indices = np.asarray(retained_episode_indices, dtype=np.int64)

    def close(self) -> dict[str, Any] | None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._pending_positions = None
        self._pending_episode_indices = None
        nonempty_buffers = [buffer for buffer in self._buffers.values() if buffer.value_chunks]
        if not nonempty_buffers:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_rows: list[dict[str, float | int | str]] = []
        layer_summaries: dict[str, dict[str, Any]] = {}
        decode_rows: list[dict[str, float | int | str]] = []
        for buffer in nonempty_buffers:
            rows, summary = self._write_layer_outputs(buffer)
            all_rows.extend(rows)
            layer_summaries[buffer.name] = summary
            decode_rows.append(_decode_csv_row(buffer.name, summary))
        metrics_path = self.output_dir / "policy_layer_metrics.csv"
        self._write_metrics_csv(metrics_path, all_rows)
        decode_path = self.output_dir / "policy_layer_decode.csv"
        self._write_decode_csv(decode_path, decode_rows)
        summary = {
            "metrics_csv_path": str(metrics_path),
            "decode_csv_path": str(decode_path),
            "layers": layer_summaries,
        }
        summary_path = self.output_dir / "policy_layer_diagnostics.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        summary["summary_path"] = str(summary_path)
        return summary

    def _write_layer_outputs(
        self, buffer: _LayerBuffer
    ) -> tuple[list[dict[str, float | int | str]], dict[str, Any]]:
        values, positions, episode_indices = buffer.arrays()
        decode_summary = _decode_layer_position(values, positions, episode_indices)
        representation, position_xy, valid_mask = _pad_by_episode(
            values, positions, episode_indices
        )
        bounds = _world_bounds(self.env_id)
        rate_map_result = compute_rate_maps(
            representation,
            position_xy,
            valid_mask,
            num_bins_x=self.num_bins_x,
            num_bins_y=self.num_bins_y,
            smoothing_sigma=self.smoothing_sigma,
            min_occupancy=self.min_occupancy,
            bounds=bounds,
        )
        raw_rate_maps = rate_map_result.rate_maps
        prepared_place_metric_maps = prepare_place_metric_rate_maps(raw_rate_maps)
        spatial_information = np.full((raw_rate_maps.shape[0],), np.nan, dtype=np.float32)
        supports_place_metrics = prepared_place_metric_maps.supported_mask
        if np.any(supports_place_metrics):
            spatial_information[supports_place_metrics] = np.asarray(
                skaggs_spatial_information(
                    prepared_place_metric_maps.clipped_rate_maps[supports_place_metrics],
                    rate_map_result.occupancy,
                ),
                dtype=np.float32,
            )
        gridness_maps = np.nan_to_num(raw_rate_maps, nan=0.0, posinf=0.0, neginf=0.0)
        gridness = np.asarray(
            [gridness_score(rate_map) for rate_map in gridness_maps], dtype=np.float32
        )
        split_half = _split_half_correlations(
            representation,
            position_xy,
            valid_mask,
            bounds=rate_map_result.bounds,
            num_bins_x=self.num_bins_x,
            num_bins_y=self.num_bins_y,
            smoothing_sigma=self.smoothing_sigma,
            min_occupancy=self.min_occupancy,
        )
        peak_activation = _nanmax_per_unit(raw_rate_maps)
        mean_activation = _nanmean_per_unit(raw_rate_maps)
        active_bin_fraction = _active_bin_fraction(prepared_place_metric_maps.clipped_rate_maps)
        reliability_weighted_information_scores = reliability_weighted_information(
            spatial_information,
            split_half,
        )
        ranking_score = _policy_layer_ranking_score(
            reliability_weighted_information_scores=reliability_weighted_information_scores,
            split_half=split_half,
            peak_activation=peak_activation,
        )
        pages = _write_rate_map_pages(
            self.output_dir,
            layer_name=buffer.name,
            rate_maps=raw_rate_maps,
            spatial_information=spatial_information,
            split_half=split_half,
            ranking_score=ranking_score,
            page_size=self.page_size,
            bounds=rate_map_result.bounds,
            env_id=self.env_id,
        )
        rows: list[dict[str, float | int | str]] = []
        for unit_index in range(raw_rate_maps.shape[0]):
            rows.append(
                {
                    "layer_name": buffer.name,
                    "unit_index": int(unit_index),
                    "supports_place_metrics": int(supports_place_metrics[unit_index]),
                    "spatial_information_bits": float(spatial_information[unit_index]),
                    "reliability_weighted_information": float(
                        reliability_weighted_information_scores[unit_index]
                    ),
                    "gridness": float(gridness[unit_index]),
                    "split_half_correlation": float(split_half[unit_index]),
                    "peak_activation": float(peak_activation[unit_index]),
                    "mean_activation": float(mean_activation[unit_index]),
                    "active_bin_fraction": float(active_bin_fraction[unit_index]),
                    "negative_bin_fraction": float(
                        prepared_place_metric_maps.negative_bin_fraction[unit_index]
                    ),
                    "negative_peak_fraction": float(
                        prepared_place_metric_maps.negative_peak_fraction[unit_index]
                    ),
                }
            )
        summary = {
            "num_units": int(raw_rate_maps.shape[0]),
            "num_samples": int(values.shape[0]),
            "rate_map_pages": [str(path) for path in pages],
            "mean_gridness": _nanmean(gridness),
            "max_gridness": _nanmax(gridness),
            "mean_spatial_information_bits": _nanmean(spatial_information),
            "mean_reliability_weighted_information": _nanmean(
                reliability_weighted_information_scores
            ),
            "max_reliability_weighted_information": _nanmax(
                reliability_weighted_information_scores
            ),
            "mean_split_half_correlation": _nanmean(split_half),
            "place_metric_supported_units": int(np.count_nonzero(supports_place_metrics)),
            "rate_map_ranking_metric": "reliability_weighted_information_with_split_half_tiebreak",
            **decode_summary,
        }
        return rows, summary

    @staticmethod
    def _write_metrics_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
        fieldnames = [
            "layer_name",
            "unit_index",
            "supports_place_metrics",
            "spatial_information_bits",
            "reliability_weighted_information",
            "gridness",
            "split_half_correlation",
            "peak_activation",
            "mean_activation",
            "active_bin_fraction",
            "negative_bin_fraction",
            "negative_peak_fraction",
        ]
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_decode_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
        fieldnames = [
            "layer_name",
            "num_samples",
            "decode_rmse",
            "decode_r2",
            "decode_shuffle_rmse",
            "nonlinear_decode_rmse",
            "nonlinear_decode_r2",
            "decode_train_size",
            "decode_validation_size",
            "decode_skip_reason",
        ]
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _decode_layer_position(
    values: np.ndarray,
    positions: np.ndarray,
    episode_indices: np.ndarray,
) -> dict[str, float | int | str]:
    skip_reason = episode_level_decode_skip_reason(episode_indices)
    if skip_reason is not None:
        return {
            "decode_rmse": float("nan"),
            "decode_r2": float("nan"),
            "decode_shuffle_rmse": float("nan"),
            "nonlinear_decode_rmse": float("nan"),
            "nonlinear_decode_r2": float("nan"),
            "decode_train_size": 0,
            "decode_validation_size": 0,
            "decode_skip_reason": skip_reason,
        }
    linear_decode = linear_decode_position(
        values,
        positions,
        episode_ids=episode_indices,
        include_shuffle=True,
    )
    nonlinear_decode = nonlinear_decode_position(
        values,
        positions,
        episode_ids=episode_indices,
    )
    return {
        "decode_rmse": float(linear_decode.rmse),
        "decode_r2": float(linear_decode.r2),
        "decode_shuffle_rmse": (
            float(linear_decode.shuffle_rmse)
            if linear_decode.shuffle_rmse is not None
            else float("nan")
        ),
        "nonlinear_decode_rmse": float(nonlinear_decode.rmse),
        "nonlinear_decode_r2": float(nonlinear_decode.r2),
        "decode_train_size": int(linear_decode.train_size),
        "decode_validation_size": int(linear_decode.validation_size),
        "decode_skip_reason": "",
    }


def _decode_csv_row(layer_name: str, summary: dict[str, Any]) -> dict[str, float | int | str]:
    return {
        "layer_name": layer_name,
        "num_samples": int(summary["num_samples"]),
        "decode_rmse": float(summary["decode_rmse"]),
        "decode_r2": float(summary["decode_r2"]),
        "decode_shuffle_rmse": float(summary["decode_shuffle_rmse"]),
        "nonlinear_decode_rmse": float(summary["nonlinear_decode_rmse"]),
        "nonlinear_decode_r2": float(summary["nonlinear_decode_r2"]),
        "decode_train_size": int(summary["decode_train_size"]),
        "decode_validation_size": int(summary["decode_validation_size"]),
        "decode_skip_reason": str(summary["decode_skip_reason"]),
    }


def _world_bounds(env_id: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    world_overlay = resolve_world_overlay(str(env_id))
    return overlay_bounds(world_overlay) if world_overlay is not None else None


def _pad_by_episode(
    values: np.ndarray,
    positions: np.ndarray,
    episode_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_episodes = np.unique(episode_indices.astype(np.int64, copy=False))
    counts = np.asarray(
        [np.count_nonzero(episode_indices == episode) for episode in unique_episodes],
        dtype=np.int64,
    )
    max_steps = int(counts.max()) if counts.size else 0
    num_units = int(values.shape[1])
    representation = np.zeros((len(unique_episodes), max_steps, num_units), dtype=np.float32)
    position_xy = np.zeros((len(unique_episodes), max_steps, 2), dtype=np.float32)
    valid_mask = np.zeros((len(unique_episodes), max_steps), dtype=bool)
    for output_episode_index, episode in enumerate(unique_episodes):
        source_indices = np.flatnonzero(episode_indices == episode)
        step_count = int(source_indices.size)
        representation[output_episode_index, :step_count] = values[source_indices]
        position_xy[output_episode_index, :step_count] = positions[source_indices]
        valid_mask[output_episode_index, :step_count] = True
    return representation, position_xy, valid_mask


def _split_half_correlations(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    num_bins_x: int,
    num_bins_y: int,
    smoothing_sigma: float,
    min_occupancy: float,
) -> np.ndarray:
    num_units = int(representation.shape[-1])
    if representation.shape[0] < 2:
        return np.full((num_units,), np.nan, dtype=np.float32)
    even_indices = np.arange(0, representation.shape[0], 2, dtype=np.int64)
    odd_indices = np.arange(1, representation.shape[0], 2, dtype=np.int64)
    if even_indices.size == 0 or odd_indices.size == 0:
        return np.full((num_units,), np.nan, dtype=np.float32)
    even_maps = np.clip(
        compute_rate_maps(
            representation[even_indices],
            position_xy[even_indices],
            valid_mask[even_indices],
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=smoothing_sigma,
            min_occupancy=min_occupancy,
            bounds=bounds,
        ).rate_maps,
        a_min=0.0,
        a_max=None,
    )
    odd_maps = np.clip(
        compute_rate_maps(
            representation[odd_indices],
            position_xy[odd_indices],
            valid_mask[odd_indices],
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=smoothing_sigma,
            min_occupancy=min_occupancy,
            bounds=bounds,
        ).rate_maps,
        a_min=0.0,
        a_max=None,
    )
    correlations = np.full((num_units,), np.nan, dtype=np.float32)
    for unit_index in range(num_units):
        first = even_maps[unit_index].reshape(-1)
        second = odd_maps[unit_index].reshape(-1)
        valid = np.isfinite(first) & np.isfinite(second)
        if np.count_nonzero(valid) < 2:
            continue
        first_valid = first[valid]
        second_valid = second[valid]
        if float(np.std(first_valid)) <= 1e-8 or float(np.std(second_valid)) <= 1e-8:
            continue
        correlations[unit_index] = float(np.corrcoef(first_valid, second_valid)[0, 1])
    return correlations


def _nanmean_per_unit(rate_maps: np.ndarray) -> np.ndarray:
    values = np.full((rate_maps.shape[0],), np.nan, dtype=np.float32)
    for unit_index, rate_map in enumerate(rate_maps):
        finite_values = rate_map[np.isfinite(rate_map)]
        if finite_values.size:
            values[unit_index] = float(np.mean(finite_values))
    return values


def _nanmax_per_unit(rate_maps: np.ndarray) -> np.ndarray:
    values = np.full((rate_maps.shape[0],), np.nan, dtype=np.float32)
    for unit_index, rate_map in enumerate(rate_maps):
        finite_values = rate_map[np.isfinite(rate_map)]
        if finite_values.size:
            values[unit_index] = float(np.max(finite_values))
    return values


def _active_bin_fraction(rate_maps: np.ndarray) -> np.ndarray:
    fractions = np.zeros((rate_maps.shape[0],), dtype=np.float32)
    for unit_index, rate_map in enumerate(rate_maps):
        finite_values = rate_map[np.isfinite(rate_map)]
        if finite_values.size == 0:
            fractions[unit_index] = np.nan
            continue
        peak = float(np.max(finite_values))
        if peak <= 1e-8:
            fractions[unit_index] = 0.0
            continue
        fractions[unit_index] = float(np.mean(finite_values >= 0.2 * peak))
    return fractions


def _policy_layer_ranking_score(
    *,
    reliability_weighted_information_scores: np.ndarray,
    split_half: np.ndarray,
    peak_activation: np.ndarray,
) -> np.ndarray:
    return (
        np.nan_to_num(reliability_weighted_information_scores, nan=0.0)
        + 0.05 * np.maximum(np.nan_to_num(split_half, nan=0.0), 0.0)
        + 1e-3 * np.nan_to_num(peak_activation, nan=0.0)
    ).astype(np.float32, copy=False)


def _write_rate_map_pages(
    output_dir: Path,
    *,
    layer_name: str,
    rate_maps: np.ndarray,
    spatial_information: np.ndarray,
    split_half: np.ndarray,
    ranking_score: np.ndarray,
    page_size: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    env_id: str,
) -> list[Path]:
    ranked_indices = np.argsort(np.nan_to_num(ranking_score, nan=-np.inf))[::-1]
    return render_rate_map_grid_pages(
        output_dir=output_dir,
        file_prefix=_safe_name(layer_name),
        source_name=layer_name,
        split_name="downstream eval",
        bounds=bounds,
        rate_maps=rate_maps,
        ranked_indices=ranked_indices,
        spatial_information_bits=spatial_information,
        mean_panel_metric_inside_fields=split_half,
        world_overlay=resolve_world_overlay(env_id),
        mean_spatial_information_bits=_nanmean(spatial_information),
        shared_color_scale=False,
        colormap_mode="auto",
        panel_metric_name="split_half_agreement",
        title_prefix="Policy Layer Rate Maps",
        render_dpi=150,
        page_size=page_size,
    )


def _nanmean(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    return float(np.mean(finite_values)) if finite_values.size else float("nan")


def _nanmax(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    return float(np.max(finite_values)) if finite_values.size else float("nan")
