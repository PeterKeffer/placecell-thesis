"""Per-traversal firing reliability of place-like units, for every unit of a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_spans(episode_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start index and length of every episode in the flattened valid-step order."""
    change = np.flatnonzero(np.diff(episode_ids)) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [episode_ids.shape[0]]]))
    return starts, lengths


def shifted_index(starts: np.ndarray, lengths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Index array that rolls every episode by its own random offset (at least 5% of its length)."""
    total = int(lengths.sum())
    index = np.empty(total, dtype=np.int64)
    for start, length in zip(starts, lengths, strict=False):
        minimum = max(1, int(0.05 * length))
        offset = int(rng.integers(minimum, max(minimum + 1, length - minimum + 1)))
        local = (np.arange(length) + offset) % length
        index[start : start + length] = start + local
    return index


def traversal_table(
    in_field: np.ndarray, segment_start: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (traversal_start_indices, traversal_id_per_step); id is -1 outside the field."""
    left_field = np.empty_like(in_field)
    left_field[0] = True
    left_field[1:] = ~in_field[:-1]
    starts = in_field & (left_field | segment_start)
    start_indices = np.flatnonzero(starts)
    traversal_id = np.cumsum(starts) - 1
    traversal_id[~in_field] = -1
    return start_indices, traversal_id


def hits_per_traversal(active: np.ndarray, start_indices: np.ndarray) -> np.ndarray:
    """One boolean per traversal: any active step between its start and the next start."""
    if start_indices.size == 0:
        return np.zeros(0, dtype=bool)
    return np.logical_or.reduceat(active, start_indices)


def run(args: argparse.Namespace) -> dict:
    from placecell_research.analysis.base import AnalysisInput
    from placecell_research.analysis.helpers import compute_rate_maps_from_episode_statistics
    from placecell_research.analysis.occupancy import get_or_compute_episode_bin_statistics
    from placecell_research.analysis.reliability_splits import compute_field_traversal_reliability
    from placecell_research.datasets.batch_iterator import load_split_indices
    from placecell_research.evaluation.inference import (
        collect_representations,
        load_model_checkpoint,
    )
    from placecell_research.numerics.rate_map_kernels import compute_place_field_mask

    repo = args.repo
    model_dir = repo / "artifacts/place_models" / args.model_id
    checkpoint = model_dir / "weights_last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    device = torch.device("cpu")
    model, _ = load_model_checkpoint(model_dir, device, selection="last")
    print(f"Collecting encoder.place_codes for {args.model_id}", flush=True)
    representations, metadata = collect_representations(
        model,
        repo / "artifacts/datasets/encoded" / args.dataset,
        repo / "artifacts/splits" / args.split,
        "test",
        ["encoder.place_codes"],
        device,
        batch_size=8,
        observation_source="latent",
        max_episodes=args.max_episodes,
    )
    del model
    codes = representations["encoder.place_codes"]
    valid = metadata["valid_steps"].astype(bool)
    position = metadata["position_xy"]
    heading = metadata["heading"]
    ids = load_split_indices(repo / "artifacts/splits" / args.split, "test")
    if args.max_episodes is not None:
        ids = ids[: codes.shape[0]]
    num_units = codes.shape[-1]
    positive = np.maximum(codes, 0.0).astype(np.float32)

    data = AnalysisInput(
        representation=positive,
        position_xy=position,
        heading=heading,
        kinematics=None,
        actions=None,
        valid_mask=valid,
        source_name="encoder.place_codes",
        label=args.model_id,
        split_name="test",
        metadata={"env_id": args.env_id},
    )
    statistics = get_or_compute_episode_bin_statistics(
        data, num_bins_x=args.bins, num_bins_y=args.bins, bounds=None
    )
    if statistics is None:
        raise RuntimeError("No valid steps.")
    num_bins = statistics.num_bins_x * statistics.num_bins_y
    rate = compute_rate_maps_from_episode_statistics(
        statistics, smoothing_sigma=0.0, min_occupancy=1e-6
    )
    rate_maps = np.asarray(rate.rate_maps).reshape(
        num_units, statistics.num_bins_y, statistics.num_bins_x
    )
    field_masks = np.zeros((num_units, num_bins), dtype=bool)
    core_masks = np.zeros((num_units, num_bins), dtype=bool)
    field_bins = np.zeros(num_units, dtype=np.int32)
    field_count = np.zeros(num_units, dtype=np.int32)
    for unit in range(num_units):
        mask, count, area = compute_place_field_mask(rate_maps[unit], args.field_threshold)
        core, _, _ = compute_place_field_mask(rate_maps[unit], args.core_threshold)
        field_masks[unit] = mask.reshape(-1)
        core_masks[unit] = core.reshape(-1)
        field_bins[unit] = int(area)
        field_count[unit] = int(count)

    flat = statistics.flat_values
    steps = flat.shape[0]
    spatial_bins = (statistics.episode_bin_codes % num_bins).astype(np.int64)
    episode_ids = (statistics.episode_bin_codes // num_bins).astype(np.int64)
    segment_start = statistics.segment_start.astype(bool)
    flat_heading = heading[valid].astype(np.float64).reshape(-1)
    if flat_heading.shape[0] != steps:
        raise ValueError("heading and flat values disagree")
    peaks = np.array(
        [np.quantile(flat[:, u].astype(np.float64), 0.995) for u in range(num_units)],
        dtype=np.float32,
    )
    strong_all = flat >= (args.hit_threshold * peaks)[None, :]
    strong_all &= flat > 0
    any_all = flat > 0
    ep_starts, ep_lengths = episode_spans(episode_ids)
    rng = np.random.default_rng(args.seed)
    shift_indices = [shifted_index(ep_starts, ep_lengths, rng) for _ in range(args.shifts)]

    sectors = args.sectors
    sector_width = 2.0 * np.pi / sectors
    out = {
        "traversals": np.zeros(num_units, dtype=np.int32),
        "hit_strong": np.full(num_units, np.nan, dtype=np.float32),
        "hit_any": np.full(num_units, np.nan, dtype=np.float32),
        "core_traversals": np.zeros(num_units, dtype=np.int32),
        "hit_strong_core": np.full(num_units, np.nan, dtype=np.float32),
        "sector_hit_strong": np.full((num_units, sectors), np.nan, dtype=np.float32),
        "sector_traversals": np.zeros((num_units, sectors), dtype=np.int32),
        "preferred_sector": np.full(num_units, -1, dtype=np.int32),
        "directional_heldout": np.full(num_units, np.nan, dtype=np.float32),
        "directional_heldout_traversals": np.zeros(num_units, dtype=np.int32),
        "null_mean": np.full(num_units, np.nan, dtype=np.float32),
        "null_p95": np.full(num_units, np.nan, dtype=np.float32),
        "null_p": np.full(num_units, np.nan, dtype=np.float32),
        "excess_over_null": np.full(num_units, np.nan, dtype=np.float32),
        "mean_traversal_steps": np.full(num_units, np.nan, dtype=np.float32),
    }
    for unit in range(num_units):
        if field_bins[unit] == 0:
            continue
        in_field = field_masks[unit][spatial_bins]
        if not in_field.any():
            continue
        start_indices, traversal_id = traversal_table(in_field, segment_start)
        n = start_indices.size
        out["traversals"][unit] = n
        if n == 0:
            continue
        strong = strong_all[:, unit]
        hit_strong = hits_per_traversal(in_field & strong, start_indices)
        hit_any = hits_per_traversal(in_field & any_all[:, unit], start_indices)
        out["hit_strong"][unit] = hit_strong.mean()
        out["hit_any"][unit] = hit_any.mean()
        out["mean_traversal_steps"][unit] = in_field.sum() / n
        core_in = in_field & core_masks[unit][spatial_bins]
        core_trav = hits_per_traversal(core_in, start_indices)
        out["core_traversals"][unit] = int(core_trav.sum())
        if core_trav.any():
            out["hit_strong_core"][unit] = hit_strong[core_trav].mean()
        in_ids = traversal_id[in_field]
        sin_sum = np.bincount(in_ids, weights=np.sin(flat_heading[in_field]), minlength=n)
        cos_sum = np.bincount(in_ids, weights=np.cos(flat_heading[in_field]), minlength=n)
        angle = np.arctan2(sin_sum, cos_sum)
        sector = (np.floor((angle + np.pi) / sector_width).astype(np.int64)) % sectors
        totals = np.bincount(sector, minlength=sectors)
        hits = np.bincount(sector[hit_strong], minlength=sectors)
        out["sector_traversals"][unit] = totals
        with np.errstate(invalid="ignore", divide="ignore"):
            out["sector_hit_strong"][unit] = np.where(
                totals > 0, hits / np.maximum(totals, 1), np.nan
            )
        held_out = (episode_ids[start_indices] % 2) == 1
        sel_tot = np.bincount(sector[~held_out], minlength=sectors)
        sel_hit = np.bincount(sector[~held_out & hit_strong], minlength=sectors)
        ho_tot = np.bincount(sector[held_out], minlength=sectors)
        ho_hit = np.bincount(sector[held_out & hit_strong], minlength=sectors)
        eligible = (sel_tot >= args.min_traversals) & (ho_tot >= args.min_traversals)
        if eligible.any():
            rates = np.where(eligible, sel_hit / np.maximum(sel_tot, 1), -1.0)
            best = int(np.argmax(rates))
            out["preferred_sector"][unit] = best
            out["directional_heldout"][unit] = ho_hit[best] / ho_tot[best]
            out["directional_heldout_traversals"][unit] = int(ho_tot[best])
        if args.shifts > 0:
            null_rates = np.empty(args.shifts, dtype=np.float32)
            for k, index in enumerate(shift_indices):
                null_rates[k] = hits_per_traversal(in_field & strong[index], start_indices).mean()
            observed = out["hit_strong"][unit]
            out["null_mean"][unit] = null_rates.mean()
            out["null_p95"][unit] = np.percentile(null_rates, 95)
            out["null_p"][unit] = (1 + int((null_rates >= observed).sum())) / (1 + args.shifts)
            out["excess_over_null"][unit] = observed - null_rates.mean()
        if unit % 64 == 0:
            print(f"unit {unit}/{num_units}", flush=True)

    pipeline_rel, pipeline_counts, pipeline_dir, pipeline_dir_counts = (
        compute_field_traversal_reliability(
            statistics,
            field_masks.reshape(num_units, statistics.num_bins_y, statistics.num_bins_x),
            threshold_mode="peak_fraction",
            threshold_fraction=args.hit_threshold,
            minimum_traversals=args.min_traversals,
            flat_headings=flat_heading,
            num_heading_sectors=sectors,
        )
    )

    args.output.mkdir(parents=True, exist_ok=True)
    enough = out["traversals"] >= args.min_traversals
    has_field = field_bins > 0

    def summary_of(values: np.ndarray, mask: np.ndarray) -> dict:
        v = values[mask & np.isfinite(values)]
        if v.size == 0:
            return {"n": 0}
        return {
            "n": int(v.size),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)),
            "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)),
            "p90": float(np.percentile(v, 90)),
        }

    summary = {
        "units": int(num_units),
        "units_with_field": int(has_field.sum()),
        "units_with_min_traversals": int(enough.sum()),
        "median_traversals_per_unit": float(np.median(out["traversals"][has_field]))
        if has_field.any()
        else None,
        "hit_strong_all_units_with_field": summary_of(out["hit_strong"], has_field),
        "hit_strong_min_traversals": summary_of(out["hit_strong"], enough),
        "hit_any_min_traversals": summary_of(out["hit_any"], enough),
        "hit_strong_core_min_traversals": summary_of(out["hit_strong_core"], enough),
        "directional_heldout": summary_of(out["directional_heldout"], enough),
        "null_mean": summary_of(out["null_mean"], enough),
        "excess_over_null": summary_of(out["excess_over_null"], enough),
        "fraction_significant_p05_min_traversals": float((out["null_p"][enough] < 0.05).mean())
        if enough.any() and args.shifts > 0
        else None,
        "pipeline_function_mean_reliability": float(np.nanmean(pipeline_rel))
        if np.isfinite(pipeline_rel).any()
        else None,
        "pipeline_function_mean_directional": float(np.nanmean(pipeline_dir))
        if np.isfinite(pipeline_dir).any()
        else None,
        "pipeline_function_assessable": int(np.isfinite(pipeline_rel).sum()),
    }
    aligned = []
    for unit in np.flatnonzero(enough & (out["preferred_sector"] >= 0)):
        aligned.append(np.roll(out["sector_hit_strong"][unit], -int(out["preferred_sector"][unit])))
    aligned_curve = np.nanmean(np.stack(aligned), axis=0).tolist() if aligned else None
    summary["sector_curve_aligned_to_preferred"] = aligned_curve

    with (args.output / "per_unit.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        header = [
            "unit",
            "field_bins",
            "field_count",
            "traversals",
            "mean_traversal_steps",
            "hit_strong",
            "hit_any",
            "core_traversals",
            "hit_strong_core",
            "preferred_sector",
            "directional_heldout",
            "directional_heldout_traversals",
            "null_mean",
            "null_p95",
            "null_p",
            "excess_over_null",
            "pipeline_reliability",
            "pipeline_directional",
        ] + [f"sector_{s}" for s in range(sectors)]
        writer.writerow(header)
        for unit in range(num_units):
            writer.writerow(
                [
                    unit,
                    int(field_bins[unit]),
                    int(field_count[unit]),
                    int(out["traversals"][unit]),
                    out["mean_traversal_steps"][unit],
                    out["hit_strong"][unit],
                    out["hit_any"][unit],
                    int(out["core_traversals"][unit]),
                    out["hit_strong_core"][unit],
                    int(out["preferred_sector"][unit]),
                    out["directional_heldout"][unit],
                    int(out["directional_heldout_traversals"][unit]),
                    out["null_mean"][unit],
                    out["null_p95"][unit],
                    out["null_p"][unit],
                    out["excess_over_null"][unit],
                    float(pipeline_rel[unit]),
                    float(pipeline_dir[unit]),
                ]
                + out["sector_hit_strong"][unit].tolist()
            )
    np.savez_compressed(
        args.output / "per_unit.npz",
        field_bins=field_bins,
        field_count=field_count,
        rate_maps=rate_maps,
        **out,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    v = out["hit_strong"][enough]
    axes[0].hist(v[np.isfinite(v)], bins=20, range=(0, 1), color="#1F77B4")
    axes[0].set(
        xlabel="strong hit rate per traversal",
        ylabel="units",
        title=f"{enough.sum()} units, >= {args.min_traversals} traversals",
    )
    d = out["directional_heldout"]
    ok = enough & np.isfinite(d) & np.isfinite(out["hit_strong"])
    axes[1].scatter(out["hit_strong"][ok], d[ok], s=6, alpha=0.6)
    axes[1].plot([0, 1], [0, 1], color="grey", linewidth=0.8)
    axes[1].set(
        xlabel="all traversals",
        ylabel="preferred heading sector (held out)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    if aligned_curve:
        axes[2].plot(range(sectors), aligned_curve, marker="o")
        axes[2].set(
            xlabel="heading sector relative to preferred",
            ylabel="mean strong hit rate",
            ylim=(0, 1),
        )
    fig.tight_layout()
    fig.savefig(args.output / "traversal_reliability.png", dpi=170)
    plt.close(fig)

    return dict(
        model_id=args.model_id,
        checkpoint=checkpoint.name,
        checkpoint_sha256=sha256(checkpoint),
        dataset_id=args.dataset,
        split_id=args.split,
        env_id=args.env_id,
        episodes=int(codes.shape[0]),
        episode_ids=list(map(int, ids)),
        valid_steps=int(valid.sum()),
        bins=args.bins,
        bounds=[list(map(float, b)) for b in statistics.bounds],
        field_threshold=args.field_threshold,
        core_threshold=args.core_threshold,
        hit_threshold=args.hit_threshold,
        min_traversals=args.min_traversals,
        sectors=sectors,
        null_shifts=args.shifts,
        null_type=(
            "within_episode_circular_activity_shift, offset >= 5% of episode length, "
            "same offsets for all units within a shift"
        ),
        summary=summary,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--field-threshold", type=float, default=0.2)
    parser.add_argument("--core-threshold", type=float, default=0.5)
    parser.add_argument("--hit-threshold", type=float, default=0.3)
    parser.add_argument("--min-traversals", type=int, default=5)
    parser.add_argument("--sectors", type=int, default=8)
    parser.add_argument("--shifts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = run(args)
    result.update(
        complete=True,
        training_performed=False,
        job_id=os.environ.get("SLURM_JOB_ID"),
    )
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result["summary"], default=str), flush=True)
