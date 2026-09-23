"""Response of each unit on single traversals of its field, by heading sector, against shifts."""

from __future__ import annotations

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.helpers import compute_rate_maps_from_episode_statistics
from placecell_research.analysis.occupancy import get_or_compute_episode_bin_statistics
from placecell_research.numerics.rate_map_kernels import compute_place_field_mask

BINS = 60
FIELD_THRESHOLD = 0.2
CORE_THRESHOLD = 0.5
HIT_THRESHOLD = 0.3
PEAK_QUANTILE = 0.995
MIN_TRAVERSALS = 5
SECTORS = 8
SEED = 0
SHIFT_CHUNK = 32


def episode_spans(episode_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start index and length of every episode in the flattened valid-step order."""
    change = np.flatnonzero(np.diff(episode_ids)) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [episode_ids.shape[0]]]))
    return starts, lengths


def shifted_index(starts: np.ndarray, lengths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Index array that rolls every episode by its own random offset (at least 5% of its length)."""
    index = np.empty(int(lengths.sum()), dtype=np.int64)
    for start, length in zip(starts, lengths, strict=True):
        minimum = max(1, int(0.05 * length))
        offset = int(rng.integers(minimum, max(minimum + 1, length - minimum + 1)))
        index[start : start + length] = start + (np.arange(length) + offset) % length
    return index


def traversal_table(
    in_field: np.ndarray, segment_start: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (traversal_start_indices, traversal_id_per_step); id is -1 outside the field."""
    left_field = np.empty_like(in_field)
    left_field[0] = True
    left_field[1:] = ~in_field[:-1]
    starts = in_field & (left_field | segment_start)
    traversal_id = np.cumsum(starts) - 1
    traversal_id[~in_field] = -1
    return np.flatnonzero(starts), traversal_id


def hits_per_traversal(active: np.ndarray, start_indices: np.ndarray) -> np.ndarray:
    """One boolean per traversal: any active step between its start and the next start."""
    if start_indices.size == 0:
        return np.zeros(0, dtype=bool)
    return np.logical_or.reduceat(active, start_indices)


def shifted_hit_rates(
    field_masks: np.ndarray,
    strong_all: np.ndarray,
    spatial_bins: np.ndarray,
    segment_start: np.ndarray,
    units: list[int],
    starts: np.ndarray,
    lengths: np.ndarray,
    shifts: int,
    chunk_size: int,
) -> np.ndarray:
    """Hit rate of every unit under each circular shift, drawing chunk_size shifts at a time."""
    rng = np.random.default_rng(SEED)
    rates = np.full((field_masks.shape[0], shifts), np.nan, dtype=np.float32)
    for chunk_start in range(0, shifts, chunk_size):
        chunk = [
            shifted_index(starts, lengths, rng)
            for _ in range(min(chunk_size, shifts - chunk_start))
        ]
        for unit in units:
            in_field = field_masks[unit][spatial_bins]
            start_indices, _ = traversal_table(in_field, segment_start)
            strong = strong_all[:, unit]
            for offset, index in enumerate(chunk):
                rates[unit, chunk_start + offset] = hits_per_traversal(
                    in_field & strong[index], start_indices
                ).mean()
    return rates


def traversal_measures(
    codes: np.ndarray,
    position_xy: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    *,
    env_id: str,
    shifts: int,
    shift_chunk_size: int = SHIFT_CHUNK,
) -> dict[str, np.ndarray]:
    """Per-unit traversal response, core response, heading sectors and shift null."""
    num_units = codes.shape[-1]
    positive = np.maximum(codes, 0.0).astype(np.float32)
    data = AnalysisInput(
        representation=positive,
        position_xy=position_xy,
        heading=heading,
        kinematics=None,
        actions=None,
        valid_mask=valid,
        source_name="encoder.place_codes",
        label="encoder.place_codes",
        split_name="test",
        metadata={"env_id": env_id},
    )
    statistics = get_or_compute_episode_bin_statistics(
        data, num_bins_x=BINS, num_bins_y=BINS, bounds=None
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
    for unit in range(num_units):
        mask, _, area = compute_place_field_mask(rate_maps[unit], FIELD_THRESHOLD)
        core, _, _ = compute_place_field_mask(rate_maps[unit], CORE_THRESHOLD)
        field_masks[unit] = mask.reshape(-1)
        core_masks[unit] = core.reshape(-1)
        field_bins[unit] = int(area)

    flat = statistics.flat_values
    spatial_bins = (statistics.episode_bin_codes % num_bins).astype(np.int64)
    episode_ids = (statistics.episode_bin_codes // num_bins).astype(np.int64)
    segment_start = statistics.segment_start.astype(bool)
    flat_heading = heading[valid].astype(np.float64).reshape(-1)
    if flat_heading.shape[0] != flat.shape[0]:
        raise ValueError("heading and flat values disagree")
    peaks = np.array(
        [np.quantile(flat[:, u].astype(np.float64), PEAK_QUANTILE) for u in range(num_units)],
        dtype=np.float32,
    )
    strong_all = (flat >= (HIT_THRESHOLD * peaks)[None, :]) & (flat > 0)
    starts, lengths = episode_spans(episode_ids)
    scored_units: list[int] = []

    sector_width = 2.0 * np.pi / SECTORS
    out = {
        "field_bins": field_bins,
        "traversals": np.zeros(num_units, dtype=np.int32),
        "hit_strong": np.full(num_units, np.nan, dtype=np.float32),
        "hit_strong_core": np.full(num_units, np.nan, dtype=np.float32),
        "sector_hit_strong": np.full((num_units, SECTORS), np.nan, dtype=np.float32),
        "preferred_sector": np.full(num_units, -1, dtype=np.int32),
        "directional_heldout": np.full(num_units, np.nan, dtype=np.float32),
        "null_mean": np.full(num_units, np.nan, dtype=np.float32),
        "null_p": np.full(num_units, np.nan, dtype=np.float32),
    }
    for unit in range(num_units):
        if field_bins[unit] == 0:
            continue
        in_field = field_masks[unit][spatial_bins]
        if not in_field.any():
            continue
        start_indices, traversal_id = traversal_table(in_field, segment_start)
        count = start_indices.size
        out["traversals"][unit] = count
        if count == 0:
            continue
        strong = strong_all[:, unit]
        hit_strong = hits_per_traversal(in_field & strong, start_indices)
        out["hit_strong"][unit] = hit_strong.mean()
        core_traversal = hits_per_traversal(
            in_field & core_masks[unit][spatial_bins], start_indices
        )
        if core_traversal.any():
            out["hit_strong_core"][unit] = hit_strong[core_traversal].mean()
        in_ids = traversal_id[in_field]
        sin_sum = np.bincount(in_ids, weights=np.sin(flat_heading[in_field]), minlength=count)
        cos_sum = np.bincount(in_ids, weights=np.cos(flat_heading[in_field]), minlength=count)
        angle = np.arctan2(sin_sum, cos_sum)
        sector = (np.floor((angle + np.pi) / sector_width).astype(np.int64)) % SECTORS
        totals = np.bincount(sector, minlength=SECTORS)
        hits = np.bincount(sector[hit_strong], minlength=SECTORS)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["sector_hit_strong"][unit] = np.where(
                totals > 0, hits / np.maximum(totals, 1), np.nan
            )
        held_out = (episode_ids[start_indices] % 2) == 1
        selection_total = np.bincount(sector[~held_out], minlength=SECTORS)
        selection_hits = np.bincount(sector[~held_out & hit_strong], minlength=SECTORS)
        held_out_total = np.bincount(sector[held_out], minlength=SECTORS)
        held_out_hits = np.bincount(sector[held_out & hit_strong], minlength=SECTORS)
        eligible = (selection_total >= MIN_TRAVERSALS) & (held_out_total >= MIN_TRAVERSALS)
        if eligible.any():
            rates = np.where(eligible, selection_hits / np.maximum(selection_total, 1), -1.0)
            best = int(np.argmax(rates))
            out["preferred_sector"][unit] = best
            out["directional_heldout"][unit] = held_out_hits[best] / held_out_total[best]
        scored_units.append(unit)
    if shifts > 0:
        null_rates = shifted_hit_rates(
            field_masks,
            strong_all,
            spatial_bins,
            segment_start,
            scored_units,
            starts,
            lengths,
            shifts,
            max(1, int(shift_chunk_size)),
        )
        for unit in scored_units:
            out["null_mean"][unit] = null_rates[unit].mean()
            out["null_p"][unit] = (1 + int((null_rates[unit] >= out["hit_strong"][unit]).sum())) / (
                1 + shifts
            )
    preferred = out["preferred_sector"]
    safe = np.clip(preferred, 0, SECTORS - 1)
    out["opposite_heading"] = np.where(
        preferred >= 0,
        out["sector_hit_strong"][np.arange(num_units), (safe + SECTORS // 2) % SECTORS],
        np.nan,
    ).astype(np.float32)
    return out


def traversal_summary(out: dict[str, np.ndarray]) -> dict[str, float]:
    """Medians over units with a field, as in the thesis traversal table."""

    def median(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else float("nan")

    has_field = out["field_bins"] > 0
    scored = np.isfinite(out["hit_strong"])
    response = out["hit_strong"][scored]
    quartiles = np.percentile(response, [25, 75]) if response.size else [np.nan, np.nan]
    return {
        "traversal_units_with_field": int(has_field.sum()),
        "traversal_count_median": float(np.median(out["traversals"][has_field]))
        if has_field.any()
        else float("nan"),
        "traversal_response_median": median(out["hit_strong"]),
        "traversal_response_q25": float(quartiles[0]),
        "traversal_response_q75": float(quartiles[1]),
        "traversal_core_response_median": median(out["hit_strong_core"]),
        "traversal_preferred_heading_held_out_median": median(out["directional_heldout"]),
        "traversal_opposite_heading_median": median(out["opposite_heading"]),
        "traversal_chance_median": median(out["null_mean"][scored]),
        "traversal_units_above_chance": int(np.sum(out["null_p"][scored] < 0.05)),
    }
