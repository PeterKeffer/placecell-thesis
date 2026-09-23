from __future__ import annotations

import csv
import math

import numpy as np

from placecell_research.measures.decoding import decode, stack_features, supported_rows
from placecell_research.measures.navigation import absorbing_tail, learning_curve
from placecell_research.measures.similarity import binned_mean, half_distance
from placecell_research.measures.single_unit import single_unit_measures
from placecell_research.measures.table import single_unit_row, summarize
from placecell_research.measures.traversal import traversal_measures, traversal_summary

BOUNDS = ((0.0, 10.0), (0.0, 10.0))


def random_walk(episodes: int, steps: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    heading = np.cumsum(rng.normal(0.0, 0.4, (episodes, steps)), axis=1)
    position = np.empty((episodes, steps, 2), dtype=np.float32)
    position[:, 0] = rng.uniform(1.0, 9.0, (episodes, 2))
    for step in range(1, steps):
        move = 0.4 * np.stack([np.cos(heading[:, step]), np.sin(heading[:, step])], axis=-1)
        position[:, step] = np.clip(position[:, step - 1] + move, 0.05, 9.95)
    return position, np.angle(np.exp(1j * heading)).astype(np.float32)


def place_codes(position: np.ndarray, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    distance = np.linalg.norm(position - np.array([3.0, 7.0]), axis=-1)
    tuned = np.exp(-(distance**2) / 2.0)
    tuned[tuned < 0.2] = 0.0
    silent = np.zeros_like(tuned)
    noise = np.maximum(rng.normal(0.0, 1.0, tuned.shape), 0.0)
    return np.stack([tuned, silent, noise], axis=-1).astype(np.float32)


def test_single_unit_measures_separate_tuned_silent_and_noise_units():
    position, heading = random_walk(12, 300)
    codes = place_codes(position)
    valid = np.ones(codes.shape[:2], dtype=bool)
    per_unit, population = single_unit_measures(
        codes, position, heading, valid, BOUNDS, env_id="", null_shuffles=19
    )
    assert per_unit["active_step_count"][1] == 0
    assert per_unit["spatial_information_bits"][0] > per_unit["spatial_information_null_95"][0]
    assert per_unit["variance_explained_held_out"][0] > per_unit["variance_explained_held_out"][2]
    assert per_unit["field_component_count"][0] >= 1
    assert population["visited_bin_count"] > 0

    row, arrays = single_unit_row(per_unit, population["visited_bin_count"])
    assert math.isclose(row["silent_fraction"], 1.0 / 3.0)
    assert row["information_above_null95_all_units_mean"] == float(
        arrays["information_above_null95"].mean()
    )
    assert row["units"] == 3


def test_ridge_decoder_recovers_a_linear_position_code():
    rng = np.random.default_rng(0)
    projection = rng.normal(size=(2, 6))
    features, targets, counts = {}, {}, []
    for split in ("train", "validation", "test"):
        position, heading = random_walk(6, 120, seed=len(split))
        codes = np.concatenate(
            [position @ projection, np.sin(heading)[..., None], np.cos(heading)[..., None]],
            axis=-1,
        )
        codes += rng.normal(0.0, 0.01, codes.shape)
        valid = np.ones(codes.shape[:2], dtype=bool)
        features[split], targets[split], counts = supported_rows(
            stack_features(codes, 1), position, heading, valid
        )
    scores = decode(features, targets, counts, nonlinear=False)
    assert scores["position_ridge_rmse"] < 0.05
    assert scores["position_ridge_r2"] > 0.99
    assert scores["heading_ridge_median_error_degrees"] < 2.0
    assert scores["position_ridge_shift_control_rmse"] > 1.0


def test_stack_features_keeps_sixteen_causal_frames():
    values = np.arange(2 * 20 * 1, dtype=np.float32).reshape(2, 20, 1)
    stacked = stack_features(values, 16)
    assert stacked.shape == (2, 5, 16)
    assert stacked[0, 0, 0] == values[0, 15, 0]
    assert stacked[0, 0, 15] == values[0, 0, 0]


def test_half_distance_interpolates_between_bins():
    centers = np.array([0.5, 1.5, 2.5, 3.5])
    assert math.isclose(half_distance(centers, np.array([1.0, 0.8, 0.4, 0.1])), 2.25)
    assert math.isnan(half_distance(centers, np.array([1.0, 0.9, 0.8, 0.7])))


def test_binned_mean_drops_thin_bins():
    distance = np.concatenate([np.full(5000, 0.5), np.full(10, 1.5)])
    similarity = np.concatenate([np.full(5000, 0.9), np.full(10, 0.1)])
    means = binned_mean(distance, similarity, np.array([0.0, 1.0, 2.0]))
    assert math.isclose(means[0], 0.9)
    assert math.isnan(means[1])


def test_traversal_measures_score_a_unit_that_fires_on_every_pass():
    steps = np.arange(400)
    x = 5.0 + 4.5 * np.sin(steps / 10.0)
    position = np.stack([x, np.full_like(x, 5.0)], axis=-1)[None].repeat(4, axis=0)
    heading = np.where(np.cos(steps / 10.0) > 0, 0.0, np.pi)[None].repeat(4, axis=0)
    codes = np.stack([(np.abs(x - 8.0) < 1.0).astype(np.float32), np.zeros_like(x)], axis=-1)
    codes = codes[None].repeat(4, axis=0).astype(np.float32)
    valid = np.ones((4, 400), dtype=bool)
    out = traversal_measures(codes, position, heading, valid, env_id="", shifts=5)
    assert out["field_bins"][1] == 0
    assert out["traversals"][0] > 10
    assert out["hit_strong"][0] == 1.0
    summary = traversal_summary(out)
    assert summary["traversal_units_with_field"] == 1
    assert summary["traversal_response_median"] == 1.0


def test_traversal_shift_null_is_identical_in_chunks():
    position, heading = random_walk(6, 200)
    codes = place_codes(position)
    valid = np.ones(codes.shape[:2], dtype=bool)
    valid[2, 150:] = False
    unchunked = traversal_measures(
        codes, position, heading, valid, env_id="", shifts=11, shift_chunk_size=11
    )
    assert np.isfinite(unchunked["null_mean"]).any()
    for chunk_size in (1, 4):
        chunked = traversal_measures(
            codes, position, heading, valid, env_id="", shifts=11, shift_chunk_size=chunk_size
        )
        assert chunked.keys() == unchunked.keys()
        for key, value in unchunked.items():
            np.testing.assert_array_equal(chunked[key], value)


def test_summarize_reports_mean_and_sample_sd(tmp_path):
    paths = []
    for seed, value in ((42, 1.0), (1, 3.0)):
        path = tmp_path / f"a_{seed}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, ["condition", "training_seed", "model", "bits"])
            writer.writeheader()
            writer.writerow({"condition": "a", "training_seed": seed, "model": "m", "bits": value})
        paths.append(path)
    (row,) = summarize(paths)
    assert row["n"] == 2
    assert row["bits_mean"] == 2.0
    assert math.isclose(row["bits_sd"], math.sqrt(2.0))
    assert "model_mean" not in row


def test_absorbing_tail_flags_period_two_loops_at_the_step_limit():
    loop = [[1.0, 2.0, 0.0], [1.0, 2.0, 0.3]] * 40
    assert absorbing_tail(False, 80, 80, loop)
    assert not absorbing_tail(True, 80, 80, loop)
    assert not absorbing_tail(False, 79, 80, loop[:79])
    walk = [[float(step), 0.0, 0.0] for step in range(80)]
    assert not absorbing_tail(False, 80, 80, walk)


def test_learning_curve_finds_three_sustained_bins():
    history = [
        {"timestep": step, "success_rate": float(step > 300_000)}
        for step in range(10_000, 1_000_001, 10_000)
    ]
    curve = learning_curve(history, 1_000_000)
    assert curve["steps_to_sustained_80_percent"] == 400_000
    assert curve["training_success_first_million"] == 0.7
