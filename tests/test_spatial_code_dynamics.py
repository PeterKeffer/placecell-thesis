from __future__ import annotations

import numpy as np
import pytest

from placecell_research.analysis.spatial_code_dynamics import (
    compute_lag_code_dynamics,
    shuffled_support_dwell_lengths,
    support_dwell_lengths,
    wallgap_room_ids,
)

WALLGAP_ASYM_LARGE_ENV_ID = "MiniWorld-WallGapAsymLarge-v0"


def test_lag_dynamics_reports_exact_cosine_and_support_change_metrics() -> None:
    representation = np.asarray(
        [[[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 6.0]]],
        dtype=np.float32,
    )
    positions = np.asarray(
        [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]],
        dtype=np.float32,
    )
    valid_mask = np.ones((1, 4), dtype=bool)

    metrics = compute_lag_code_dynamics(
        representation,
        positions,
        valid_mask,
        lag=1,
        cosine_event_thresholds=(0.5,),
    )

    assert metrics["pair_count"] == 3
    assert metrics["cosine_distance_mean"] == pytest.approx(1.0 / 3.0)
    assert metrics["cosine_distance_median"] == pytest.approx(0.0)
    assert metrics["cosine_distance_p90"] == pytest.approx(0.8)
    assert metrics["support_change_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["support_jaccard_distance_mean"] == pytest.approx(1.0 / 3.0)
    assert metrics["position_displacement_mean"] == pytest.approx(1.0)
    assert metrics["cosine_event_fraction_ge_0p5"] == pytest.approx(1.0 / 3.0)
    assert metrics["cosine_event_mean_interval_steps_ge_0p5"] == pytest.approx(3.0)


def test_lag_dynamics_never_pairs_across_episode_or_invalid_step_boundaries() -> None:
    representation = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    positions = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.asarray([[True, True, False], [True, True, True]])

    lag_one = compute_lag_code_dynamics(
        representation,
        positions,
        valid_mask,
        lag=1,
    )
    lag_two = compute_lag_code_dynamics(
        representation,
        positions,
        valid_mask,
        lag=2,
    )

    assert lag_one["pair_count"] == 3
    assert lag_one["cosine_distance_mean"] == pytest.approx(0.0)
    assert lag_one["support_change_fraction"] == pytest.approx(0.0)
    assert lag_two["pair_count"] == 1
    assert lag_two["support_change_fraction"] == pytest.approx(0.0)


def test_support_dwell_lengths_use_exact_support_and_reset_per_episode() -> None:
    support_a = [1.0, 0.0]
    support_b = [0.0, 1.0]
    representation = np.asarray(
        [
            [support_a, support_a, support_b, support_b, support_b, support_a],
            [support_a, support_a, support_b, support_b, support_b, support_b],
        ],
        dtype=np.float32,
    )
    valid_mask = np.asarray(
        [
            [True, True, True, True, True, True],
            [True, True, True, False, False, False],
        ]
    )

    dwell_lengths = support_dwell_lengths(representation, valid_mask)

    np.testing.assert_array_equal(dwell_lengths, np.asarray([2, 3, 1, 2, 1]))


def test_wallgap_room_ids_and_transition_conditioned_dynamics() -> None:
    positions = np.asarray(
        [[[0.0, 0.0], [1.0, 0.0], [7.0, 0.0], [8.0, 0.0]]],
        dtype=np.float32,
    )
    representation = np.asarray(
        [[[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]]],
        dtype=np.float32,
    )
    valid_mask = np.ones((1, 4), dtype=bool)
    room_ids = wallgap_room_ids(positions, env_id=WALLGAP_ASYM_LARGE_ENV_ID)

    np.testing.assert_array_equal(
        room_ids,
        np.asarray(
            [
                [
                    "central_corridor",
                    "central_corridor",
                    "southern_yard_right",
                    "southern_yard_right",
                ]
            ]
        ),
    )

    metrics = compute_lag_code_dynamics(
        representation,
        positions,
        valid_mask,
        lag=1,
        room_ids=room_ids,
    )

    assert metrics["room_transition_pair_count"] == 1
    assert metrics["same_room_pair_count"] == 2
    assert metrics["room_transition_cosine_distance_mean"] == pytest.approx(1.0)
    assert metrics["same_room_cosine_distance_mean"] == pytest.approx(0.0)
    assert metrics["movement_matched_same_room_cosine_distance_mean"] == pytest.approx(0.0)
    assert metrics["room_transition_cosine_distance_excess"] == pytest.approx(1.0)
    assert metrics["room_transition_support_change_fraction"] == pytest.approx(1.0)
    assert metrics["same_room_support_change_fraction"] == pytest.approx(0.0)


def test_wallgap_room_ids_cover_all_rooms_and_outside() -> None:
    positions = np.asarray(
        [
            [
                [0.0, 12.0],
                [0.0, 0.0],
                [-12.0, -12.0],
                [12.0, -12.0],
                [30.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    room_ids = wallgap_room_ids(positions, env_id=WALLGAP_ASYM_LARGE_ENV_ID)

    np.testing.assert_array_equal(
        room_ids,
        np.asarray(
            [
                [
                    "northern_courtyard",
                    "central_corridor",
                    "southern_yard_left",
                    "southern_yard_right",
                    "outside",
                ]
            ]
        ),
    )


def _random_kwinner_codes(
    *, episodes: int, steps: int, code_dim: int, k: int, seed: int
) -> np.ndarray:
    """Temporally INDEPENDENT k-winner codes: no temporal structure to find, by construction."""
    generator = np.random.default_rng(seed)
    codes = np.zeros((episodes, steps, code_dim), dtype=np.float32)
    for episode in range(episodes):
        for step in range(steps):
            winners = generator.choice(code_dim, size=k, replace=False)
            codes[episode, step, winners] = 1.0
    return codes


def test_shuffled_null_recovers_the_dimension_effect_on_temporally_independent_codes() -> None:
    shape = {"episodes": 4, "steps": 400, "k": 4}
    narrow = _random_kwinner_codes(code_dim=16, seed=0, **shape)
    wide = _random_kwinner_codes(code_dim=64, seed=1, **shape)
    valid = np.ones((shape["episodes"], shape["steps"]), dtype=bool)

    narrow_dwell = float(np.mean(support_dwell_lengths(narrow, valid)))
    wide_dwell = float(np.mean(support_dwell_lengths(wide, valid)))
    assert narrow_dwell > wide_dwell, "narrower code must dwell longer by chance alone"

    narrow_ratio = narrow_dwell / float(
        np.mean(shuffled_support_dwell_lengths(narrow, valid, seed=0))
    )
    wide_ratio = wide_dwell / float(np.mean(shuffled_support_dwell_lengths(wide, valid, seed=0)))
    assert narrow_ratio == pytest.approx(1.0, abs=0.15)
    assert wide_ratio == pytest.approx(1.0, abs=0.15)


def test_shuffled_null_leaves_genuine_temporal_persistence_visible() -> None:
    base = _random_kwinner_codes(episodes=4, steps=50, code_dim=16, k=4, seed=2)
    held = np.repeat(base, 8, axis=1)
    valid = np.ones(held.shape[:2], dtype=bool)

    observed = float(np.mean(support_dwell_lengths(held, valid)))
    shuffled = float(np.mean(shuffled_support_dwell_lengths(held, valid, seed=0)))
    assert observed / shuffled > 3.0, "real persistence must survive the null"


def test_shuffled_null_preserves_the_code_multiset_per_episode() -> None:
    codes = _random_kwinner_codes(episodes=3, steps=60, code_dim=16, k=4, seed=3)
    valid = np.ones(codes.shape[:2], dtype=bool)
    lengths = shuffled_support_dwell_lengths(codes, valid, seed=0)
    assert int(lengths.sum()) == int(valid.sum())


def test_shuffled_null_respects_the_valid_mask() -> None:
    codes = _random_kwinner_codes(episodes=2, steps=40, code_dim=16, k=4, seed=4)
    valid = np.ones(codes.shape[:2], dtype=bool)
    valid[:, 20:] = False
    lengths = shuffled_support_dwell_lengths(codes, valid, seed=0)
    assert int(lengths.sum()) == int(valid.sum()) == 40
