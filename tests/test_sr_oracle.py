from __future__ import annotations

import numpy as np
import pytest

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.sr_oracle import (
    SuccessorOracle,
    SuccessorOracleComparisonModule,
    build_successor_oracle,
    compare_to_oracle,
    eigen_comparison,
    oracle_features,
)


def _wallgap_trajectories(episodes: int = 300, time: int = 250, seed: int = 0):
    """A random walk in a 10x10 box with a barrier at x=5 for y<7 -- a toy WallGap."""
    rng = np.random.default_rng(seed)
    positions = np.zeros((episodes, time, 2))
    point = rng.uniform(0.5, 9.5, size=(episodes, 2))
    for step in range(time):
        proposal = np.clip(point + rng.normal(0, 0.45, size=(episodes, 2)), 0.05, 9.95)
        blocked = (np.sign(point[:, 0] - 5.0) != np.sign(proposal[:, 0] - 5.0)) & (
            proposal[:, 1] < 7.0
        )
        proposal[blocked] = point[blocked]
        point = proposal
        positions[:, step] = point
    return positions, np.ones((episodes, time), dtype=bool)


def _oracle(gamma: float = 0.9):
    positions, valid = _wallgap_trajectories()
    return build_successor_oracle(
        positions, valid, discount_gamma=gamma, num_bins_x=10, num_bins_y=10
    ), positions


def test_normalised_successor_rows_sum_to_one() -> None:
    oracle, _ = _oracle()
    assert np.allclose(oracle.successor_matrix.sum(axis=1), 1.0, atol=1e-6)


def test_transition_matrix_is_row_stochastic() -> None:
    oracle, _ = _oracle()
    assert np.allclose(oracle.transition_matrix.sum(axis=1), 1.0, atol=1e-9)


def test_successor_respects_the_wall() -> None:
    oracle, _ = _oracle()
    across = oracle.successor_matrix[oracle.state_of_bin[4, 2], oracle.state_of_bin[5, 2]]
    same_side = oracle.successor_matrix[oracle.state_of_bin[4, 2], oracle.state_of_bin[4, 6]]
    assert same_side > 100 * max(across, 1e-12)


def test_gamma_one_is_rejected() -> None:
    positions, valid = _wallgap_trajectories(episodes=20, time=30)
    with pytest.raises(ValueError, match="discount_gamma"):
        build_successor_oracle(positions, valid, discount_gamma=1.0)


def test_transitions_never_cross_invalid_steps() -> None:
    positions, valid = _wallgap_trajectories(episodes=60, time=60)
    positions[:, 30:] = 999.0
    valid[:, 30:] = False
    oracle = build_successor_oracle(
        positions, valid, discount_gamma=0.9, num_bins_x=6, num_bins_y=6
    )
    assert np.isfinite(oracle.successor_matrix).all()
    assert np.allclose(oracle.successor_matrix.sum(axis=1), 1.0, atol=1e-6)


def test_rsa_separates_a_successor_representation_from_a_perfect_position_code() -> None:
    oracle, positions = _oracle()
    rng = np.random.default_rng(1)
    features = rng.normal(size=(oracle.num_states, 8))
    states = oracle.states_for_positions(positions.reshape(-1, 2))
    index = np.clip(states, 0, None)

    truth = compare_to_oracle(oracle_features(oracle, features)[index], states, oracle, features)
    position_only = compare_to_oracle(np.eye(oracle.num_states)[index], states, oracle, features)
    noise = compare_to_oracle(rng.normal(size=(len(index), 16)), states, oracle, features)

    assert truth["oracle_rsa"] > 0.95, "the true successor features must score ~1"
    assert position_only["oracle_rsa"] < 0.1, "a perfect POSITION code must NOT pass as an SR"
    assert noise["oracle_rsa"] < 0.1
    assert truth["oracle_rsa"] > truth["euclidean_rsa_baseline"] + 0.3


def test_euclidean_baseline_is_the_bar_and_is_well_above_zero() -> None:
    oracle, positions = _oracle()
    rng = np.random.default_rng(2)
    features = rng.normal(size=(oracle.num_states, 8))
    states = oracle.states_for_positions(positions.reshape(-1, 2))
    report = compare_to_oracle(
        oracle_features(oracle, features)[np.clip(states, 0, None)], states, oracle, features
    )
    assert 0.15 < report["euclidean_rsa_baseline"] < 0.95
    assert report["position_rsa_baseline"] == pytest.approx(0.0, abs=0.1)


def test_occupancy_is_reported_for_the_sampling_confound() -> None:
    oracle, positions = _oracle()
    rng = np.random.default_rng(3)
    features = rng.normal(size=(oracle.num_states, 8))
    states = oracle.states_for_positions(positions.reshape(-1, 2))
    report = compare_to_oracle(
        oracle_features(oracle, features)[np.clip(states, 0, None)], states, oracle, features
    )
    assert report["occupancy_max_over_median"] >= 1.0
    assert oracle.occupancy.shape[0] == oracle.num_states


def test_oracle_features_rejects_a_mismatched_feature_matrix() -> None:
    oracle, _ = _oracle()
    with pytest.raises(ValueError, match="rows"):
        oracle_features(oracle, np.zeros((oracle.num_states + 3, 4)))


def test_analysis_module_reports_successor_geometry(tmp_path) -> None:
    positions, valid = _wallgap_trajectories(episodes=80, time=100)
    oracle = build_successor_oracle(
        positions,
        valid,
        discount_gamma=0.9,
        num_bins_x=8,
        num_bins_y=8,
    )
    states = oracle.states_for_positions(positions)
    rng = np.random.default_rng(5)
    features_by_state = rng.normal(size=(oracle.num_states, 8))
    state_indices = np.clip(states, 0, None)
    successor = oracle_features(oracle, features_by_state)[state_indices]
    feature = features_by_state[state_indices]

    def analysis_input(representation: np.ndarray, label: str) -> AnalysisInput:
        return AnalysisInput(
            representation=representation,
            position_xy=positions,
            heading=None,
            kinematics=None,
            actions=None,
            valid_mask=valid,
            source_name=label,
            label=label,
            split_name="test",
        )

    result = SuccessorOracleComparisonModule().run(
        [analysis_input(successor, "successor"), analysis_input(feature, "feature")],
        ["successor", "feature"],
        tmp_path,
        {
            "sr_oracle_discount_gamma": 0.9,
            "sr_oracle_num_bins_x": 8,
            "sr_oracle_num_bins_y": 8,
        },
    )

    assert result.metrics["oracle_rsa"] > 0.99
    assert result.metadata["sr_oracle_num_states"] == oracle.num_states


def _random_walk(wall: bool, episodes: int = 200, steps: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    positions = np.zeros((episodes, steps, 2))
    point = np.stack([rng.uniform(0.5, 19.5, episodes), rng.uniform(0.5, 19.5, episodes)], 1)
    for step in range(steps):
        proposal = np.clip(point + rng.normal(0, 0.7, (episodes, 2)), 0.01, 19.99)
        if wall:
            crossed = ((point[:, 0] < 10) & (proposal[:, 0] >= 10)) | (
                (point[:, 0] >= 10) & (proposal[:, 0] < 10)
            )
            through = (proposal[:, 1] >= 8) & (proposal[:, 1] < 12)
            proposal[crossed & ~through, 0] = point[crossed & ~through, 0]
        point = proposal
        positions[:, step] = point
    return positions


def _oracle_for(wall: bool):
    positions = _random_walk(wall)
    return positions, build_successor_oracle(
        positions, np.ones(positions.shape[:2], dtype=bool), discount_gamma=0.95
    )


def test_leading_eigenvector_fragments_only_when_a_barrier_exists() -> None:
    _, open_field = _oracle_for(wall=False)
    _, wall_gap = _oracle_for(wall=True)
    open_ideal = oracle_features(open_field, np.eye(open_field.num_states))
    wall_ideal = oracle_features(wall_gap, np.eye(wall_gap.num_states))
    open_metrics = eigen_comparison(open_field, open_ideal)
    wall_metrics = eigen_comparison(wall_gap, wall_ideal)
    assert open_metrics["eigen_leading_fragmentation"] < 1.5
    assert wall_metrics["eigen_leading_fragmentation"] > 2.5
    assert (
        wall_metrics["eigen_leading_fragmentation"]
        > 2.0 * open_metrics["eigen_leading_fragmentation"]
    )


def test_alignment_separates_a_true_sr_from_a_position_code() -> None:
    positions, oracle = _oracle_for(wall=True)
    ideal = oracle_features(oracle, np.eye(oracle.num_states))
    centres_x = 0.5 * (oracle.bin_edges_x[:-1] + oracle.bin_edges_x[1:])
    centres_y = 0.5 * (oracle.bin_edges_y[:-1] + oracle.bin_edges_y[1:])
    grid_x, grid_y = np.meshgrid(centres_x, centres_y, indexing="ij")
    coordinates = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    position_code = np.zeros((oracle.num_states, 2))
    for bin_index, state in enumerate(oracle.state_of_bin.reshape(-1)):
        if state >= 0:
            position_code[state] = coordinates[bin_index]

    true_sr = eigen_comparison(oracle, ideal)
    position_only = eigen_comparison(oracle, np.repeat(position_code, 4, axis=1))
    assert true_sr["eigen_alignment"] > 0.7
    assert position_only["eigen_alignment"] < 0.6
    assert position_only["eigen_alignment"] > 10 * true_sr["eigen_alignment_chance"]
    assert true_sr["eigen_alignment_chance"] == pytest.approx(8 / oracle.num_states)


def test_constant_mode_is_skipped() -> None:
    _, oracle = _oracle_for(wall=True)
    values, vectors = np.linalg.eig(oracle.successor_matrix)
    leading = np.real(vectors[:, np.argsort(-values.real)[0]])
    assert leading.std() / abs(leading.mean()) < 1e-6, "mode 0 should be constant"
    metrics = eigen_comparison(oracle, oracle_features(oracle, np.eye(oracle.num_states)))
    assert metrics["eigen_num_modes"] == 8.0


def test_eigen_comparison_uses_only_feasible_nonconstant_modes() -> None:
    transition = np.asarray(
        [
            [0.5, 0.5, 0.0],
            [0.25, 0.5, 0.25],
            [0.0, 0.5, 0.5],
        ]
    )
    gamma = 0.9
    successor = (1.0 - gamma) * np.linalg.inv(np.eye(len(transition)) - gamma * transition)
    oracle = SuccessorOracle(
        successor_matrix=successor,
        transition_matrix=transition,
        occupancy=np.ones(3),
        state_of_bin=np.arange(3).reshape(3, 1),
        bin_edges_x=np.arange(4),
        bin_edges_y=np.arange(2),
        discount_gamma=gamma,
        dropped_bins=0,
    )

    metrics = eigen_comparison(oracle, successor, num_modes=8)

    assert metrics["eigen_num_modes"] == 2.0
    assert metrics["eigen_alignment_chance"] == pytest.approx(2.0 / 3.0)
    assert metrics["eigen_alignment"] == pytest.approx(1.0)
