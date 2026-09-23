from __future__ import annotations

import numpy as np

from placecell_research.evaluation.decode import (
    _split_indices,
    chunked_ridge_fit_predict,
    linear_decode_position,
    linear_decode_position_transfer,
    nonlinear_decode_position,
)


def test_linear_decode_position_recovers_linear_mapping() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(256, 8)).astype(np.float32)
    readout = rng.normal(size=(8, 2)).astype(np.float32)
    positions = features @ readout
    result = linear_decode_position(features, positions, train_fraction=0.75, include_shuffle=True)
    assert result.rmse < 1e-3
    assert result.r2 > 0.99
    assert result.shuffle_rmse is not None


def test_nonlinear_decode_position_recovers_nonlinear_mapping() -> None:
    rng = np.random.default_rng(5)
    features = rng.uniform(-1.0, 1.0, size=(512, 2)).astype(np.float32)
    positions = np.stack(
        [
            features[:, 0] * features[:, 1],
            np.square(features[:, 0]) - np.square(features[:, 1]),
        ],
        axis=-1,
    ).astype(np.float32)

    linear = linear_decode_position(features, positions, train_fraction=0.75, include_shuffle=False)
    nonlinear = nonlinear_decode_position(
        features,
        positions,
        train_fraction=0.75,
        hidden_sizes=(64, 64),
        max_epochs=180,
        batch_size=128,
        random_seed=7,
        include_shuffle=False,
    )

    assert nonlinear.r2 > 0.8
    assert nonlinear.rmse < linear.rmse * 0.6


def test_linear_decode_position_matches_sklearn_ridge() -> None:
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(2)
    features = rng.normal(size=(500, 16)).astype(np.float32)
    readout = rng.normal(size=(16, 2)).astype(np.float32)
    positions = (features @ readout + 0.3 * rng.normal(size=(500, 2))).astype(np.float32)
    alpha = 1.0e-2

    train_indices, validation_indices = _split_indices(len(features), 0.8)
    reference = Ridge(alpha=alpha, fit_intercept=True)
    reference.fit(features[train_indices], positions[train_indices])
    expected = reference.predict(features[validation_indices])

    result = linear_decode_position(
        features, positions, train_fraction=0.8, alpha=alpha, include_shuffle=False
    )
    assert np.allclose(result.predictions, expected, atol=1e-3)


def test_linear_decode_position_is_invariant_to_chunk_size() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(400, 12)).astype(np.float32)
    readout = rng.normal(size=(12, 2)).astype(np.float32)
    positions = (features @ readout + 0.1 * rng.normal(size=(400, 2))).astype(np.float32)

    tiny_chunks = linear_decode_position(
        features, positions, train_fraction=0.8, include_shuffle=False, chunk_size=13
    )
    one_chunk = linear_decode_position(
        features, positions, train_fraction=0.8, include_shuffle=False, chunk_size=100_000
    )
    assert np.allclose(tiny_chunks.predictions, one_chunk.predictions, atol=1e-4)
    assert abs(tiny_chunks.rmse - one_chunk.rmse) < 1e-5
    assert abs(tiny_chunks.r2 - one_chunk.r2) < 1e-5


def test_linear_decode_transfer_reuses_the_fit_condition_decoder() -> None:
    rng = np.random.default_rng(4)
    fit_features = rng.normal(size=(256, 6)).astype(np.float32)
    weights = rng.normal(size=(6, 2)).astype(np.float32)
    positions = fit_features @ weights
    rotated_features = -fit_features

    fit_result = linear_decode_position(
        fit_features,
        positions,
        train_fraction=0.75,
        include_shuffle=False,
    )
    transfer_result = linear_decode_position_transfer(
        fit_features,
        rotated_features,
        positions,
        train_fraction=0.75,
    )

    assert fit_result.r2 > 0.99
    assert transfer_result.r2 < 0.0


def test_chunked_ridge_fit_predict_matches_sklearn_for_scalar_target() -> None:
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(3)
    features = rng.normal(size=(300, 10)).astype(np.float32)
    weights = rng.normal(size=(10, 1)).astype(np.float32)
    targets = (features @ weights + 0.2 * rng.normal(size=(300, 1))).astype(np.float32)
    alpha = 1.0e-2
    train_indices = np.arange(0, 240, dtype=np.int64)
    validation_indices = np.arange(240, 300, dtype=np.int64)

    reference = Ridge(alpha=alpha, fit_intercept=True)
    reference.fit(features[train_indices], targets[train_indices])
    expected = reference.predict(features[validation_indices])

    predictions = chunked_ridge_fit_predict(
        features, targets, train_indices, validation_indices, alpha
    )
    assert predictions.shape == (60, 1)
    assert expected.shape == (60,)
    assert np.allclose(predictions, expected.reshape(60, 1), atol=1e-3)


def test_chunked_ridge_fit_predict_is_chunk_invariant_for_scalar_target() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(256, 9)).astype(np.float32)
    weights = rng.normal(size=(9, 1)).astype(np.float32)
    targets = (features @ weights + 0.15 * rng.normal(size=(256, 1))).astype(np.float32)
    train_indices = np.arange(0, 200, dtype=np.int64)
    validation_indices = np.arange(200, 256, dtype=np.int64)

    tiny_chunks = chunked_ridge_fit_predict(
        features, targets, train_indices, validation_indices, 1.0e-3, chunk_size=7
    )
    one_chunk = chunked_ridge_fit_predict(
        features, targets, train_indices, validation_indices, 1.0e-3, chunk_size=100_000
    )
    assert np.allclose(tiny_chunks, one_chunk, atol=1e-4)


def test_split_indices_uses_deterministic_shuffled_samples_without_episode_ids() -> None:
    train_indices, validation_indices = _split_indices(10, 0.6)

    assert train_indices.tolist() != [0, 1, 2, 3, 4, 5]
    assert sorted(np.concatenate([train_indices, validation_indices]).tolist()) == list(range(10))


def test_split_indices_holds_out_entire_episodes_when_episode_ids_are_provided() -> None:
    episode_ids = np.repeat(np.arange(5, dtype=np.int64), 4)

    train_indices, validation_indices = _split_indices(20, 0.6, episode_ids=episode_ids)

    train_episodes = set(episode_ids[train_indices].tolist())
    validation_episodes = set(episode_ids[validation_indices].tolist())
    assert train_episodes
    assert validation_episodes
    assert train_episodes.isdisjoint(validation_episodes)


def test_nonlinear_decode_survives_a_unit_that_never_fires_in_train() -> None:
    rng = np.random.default_rng(0)
    episodes, steps, code_dim = 40, 200, 12
    num_samples = episodes * steps
    episode_ids = np.repeat(np.arange(episodes), steps)
    positions = rng.uniform(0.0, 20.0, size=(num_samples, 2)).astype(np.float32)

    codes = np.zeros((num_samples, code_dim), dtype=np.float32)
    codes[:, 0] = positions[:, 0] / 20.0
    codes[:, 1] = positions[:, 1] / 20.0
    codes[:, 2:8] = rng.normal(0.0, 1.0, size=(num_samples, 6)).astype(np.float32)

    train_indices, validation_indices = _split_indices(num_samples, 0.8, episode_ids=episode_ids)
    held_out_episode = int(episode_ids[validation_indices][0])
    codes[episode_ids == held_out_episode, 8] = 1.0
    assert codes[train_indices, 8].std() == 0.0, "unit must be constant in train"
    assert codes[validation_indices, 8].max() > 0.0, "unit must fire in validation"

    result = nonlinear_decode_position(
        codes, positions, episode_ids=episode_ids, max_epochs=30
    )
    assert np.abs(result.predictions).max() < 1e3, "no prediction may explode"
    assert result.r2 > 0.5, f"decode must stay usable, got r2={result.r2}"
