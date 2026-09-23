from __future__ import annotations

import numpy as np
import pytest

from placecell_research.collection.policies import RandomDiscretePolicy


def test_ou_smoothed_policy_prefers_forward_with_forward_heavier_probabilities() -> None:
    policy = RandomDiscretePolicy(
        num_actions=3,
        seed=7,
        mode="ou_smoothed_random",
        action_names=["turn_left", "turn_right", "move_forward"],
        action_probabilities={"turn_left": 0.30, "turn_right": 0.30, "move_forward": 0.40},
    )
    policy.reset()

    samples = [policy.sample() for _ in range(512)]
    move_forward_count = samples.count(2)
    turn_left_count = samples.count(0)
    turn_right_count = samples.count(1)

    assert move_forward_count > turn_left_count
    assert move_forward_count > turn_right_count


def test_ou_smoothed_random_policy_rejects_unknown_action_probability_names() -> None:
    with pytest.raises(ValueError, match="Unknown action probability names"):
        RandomDiscretePolicy(
            num_actions=3,
            seed=7,
            mode="ou_smoothed_random",
            action_names=["turn_left", "turn_right", "move_forward"],
            action_probabilities={"strafe_left": 0.2},
        )


def test_ou_smoothed_random_policy_requires_full_three_action_probability_spec() -> None:
    with pytest.raises(ValueError, match="must cover every action"):
        RandomDiscretePolicy(
            num_actions=3,
            seed=7,
            mode="ou_smoothed_random",
            action_names=["turn_left", "turn_right", "move_forward"],
            action_probabilities={"move_forward": 0.40},
        )


def test_uniform_random_policy_rejects_action_probabilities() -> None:
    with pytest.raises(ValueError, match="OU-smoothed or independent random"):
        RandomDiscretePolicy(
            num_actions=3,
            seed=7,
            mode="uniform_random",
            action_names=["turn_left", "turn_right", "move_forward"],
            action_probabilities={"move_forward": 0.2},
        )


def test_independent_random_policy_preserves_named_action_marginals() -> None:
    policy = RandomDiscretePolicy(
        num_actions=3,
        seed=7,
        mode="independent_random",
        action_names=["turn_left", "turn_right", "move_forward"],
        action_probabilities={"turn_left": 0.30, "turn_right": 0.30, "move_forward": 0.40},
    )

    samples = np.asarray([policy.sample() for _ in range(50_000)])
    observed = np.bincount(samples, minlength=3) / samples.size

    np.testing.assert_allclose(observed, [0.30, 0.30, 0.40], atol=0.01)
