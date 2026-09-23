import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location(
    "navigation_memory",
    Path(__file__).parents[1] / "scripts/experiments/evaluate_navigation_memory.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_epsilon_endpoints_and_uniform_actions():
    rng = np.random.default_rng(1)
    assert all(module.epsilon_action(2, 4, 0, rng) == (2, False) for _ in range(100))
    counts = np.bincount([module.epsilon_action(2, 4, 1, rng)[0] for _ in range(10000)])
    assert np.all(np.abs(counts - 2500) < 200)


def test_reset_occurs_before_extract_and_preserves_action_context():
    calls = []
    context = object()
    source = SimpleNamespace(reset=lambda: calls.append("reset"))
    source.extract = lambda context, previous_action: calls.append((context, previous_action))
    module.reset_representation_before_extract(source)
    source.extract(context, 3)
    source.extract(context, 1)
    assert calls == ["reset", (context, 3), "reset", (context, 1)]


def test_failure_loop_does_not_count_success_or_movement():
    summary_spec = importlib.util.spec_from_file_location(
        "navigation_summary",
        Path(__file__).parents[1] / "scripts/experiments/summarize_navigation_memory.py",
    )
    summary = importlib.util.module_from_spec(summary_spec)
    summary_spec.loader.exec_module(summary)
    episode = {
        "success": False,
        "steps": 1024,
        "positions_and_heading": [[0, 0, 0], [0, 0, 1]] * 32,
    }
    assert summary.absorbing_tail(episode, 1024)
    assert not summary.absorbing_tail({**episode, "success": True}, 1024)
    assert not summary.absorbing_tail({**episode, "steps": 63}, 1024)
    moving = {**episode, "positions_and_heading": [[i, 0, 0] for i in range(64)]}
    assert not summary.absorbing_tail(moving, 1024)
