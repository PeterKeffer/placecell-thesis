from __future__ import annotations

import sys
import types

import pytest

from placecell_research.envs.miniworld_adapter import MiniWorldAdapter


class _FakeEnv:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _make_adapter() -> tuple[MiniWorldAdapter, _FakeEnv]:
    adapter = MiniWorldAdapter.__new__(MiniWorldAdapter)
    fake = _FakeEnv()
    adapter._env = fake
    return adapter, fake


def _apply_slurm_egl_headless_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("SLURM_JOB_ID", "9253871")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("MINIWORLD_HEADLESS", "1")
    monkeypatch.delenv("PYGLET_HEADLESS", raising=False)


def test_close_runs_underlying_close_on_slurm_headless_egl(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, fake = _make_adapter()
    _apply_slurm_egl_headless_env(monkeypatch)

    adapter.close()

    assert fake.close_calls == 1
    assert adapter._env is None


def test_close_runs_normally_off_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, fake = _make_adapter()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("MINIWORLD_HEADLESS", "1")

    adapter.close()

    assert fake.close_calls == 1
    assert adapter._env is None


def test_init_closes_partial_env_when_reset_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeActionSpace:
        n = 3

    class FakeEnv:
        def __init__(self) -> None:
            self.unwrapped = self
            self.action_space = FakeActionSpace()
            self.close_calls = 0

        def reset(self, seed: int | None = None):
            del seed
            raise ValueError("reset failed")

        def close(self) -> None:
            self.close_calls += 1

    fake_env = FakeEnv()

    class FakeGymModule:
        @staticmethod
        def make(env_id: str, **kwargs):
            del env_id, kwargs
            return fake_env

    monkeypatch.setattr(
        "placecell_research.envs.miniworld_adapter._ensure_miniworld_env_registered",
        lambda env_id: None,
    )
    monkeypatch.setitem(sys.modules, "gymnasium", FakeGymModule())
    monkeypatch.setitem(sys.modules, "miniworld", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "placecell_research.envs.miniworld_runtime_compat",
        types.SimpleNamespace(ensure_miniworld_runtime_compatibility=lambda: None),
    )

    with pytest.raises(ValueError, match="reset failed"):
        MiniWorldAdapter(
            env_id="MiniWorld-WallGapAsym-v0",
            seed=0,
            episode_length=1,
        )

    assert fake_env.close_calls == 1
