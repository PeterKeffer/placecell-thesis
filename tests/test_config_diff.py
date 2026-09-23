from __future__ import annotations

from placecell_research.config.diff import compute_config_diff, compute_salient_diff


def _base() -> dict:
    return {
        "environment": {"env_id": "MiniWorld-WallGapAsymLarge-v0"},
        "launcher": {"name": "slurm", "gpu_type": "gpu"},
        "tracking": {"wandb_project": "example"},
        "policies": {
            "artifact_reuse": "reuse_if_config_match",
            "checkpoint_selection": "best",
        },
        "spatial_model": {
            "encoder": {
                "family": "lstm",
                "layer_sizes": [1024, 1024, 1024],
                "head_activation": "none",
                "normalize_codes": False,
            },
            "sparsifier": {"type": "kwinners", "temperature": 0.5},
            "objectives": {
                "vicreg_encoder": {"type": "vicreg", "variance_weight": 5.0},
            },
        },
    }


def _changed() -> dict:
    config = _base()
    encoder = config["spatial_model"]["encoder"]
    encoder["family"] = "gru"
    encoder["layer_sizes"] = [512, 512, 512]
    encoder["head_activation"] = "softplus"
    encoder["normalize_codes"] = True
    config["spatial_model"]["sparsifier"] = {"type": "sparsemax", "temperature": 1.0}
    config["spatial_model"]["objectives"]["vicreg_encoder"]["variance_weight"] = 0.5
    config["launcher"]["gpu_type"] = "a100"
    config["tracking"]["wandb_project"] = "example-studies"
    config["policies"]["artifact_reuse"] = "force_recompute"
    config["policies"]["checkpoint_selection"] = "last"
    return config


def test_config_diff_surfaces_architecture_knobs_that_salient_diff_drops() -> None:
    diff = compute_config_diff(_base(), _changed())
    assert diff["spatial_model.encoder.layer_sizes"] == {
        "base": [1024, 1024, 1024],
        "current": [512, 512, 512],
    }
    assert "spatial_model.encoder.head_activation" in diff
    assert "spatial_model.encoder.normalize_codes" in diff
    assert "spatial_model.encoder.family" in diff
    assert "spatial_model.sparsifier.type" in diff
    assert "spatial_model.objectives.vicreg_encoder.variance_weight" in diff


def test_config_diff_excludes_infrastructure_namespaces() -> None:
    diff = compute_config_diff(_base(), _changed())
    assert not any(key.startswith("launcher.") for key in diff)
    assert not any(key.startswith("tracking.") for key in diff)
    assert "policies.artifact_reuse" not in diff
    assert diff["policies.checkpoint_selection"] == {
        "base": "best",
        "current": "last",
    }


def test_config_diff_shows_new_keys_by_default() -> None:
    changed = _base()
    changed["spatial_model"]["encoder"]["future_unknown_knob"] = 1.5
    diff = compute_config_diff(_base(), changed)
    assert diff["spatial_model.encoder.future_unknown_knob"] == {
        "base": None,
        "current": 1.5,
    }


def test_config_diff_empty_when_nothing_changed() -> None:
    assert compute_config_diff(_base(), _base()) == {}


def test_salient_diff_still_omits_architecture_knobs() -> None:
    diff = compute_salient_diff(_base(), _changed())
    assert "spatial_model.encoder.layer_sizes" not in diff
    assert "spatial_model.encoder.normalize_codes" not in diff
