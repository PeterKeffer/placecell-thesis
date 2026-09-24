from __future__ import annotations

from placecell_research.stages.analyze_model import (
    _analysis_config_with_overlay_kwinners_default,
    _analysis_config_with_seed_defaults,
    _analysis_config_with_target_order,
    _enabled_comparative_items,
)


def _targets_with_grid_appended() -> dict:
    return {
        "target_order": ["encoder_place_cells", "grid_cells", "grid_integrator", "grid_target"],
        "targets": {
            "encoder_place_cells": {"source": "encoder.place_codes"},
            "predictor_place_cells": {"source": "predictor.place_codes"},
            "teacher_place_cells": {"source": "teacher.place_codes"},
            "grid_cells": {"source": "grid.hidden_state"},
            "grid_integrator": {"source": "grid.integrator_state"},
            "grid_target": {"source": "grid.target"},
        },
    }


def test_target_order_leads_and_unlisted_targets_follow_in_original_order() -> None:
    resolved = _analysis_config_with_target_order(_targets_with_grid_appended())
    assert list(resolved["targets"]) == [
        "encoder_place_cells",
        "grid_cells",
        "grid_integrator",
        "grid_target",
        "predictor_place_cells",
        "teacher_place_cells",
    ]


def test_target_order_is_noop_when_unset() -> None:
    config = {
        "targets": {
            "encoder_place_cells": {"source": "encoder.place_codes"},
            "grid_cells": {"source": "grid.hidden_state"},
            "predictor_place_cells": {"source": "predictor.place_codes"},
        }
    }
    resolved = _analysis_config_with_target_order(config)
    assert list(resolved["targets"]) == [
        "encoder_place_cells",
        "grid_cells",
        "predictor_place_cells",
    ]


def test_target_order_ignores_unknown_names() -> None:
    config = {
        "target_order": ["grid_cells", "does_not_exist", "encoder_place_cells"],
        "targets": {
            "encoder_place_cells": {"source": "encoder.place_codes"},
            "grid_cells": {"source": "grid.hidden_state"},
            "predictor_place_cells": {"source": "predictor.place_codes"},
        },
    }
    resolved = _analysis_config_with_target_order(config)
    assert list(resolved["targets"]) == [
        "grid_cells",
        "encoder_place_cells",
        "predictor_place_cells",
    ]


def _overlay_analysis_config() -> dict:
    return {
        "targets": {
            "predictor_place_cells": {
                "enabled": True,
                "modules": ["place_field_overlay"],
            }
        }
    }


def test_overlay_kwinners_defaults_to_encoder_fraction_for_hard_k_sparsifier() -> None:
    resolved = _analysis_config_with_overlay_kwinners_default(
        _overlay_analysis_config(),
        encoder_sparsifier_type="kwinners",
        encoder_k_fraction=0.02,
    )
    assert resolved["place_field_overlay_kwinners_k_fraction"] == 0.02


def test_overlay_kwinners_skipped_for_soft_sparsifier() -> None:
    resolved = _analysis_config_with_overlay_kwinners_default(
        _overlay_analysis_config(),
        encoder_sparsifier_type="sparsemax",
        encoder_k_fraction=0.06,
    )
    assert "place_field_overlay_kwinners_k_fraction" not in resolved


def test_overlay_kwinners_respects_explicit_value() -> None:
    config = {**_overlay_analysis_config(), "place_field_overlay_kwinners_k_fraction": 0.0}
    resolved = _analysis_config_with_overlay_kwinners_default(
        config,
        encoder_sparsifier_type="kwinners",
        encoder_k_fraction=0.02,
    )
    assert resolved["place_field_overlay_kwinners_k_fraction"] == 0.0


def test_overlay_kwinners_skipped_when_module_not_enabled() -> None:
    config = {"targets": {"encoder_place_cells": {"enabled": True, "modules": ["rate_map_fields"]}}}
    resolved = _analysis_config_with_overlay_kwinners_default(
        config,
        encoder_sparsifier_type="kwinners",
        encoder_k_fraction=0.02,
    )
    assert "place_field_overlay_kwinners_k_fraction" not in resolved


def test_analysis_seed_defaults_fill_unseeded_analysis_and_comparative_modules() -> None:
    analysis_config = {
        "num_bins_x": 24,
        "num_bins_y": 18,
        "example_episode_random_seed": None,
        "remapping_shuffle_seed": None,
        "sr_oracle_discount_gamma": 0.9,
        "sr_oracle_num_bins_x": 12,
        "sr_oracle_num_bins_y": 10,
        "comparative": {
            "custom_remap": {
                "module": "remapping_comparison",
                "enabled": True,
                "inputs": [{"label": "a"}, {"label": "b"}],
            },
            "combined_episode": {
                "module": "combined_episode_dynamics",
                "enabled": True,
                "example_episode_random_seed": None,
                "inputs": [{"label": "a"}, {"label": "b"}],
            },
            "successor_oracle": {
                "module": "sr_oracle",
                "enabled": True,
                "inputs": [{"label": "successor"}, {"label": "features"}],
            },
        },
    }

    resolved = _analysis_config_with_seed_defaults(analysis_config, seed=13)
    comparative_items = dict(_enabled_comparative_items(resolved))

    assert resolved["example_episode_random_seed"] == 13
    assert resolved["probing_shuffle_seed"] == 13
    assert resolved["remapping_shuffle_seed"] == 13
    assert comparative_items["custom_remap"]["num_bins_x"] == 24
    assert comparative_items["custom_remap"]["num_bins_y"] == 18
    assert comparative_items["custom_remap"]["remapping_shuffle_seed"] == 13
    assert comparative_items["combined_episode"]["example_episode_random_seed"] == 13
    assert comparative_items["successor_oracle"]["sr_oracle_discount_gamma"] == 0.9
    assert comparative_items["successor_oracle"]["sr_oracle_num_bins_x"] == 12
    assert comparative_items["successor_oracle"]["sr_oracle_num_bins_y"] == 10


def test_analysis_seed_defaults_keep_explicit_module_seed() -> None:
    analysis_config = {
        "example_episode_random_seed": None,
        "remapping_shuffle_seed": 3,
        "comparative": {
            "custom_remap": {
                "module": "remapping_comparison",
                "enabled": True,
                "remapping_shuffle_seed": 99,
                "inputs": [{"label": "a"}, {"label": "b"}],
            },
        },
    }

    resolved = _analysis_config_with_seed_defaults(analysis_config, seed=13)
    comparative_items = dict(_enabled_comparative_items(resolved))

    assert resolved["remapping_shuffle_seed"] == 3
    assert comparative_items["custom_remap"]["remapping_shuffle_seed"] == 99
