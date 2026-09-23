from pathlib import Path

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.decode_region import DecodeRegionModule


def _input_from(representation, positions, valid):
    return AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=valid,
        source_name="level_2.place_codes",
        label="level_2",
        split_name="val",
    )


def _region_coded_dataset(num_episodes=8, time=40, num_regions=4):
    rng = np.random.default_rng(0)
    positions = rng.uniform(0.0, 1.0, size=(num_episodes, time, 2)).astype(np.float32)
    region = (positions[..., 0] * (num_regions**0.5)).astype(int)
    one_hot = np.eye(num_regions, dtype=np.float32)[np.clip(region, 0, num_regions - 1)]
    valid = np.ones((num_episodes, time), dtype=bool)
    return one_hot, positions, valid


def test_region_coded_representation_decodes_well(tmp_path: Path):
    representation, positions, valid = _region_coded_dataset()
    result = DecodeRegionModule().run(
        _input_from(representation, positions, valid),
        tmp_path,
        {"decode_region_num_regions_x": 2, "decode_region_num_regions_y": 1},
    )
    assert result.metrics["region_decode_accuracy"] > 0.9
    assert result.metrics["region_decode_accuracy"] > result.metrics["region_decode_chance"]


def test_random_representation_decodes_near_chance(tmp_path: Path):
    _, positions, valid = _region_coded_dataset()
    rng = np.random.default_rng(1)
    representation = rng.standard_normal((8, 40, 4)).astype(np.float32)
    result = DecodeRegionModule().run(
        _input_from(representation, positions, valid),
        tmp_path,
        {"decode_region_num_regions_x": 2, "decode_region_num_regions_y": 1},
    )
    assert result.metrics["region_decode_accuracy"] < 0.75
