from __future__ import annotations

import torch

from placecell_research.spatial_model.components.input_assemblers import (
    PREDICTOR_INPUT_ASSEMBLER_BUILDERS,
    BeliefInputAssembler,
    ConditionalInputAssembler,
    DualInputAssembler,
    EncoderInputAssembler,
    GatedInputAssembler,
    predictor_assembler_uses_belief,
)


def test_input_assemblers_declare_belief_dependency() -> None:
    assert EncoderInputAssembler(8, 4, 2).uses_belief is False
    assert BeliefInputAssembler(8, 4, 2).uses_belief is True
    assert DualInputAssembler(8, 4, 2).uses_belief is True
    assert ConditionalInputAssembler(8, 4, 2).uses_belief is True
    assert GatedInputAssembler(8, 4, 2).uses_belief is True
    assert predictor_assembler_uses_belief(object()) is True


def test_predictor_input_assembler_registry_lists_all_modes() -> None:
    assert PREDICTOR_INPUT_ASSEMBLER_BUILDERS == {
        "encoder": EncoderInputAssembler,
        "belief": BeliefInputAssembler,
        "dual": DualInputAssembler,
        "conditional": ConditionalInputAssembler,
        "gated": GatedInputAssembler,
    }


def test_input_assembler_belief_dependency_declaration_matches_behavior() -> None:
    code_dim, action_dim, kinematics_dim = 8, 4, 2
    encoder_code = torch.randn(3, code_dim)
    belief_a = torch.randn(3, code_dim)
    belief_b = belief_a + 10.0
    action_embedding = torch.randn(3, action_dim)
    kinematics = torch.randn(3, kinematics_dim)
    corruption_info = {
        "noise_level": torch.zeros(3, 1),
        "is_blackout": torch.ones(3, 1),
    }

    for assembler_type in PREDICTOR_INPUT_ASSEMBLER_BUILDERS.values():
        assembler = assembler_type(code_dim, action_dim, kinematics_dim)
        output_a = assembler.assemble(
            encoder_code,
            belief_a,
            action_embedding,
            kinematics,
            None,
            corruption_info,
        )
        output_b = assembler.assemble(
            encoder_code,
            belief_b,
            action_embedding,
            kinematics,
            None,
            corruption_info,
        )
        behavior_uses_belief = not torch.allclose(output_a, output_b)
        assert predictor_assembler_uses_belief(assembler) is behavior_uses_belief
