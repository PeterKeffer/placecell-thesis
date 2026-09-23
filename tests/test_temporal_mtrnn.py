from __future__ import annotations

import pytest
import torch

from placecell_research.spatial_model.components.temporal.mtrnn import MTRNNTemporal


def _build(
    input_dim: int = 4,
    hidden: int = 8,
    time_constants: list[float] | None = None,
) -> MTRNNTemporal:
    torch.manual_seed(0)
    module = MTRNNTemporal(
        input_dim=input_dim,
        layer_sizes=[hidden],
        time_constants=time_constants if time_constants is not None else [1.0, 4.0],
    )
    module.eval()
    return module


def test_mtrnn_sequence_output_shape() -> None:
    module = _build(hidden=8)
    inputs = torch.randn(3, 5, 4)
    outputs, _ = module.forward_sequence(inputs)
    assert outputs.shape == (3, 5, 8)


def test_mtrnn_output_width_equals_last_layer_size() -> None:
    module = _build(hidden=8)
    outputs, _ = module.forward_sequence(torch.randn(2, 4, 4))
    assert outputs.shape[-1] == module.layer_sizes[-1] == 8


def test_mtrnn_initial_state_is_membrane_potential_zeros() -> None:
    module = _build(hidden=8)
    state = module.initial_state(batch_size=3, device=torch.device("cpu"))
    assert isinstance(state, list)
    assert len(state) == 1
    assert state[0].shape == (3, 8)
    assert torch.count_nonzero(state[0]) == 0


def test_mtrnn_step_rollout_matches_full_sequence() -> None:
    module = _build(hidden=8)
    inputs = torch.randn(2, 6, 4)

    seq_outputs, seq_state = module.forward_sequence(inputs)

    step_outputs = []
    state = None
    for timestep in range(inputs.shape[1]):
        out_t, state = module.forward_step(inputs[:, timestep], state)
        step_outputs.append(out_t.unsqueeze(1))
    step_outputs = torch.cat(step_outputs, dim=1)

    torch.testing.assert_close(seq_outputs, step_outputs, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(seq_state[0], state[0], atol=1e-5, rtol=1e-5)


def test_mtrnn_leak_matches_analytic_decay() -> None:
    module = _build(input_dim=3, hidden=4, time_constants=[2.0, 8.0])
    fixed_input = torch.randn(1, 3)

    tau = module.time_constant
    leak = 1.0 / tau

    membrane = torch.zeros(1, 4)
    state = None
    for _ in range(7):
        rate = torch.tanh(membrane)
        drive = module.input_to_membrane(fixed_input) + module.recurrent_to_membrane(rate)
        membrane = (1.0 - leak) * membrane + leak * drive
        expected_rate = torch.tanh(membrane)

        actual_rate, state = module.forward_step(fixed_input, state)
        torch.testing.assert_close(actual_rate, expected_rate, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(state[0], membrane, atol=1e-6, rtol=1e-6)


def test_mtrnn_per_group_tau_decays_at_different_rates() -> None:
    module = _build(input_dim=3, hidden=4, time_constants=[1.0, 64.0])
    fixed_input = torch.randn(1, 3)

    _, state = module.forward_step(fixed_input, None)
    membrane_after_one = state[0]

    fast_group = membrane_after_one[:, :2]
    slow_group = membrane_after_one[:, 2:]

    assert fast_group.abs().mean() > slow_group.abs().mean()


def test_mtrnn_gradient_flows_back_to_first_timestep() -> None:
    module = _build(hidden=8)
    inputs = torch.randn(2, 5, 4, requires_grad=True)
    outputs, _ = module.forward_sequence(inputs)
    outputs.sum().backward()
    assert inputs.grad is not None
    assert inputs.grad[:, 0].abs().sum() > 0


def test_mtrnn_rejects_indivisible_width() -> None:
    with pytest.raises(ValueError):
        MTRNNTemporal(input_dim=4, layer_sizes=[7], time_constants=[1.0, 4.0])


def test_mtrnn_rejects_tau_below_one() -> None:
    with pytest.raises(ValueError):
        MTRNNTemporal(input_dim=4, layer_sizes=[8], time_constants=[0.5, 4.0])


def test_mtrnn_rejects_multilayer() -> None:
    with pytest.raises(ValueError):
        MTRNNTemporal(input_dim=4, layer_sizes=[8, 8], time_constants=[1.0, 4.0])
