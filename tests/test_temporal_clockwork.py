from __future__ import annotations

import pytest
import torch

from placecell_research.config.schema import TemporalFamilyConfig
from placecell_research.spatial_model.components.temporal.clockwork import ClockworkTemporal

INPUT_DIM = 5
LAYER_SIZES = [16]
CLOCK_PERIODS = [1, 2, 4, 8]
BATCH = 3
TIME = 12


def _make_module() -> ClockworkTemporal:
    torch.manual_seed(0)
    module = ClockworkTemporal(
        input_dim=INPUT_DIM,
        layer_sizes=LAYER_SIZES,
        dropout=0.0,
        clock_periods=CLOCK_PERIODS,
    )
    module.eval()
    return module


def _module_slice(module_index: int, module: ClockworkTemporal) -> slice:
    start = module_index * module.module_width
    return slice(start, start + module.module_width)


def test_clockwork_sequence_output_shape() -> None:
    module = _make_module()
    inputs = torch.randn(BATCH, TIME, INPUT_DIM)
    outputs, _ = module.forward_sequence(inputs)
    assert outputs.shape == (BATCH, TIME, LAYER_SIZES[-1])


def test_clockwork_output_width_equals_last_layer_size() -> None:
    module = _make_module()
    inputs = torch.randn(BATCH, TIME, INPUT_DIM)
    outputs, _ = module.forward_sequence(inputs)
    assert outputs.shape[-1] == LAYER_SIZES[-1]


def test_clockwork_initial_state_carries_a_zero_hidden_and_a_zero_phase() -> None:
    module = _make_module()
    state = module.initial_state(BATCH, torch.device("cpu"))
    assert isinstance(state, list)
    assert len(state) == 2
    assert state[0].shape == (BATCH, LAYER_SIZES[-1])
    assert torch.count_nonzero(state[0]) == 0
    assert state[1].shape == (BATCH,)
    assert torch.count_nonzero(state[1]) == 0


def test_clockwork_step_rollout_matches_full_sequence() -> None:
    module = _make_module()
    inputs = torch.randn(BATCH, TIME, INPUT_DIM)
    sequence_outputs, _ = module.forward_sequence(inputs)

    state = None
    step_outputs = []
    for offset in range(TIME):
        output_t, state = module.forward_step(inputs[:, offset], state, t=offset + 1)
        step_outputs.append(output_t.unsqueeze(1))
    step_outputs = torch.cat(step_outputs, dim=1)

    torch.testing.assert_close(sequence_outputs, step_outputs, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("split", [1, 2, 5])
def test_clockwork_two_halves_with_carried_state_equal_one_full_sequence(split: int) -> None:
    """The tick phase must survive the call boundary, or the second half restarts at t=1."""
    module = _make_module()
    inputs = torch.randn(BATCH, TIME, INPUT_DIM)
    full_outputs, full_state = module.forward_sequence(inputs)

    first_outputs, carried = module.forward_sequence(inputs[:, :split])
    second_outputs, split_state = module.forward_sequence(inputs[:, split:], carried)
    chunked = torch.cat((first_outputs, second_outputs), dim=1)

    torch.testing.assert_close(chunked, full_outputs, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(split_state[0], full_state[0], atol=1e-5, rtol=1e-5)
    assert torch.equal(split_state[1], full_state[1])


def test_clockwork_explicit_timestep_overrides_the_carried_phase() -> None:
    """The predictor rollout starts mid-sequence from a cold state and passes its own t."""
    module = _make_module()
    inputs = torch.randn(BATCH, INPUT_DIM)
    state = module.initial_state(BATCH, torch.device("cpu"))
    _output, next_state = module.forward_step(inputs, state, t=8)
    assert torch.equal(next_state[1], torch.full((BATCH,), 8, dtype=next_state[1].dtype))


def test_clockwork_reset_phase_makes_the_next_step_the_first_of_an_episode() -> None:
    """Zeroing the carry is how a stream says "new episode", so t must start over at 1."""
    module = _make_module()
    inputs = torch.randn(BATCH, TIME, INPUT_DIM)
    _outputs, state = module.forward_sequence(inputs)
    cold = module.initial_state(BATCH, torch.device("cpu"))
    state[0].zero_()
    state[1].zero_()
    after_reset, _ = module.forward_sequence(inputs, state)
    from_cold, _ = module.forward_sequence(inputs, cold)
    torch.testing.assert_close(after_reset, from_cold, atol=1e-5, rtol=1e-5)


def test_clockwork_slow_module_holds_on_non_tick_steps() -> None:
    module = _make_module()
    inputs = torch.randn(BATCH, TIME, INPUT_DIM)
    outputs, _ = module.forward_sequence(inputs)

    for module_index, period in enumerate(CLOCK_PERIODS):
        block = _module_slice(module_index, module)
        for offset in range(1, TIME):
            timestep = offset + 1
            if timestep % period != 0:
                previous = outputs[:, offset - 1, block]
                current = outputs[:, offset, block]
                assert torch.equal(previous, current), (
                    f"module {module_index} (period {period}) changed on non-tick step {timestep}"
                )


def test_clockwork_mask_zeros_slow_reads_from_fast() -> None:
    module = _make_module()
    mask = module.recurrent_mask
    slow_index, fast_index = module.group_count - 1, 0
    slow_block = _module_slice(slow_index, module)
    fast_block = _module_slice(fast_index, module)
    assert not mask[slow_block, fast_block].any()
    assert mask[fast_block, slow_block].all()


def test_clockwork_information_flows_slow_to_fast_not_reverse() -> None:
    module = _make_module()
    inputs = torch.randn(BATCH, TIME, INPUT_DIM)

    slow_index = module.group_count - 1
    fast_index = 0
    slow_block = _module_slice(slow_index, module)
    fast_block = _module_slice(fast_index, module)

    base_state = module.initial_state(BATCH, torch.device("cpu"))

    def _cold_state_with(perturbed_block: slice) -> list[torch.Tensor]:
        hidden, phase = base_state[0].clone(), base_state[1].clone()
        hidden[:, perturbed_block] += 1.0
        return [hidden, phase]

    perturbed_slow = _cold_state_with(slow_block)

    out_base, _ = module.forward_sequence(inputs, [leaf.clone() for leaf in base_state])
    out_slow, _ = module.forward_sequence(inputs, perturbed_slow)
    assert not torch.allclose(out_base[:, :, fast_block], out_slow[:, :, fast_block], atol=1e-6)

    perturbed_fast = _cold_state_with(fast_block)
    out_fast, _ = module.forward_sequence(inputs, perturbed_fast)
    torch.testing.assert_close(
        out_base[:, :, slow_block],
        out_fast[:, :, slow_block],
        atol=1e-6,
        rtol=1e-6,
    )


def test_clockwork_gradient_flows_back_to_first_timestep() -> None:
    module = _make_module()
    inputs = torch.randn(BATCH, TIME, INPUT_DIM, requires_grad=True)
    outputs, _ = module.forward_sequence(inputs)
    outputs.sum().backward()
    assert inputs.grad is not None
    assert torch.count_nonzero(inputs.grad[:, 0]) > 0


def test_clockwork_rejects_indivisible_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ClockworkTemporal(
            input_dim=INPUT_DIM,
            layer_sizes=[10],
            clock_periods=[1, 2, 4],
        )


def test_clockwork_rejects_multilayer() -> None:
    with pytest.raises(ValueError, match="single-layer"):
        ClockworkTemporal(
            input_dim=INPUT_DIM,
            layer_sizes=[16, 16],
            clock_periods=CLOCK_PERIODS,
        )


def test_clockwork_rejects_empty_periods() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ClockworkTemporal(
            input_dim=INPUT_DIM,
            layer_sizes=LAYER_SIZES,
            clock_periods=[],
        )


def test_clockwork_schema_rejects_periods_outside_fastest_to_slowest_order() -> None:
    with pytest.raises(ValueError, match="fastest-to-slowest"):
        TemporalFamilyConfig(
            family="clockwork",
            layer_sizes=[16],
            clockwork_periods=[1, 4, 2, 8],
        )
