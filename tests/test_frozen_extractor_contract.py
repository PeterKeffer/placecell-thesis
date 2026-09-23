import pytest

from placecell_research.downstream.frozen_extractor import (
    ModelContract,
    _validate_online_extraction_contract,
)


def test_online_extraction_rejects_state_readout_encoder_contract() -> None:
    contract = ModelContract(
        available_representations=["encoder.place_codes"],
        tensor_shapes={"encoder.place_codes": ["B", "T", 8]},
        predictor_input_channels=[],
        encoder_input_channels=["observation"],
        encoder_family="gru",
        encoder_readout="state",
    )

    with pytest.raises(ValueError, match="encoder_readout='state'"):
        _validate_online_extraction_contract(contract, "encoder.place_codes")
