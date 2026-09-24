"""Frozen representation extractor for online RL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from einops import rearrange

from placecell_research.config import materialize_dataclass
from placecell_research.config.schema import VisionConfig
from placecell_research.spatial_model.components.temporal import TEMPORAL_BACKEND_CAPABILITIES
from placecell_research.spatial_model.loading import (
    load_model_from_checkpoint,
    load_place_model_artifact,
)
from placecell_research.spatial_model.predictor_context import (
    kinematics_channels,
    required_kinematics_width,
)
from placecell_research.spatial_model.types import RepresentationBundle
from placecell_research.vision.builder import build_vision_model

_ONLINE_ENCODER_FIELDS = {
    "encoder.backbone_output",
    "encoder.hidden_state",
    "encoder.place_codes",
    "encoder.place_logits",
    "encoder.pre_sparsifier",
}


@dataclass(slots=True)
class ModelContract:
    """Minimal subset of the model contract needed for downstream inference."""

    available_representations: list[str]
    tensor_shapes: dict[str, list[Any]]
    predictor_input_channels: list[str]
    encoder_input_channels: list[str]
    encoder_family: str | None = None
    predictor_family: str | None = None
    encoder_readout: str = "mixed"
    encoder_observation_delay_steps: int = 0

    @property
    def requires_kinematics(self) -> bool:
        return bool(
            kinematics_channels(self.encoder_input_channels)
            or kinematics_channels(self.predictor_input_channels)
        )

    @property
    def required_kinematics_width(self) -> int:
        return max(
            required_kinematics_width(self.encoder_input_channels),
            required_kinematics_width(self.predictor_input_channels),
        )


def _load_contract(
    model_checkpoint: Path, model_contract_path: Path | None = None
) -> ModelContract:
    contract_path = model_contract_path or (
        model_checkpoint / "model_contract.json"
        if model_checkpoint.is_dir()
        else model_checkpoint.with_name("model_contract.json")
    )
    payload = json.loads(contract_path.read_text())
    return ModelContract(
        available_representations=list(payload.get("available_representations", [])),
        tensor_shapes=dict(payload.get("tensor_shapes", {})),
        predictor_input_channels=list(payload.get("predictor_input_channels", [])),
        encoder_input_channels=list(payload.get("encoder_input_channels", ["observation"])),
        encoder_family=payload.get("encoder_family"),
        predictor_family=payload.get("predictor_family"),
        encoder_readout=str(payload.get("encoder_readout", "mixed")),
        encoder_observation_delay_steps=int(payload.get("encoder_observation_delay_steps", 0)),
    )


def _validate_online_extraction_contract(
    contract: ModelContract,
    representation_source: str,
) -> None:
    local_source = representation_source
    if local_source.startswith("predictor."):
        raise ValueError(
            "Online predictor representations require a stateful online extractor. "
            "The current downstream extractor intentionally rejects predictor sources "
            "instead of replaying one-frame sequences."
        )
    if not local_source.startswith("encoder."):
        raise ValueError(
            "Online frozen extraction currently supports encoder.* representations only; "
            f"got {representation_source!r}."
        )
    if local_source not in _ONLINE_ENCODER_FIELDS:
        raise ValueError(
            "Online frozen extraction does not implement the requested encoder field "
            f"{local_source!r}. Supported fields: {sorted(_ONLINE_ENCODER_FIELDS)}."
        )
    if local_source.startswith("encoder."):
        if contract.encoder_readout != "mixed":
            raise ValueError(
                "Online frozen extraction does not support encoder_readout='state'; "
                "the state-readout path must use a canonical stateful encoder runtime."
            )
        encoder_family = contract.encoder_family
        if encoder_family is not None and encoder_family not in TEMPORAL_BACKEND_CAPABILITIES:
            raise ValueError(
                f"Unknown encoder_family={encoder_family!r} in the model contract; online "
                "extraction has no capability descriptor for it."
            )
    return None


def _tensor_batch_dim(tensor: torch.Tensor, batch_size: int) -> int | None:
    if tensor.ndim >= 1 and int(tensor.shape[0]) == int(batch_size):
        return 0
    if tensor.ndim >= 2 and int(tensor.shape[1]) == int(batch_size):
        return 1
    return None


def _is_batch_independent_leaf(state: Any) -> bool:
    """A carry leaf that holds no per-environment axis, so no batch size can disagree with it."""
    return isinstance(state, (bool, int, float))


def _raise_unknown_state_leaf(state: Any, action: str) -> None:
    raise TypeError(
        f"Cannot {action} recurrent encoder state: leaf of type {type(state).__name__!r} is "
        "neither a tensor, a batch-independent scalar, None, nor a list/tuple of those. Teach "
        "these helpers the new leaf type instead of letting the carry be dropped in silence."
    )


def _state_has_batch_size(state: Any, batch_size: int) -> bool:
    """Does every leaf of state agree with batch_size?"""
    if state is None:
        return True
    if isinstance(state, torch.Tensor):
        return _tensor_batch_dim(state, batch_size) is not None
    if _is_batch_independent_leaf(state):
        return True
    if isinstance(state, (list, tuple)):
        return all(_state_has_batch_size(value, batch_size) for value in state)
    _raise_unknown_state_leaf(state, "size-check")
    return False


def _rebuild_tuple(state: tuple, items: list[Any]) -> tuple:
    """Rebuild state's tuple type: NamedTuple carries (_fields) keep their class."""
    if hasattr(state, "_fields"):
        return type(state)(*items)
    return tuple(items)


def _zero_state_indices(state: Any, indices: np.ndarray, *, batch_size: int) -> Any:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        batch_dim = _tensor_batch_dim(state, batch_size)
        if batch_dim is None:
            raise RuntimeError(
                "Cannot reset recurrent encoder state because no tensor axis matches "
                f"batch_size={batch_size}. Got state tensor shape {tuple(state.shape)}."
            )
        state = state.clone()
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=state.device)
        slicer = [slice(None)] * state.ndim
        slicer[batch_dim] = index_tensor
        state[tuple(slicer)] = 0
        return state
    if _is_batch_independent_leaf(state):
        raise RuntimeError(
            "Cannot reset one environment of this encoder state: the carry holds a "
            f"batch-independent {type(state).__name__} leaf ({state!r}), which every row "
            "shares. Zeroing the per-row tensors around it would leave the reset row reading a "
            "counter from the episode it just left. Reset the whole batch, or give the backend "
            "a per-row carry."
        )
    if isinstance(state, list):
        return [_zero_state_indices(value, indices, batch_size=batch_size) for value in state]
    if isinstance(state, tuple):
        return _rebuild_tuple(
            state, [_zero_state_indices(value, indices, batch_size=batch_size) for value in state]
        )
    _raise_unknown_state_leaf(state, "reset")
    return state


def _slice_state_indices(state: Any, indices: np.ndarray, *, batch_size: int) -> Any:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        batch_dim = _tensor_batch_dim(state, batch_size)
        if batch_dim is None:
            raise RuntimeError(
                "Cannot slice recurrent encoder state because no tensor axis matches "
                f"batch_size={batch_size}. Got state tensor shape {tuple(state.shape)}."
            )
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=state.device)
        return torch.index_select(state, batch_dim, index_tensor)
    if _is_batch_independent_leaf(state):
        return state
    if isinstance(state, list):
        return [_slice_state_indices(value, indices, batch_size=batch_size) for value in state]
    if isinstance(state, tuple):
        return _rebuild_tuple(
            state, [_slice_state_indices(value, indices, batch_size=batch_size) for value in state]
        )
    _raise_unknown_state_leaf(state, "slice")
    return state


def _assign_state_indices(
    state: Any,
    updated_state: Any,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> Any:
    if state is None:
        return updated_state
    if isinstance(state, torch.Tensor):
        batch_dim = _tensor_batch_dim(state, batch_size)
        if batch_dim is None:
            raise RuntimeError(
                "Cannot update recurrent encoder state because no tensor axis matches "
                f"batch_size={batch_size}. Got state tensor shape {tuple(state.shape)}."
            )
        state = state.clone()
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=state.device)
        slicer = [slice(None)] * state.ndim
        slicer[batch_dim] = index_tensor
        state[tuple(slicer)] = updated_state
        return state
    if _is_batch_independent_leaf(state):
        raise RuntimeError(
            "Cannot update a subset of environments in this encoder state: the carry holds a "
            f"batch-independent {type(state).__name__} leaf ({state!r}). Advancing it for the "
            "rows in this call would advance it for the rows that did not step. Step the whole "
            "batch, or give the backend a per-row carry."
        )
    if isinstance(state, list):
        return [
            _assign_state_indices(value, updated_value, indices, batch_size=batch_size)
            for value, updated_value in zip(state, updated_state, strict=False)
        ]
    if isinstance(state, tuple):
        return _rebuild_tuple(
            state,
            [
                _assign_state_indices(value, updated_value, indices, batch_size=batch_size)
                for value, updated_value in zip(state, updated_state, strict=False)
            ],
        )
    _raise_unknown_state_leaf(state, "update")
    return updated_state


def _uses_stateful_encoder_source(contract: ModelContract, representation_source: str) -> bool:
    """Does this encoder carry state between steps, so the streaming runtime is required?"""
    if not representation_source.startswith("encoder."):
        return False
    capabilities = TEMPORAL_BACKEND_CAPABILITIES.get(contract.encoder_family or "")
    return capabilities is not None and capabilities.state_layout != "none"


@dataclass(slots=True)
class _OnlineEncoderRuntime:
    model: torch.nn.Module
    representation_source: str
    observation_delay_steps: int = 0
    state: Any = None
    batch_size: int | None = None
    observation_history: torch.Tensor | None = None

    def __post_init__(self) -> None:
        encoder_stack = getattr(self.model, "encoder_stack", None)
        if encoder_stack is None:
            raise RuntimeError("Temporal encoder online extraction requires model.encoder_stack.")
        if not hasattr(encoder_stack, "forward_stateful"):
            raise RuntimeError(
                "Temporal encoder online extraction requires encoder_stack.forward_stateful, "
                "the canonical stateful encoder API that training also runs."
            )
        if getattr(encoder_stack, "readout_mode", "mixed") != "mixed":
            raise NotImplementedError(
                "Online encoder extraction cannot stream an EncoderStack with "
                "encoder_readout='state': the state readout is not carried between calls, so a "
                "stepwise stream would not equal the offline forward pass."
            )

    @property
    def encoder_stack(self) -> torch.nn.Module:
        return self.model.encoder_stack

    def _delay_observations(
        self,
        observations: torch.Tensor,
        indices: np.ndarray | None,
    ) -> torch.Tensor:
        delay_steps = int(self.observation_delay_steps)
        if delay_steps <= 0:
            return observations

        observed_batch_size = int(observations.shape[0])
        full_batch_size = observed_batch_size if indices is None else self.batch_size
        if full_batch_size is None:
            raise RuntimeError(
                "Indexed encoder extraction requires initialized observation history."
            )
        expected_shape = (int(full_batch_size), delay_steps, *observations.shape[2:])
        history_matches = (
            self.observation_history is not None
            and tuple(self.observation_history.shape) == expected_shape
            and self.observation_history.device == observations.device
            and self.observation_history.dtype == observations.dtype
        )
        if not history_matches:
            if indices is not None:
                raise RuntimeError(
                    "Indexed encoder extraction requires initialized observation history."
                )
            self.observation_history = observations.new_zeros(expected_shape)

        assert self.observation_history is not None
        if indices is None:
            selected_history = self.observation_history
        else:
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=observations.device)
            selected_history = torch.index_select(self.observation_history, 0, index_tensor)
        combined = torch.cat((selected_history, observations), dim=1)
        delayed = combined[:, : observations.shape[1]]
        next_history = combined[:, -delay_steps:]
        if indices is None:
            self.observation_history = next_history
        else:
            updated_history = self.observation_history.clone()
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=observations.device)
            updated_history.index_copy_(0, index_tensor, next_history)
            self.observation_history = updated_history
        return delayed

    def reset(self, indices: np.ndarray | None = None) -> None:
        if indices is None:
            self.state = None
            self.batch_size = None
            self.observation_history = None
            return
        if self.batch_size is None:
            self.state = None
            self.observation_history = None
            return
        if self.state is not None:
            self.state = _zero_state_indices(
                self.state,
                np.asarray(indices, dtype=np.int64),
                batch_size=int(self.batch_size),
            )
        if self.observation_history is not None:
            index_tensor = torch.as_tensor(
                indices,
                dtype=torch.long,
                device=self.observation_history.device,
            )
            self.observation_history = self.observation_history.clone()
            self.observation_history.index_fill_(0, index_tensor, 0)

    def extract(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor | None = None,
        kinematics: torch.Tensor | None = None,
        indices: np.ndarray | None = None,
    ) -> RepresentationBundle:
        """One chunk of steps for a subset of environments, through the canonical encoder API."""
        observed_batch_size = int(observations.shape[0])
        if indices is not None:
            indices_array = np.asarray(indices, dtype=np.int64)
            if np.array_equal(indices_array, np.arange(observed_batch_size, dtype=np.int64)):
                indices = None
        observations = self._delay_observations(observations, indices)
        full_batch_size = observed_batch_size if indices is None else self.batch_size
        if full_batch_size is None:
            raise RuntimeError("Indexed recurrent encoder extraction requires initialized state.")
        if not _state_has_batch_size(self.state, int(full_batch_size)):
            self.state = None
            self.batch_size = None
            if indices is not None:
                raise RuntimeError(
                    "Indexed recurrent encoder extraction requires initialized state."
                )
            full_batch_size = observed_batch_size
        initial_state = self.state
        if indices is not None:
            if self.state is None:
                raise RuntimeError(
                    "Indexed recurrent encoder extraction requires initialized state."
                )
            initial_state = _slice_state_indices(
                self.state,
                np.asarray(indices, dtype=np.int64),
                batch_size=int(full_batch_size),
            )
        outputs, next_state = self.encoder_stack.forward_stateful(
            observations.float(),
            actions=actions,
            kinematics=kinematics,
            initial_state=initial_state,
        )
        if indices is None:
            self.state = next_state
        else:
            self.state = _assign_state_indices(
                self.state,
                next_state,
                np.asarray(indices, dtype=np.int64),
                batch_size=int(full_batch_size),
            )
        self.batch_size = int(full_batch_size)
        return RepresentationBundle(modules={"encoder": outputs})


class FrozenRepresentationExtractor:
    """Step-wise frozen inference for RL."""

    def __init__(
        self,
        model_checkpoint: Path,
        vision_encoder: Path | None = None,
        device: str = "cpu",
        representation_source: str = "encoder.place_codes",
        model_contract_path: Path | None = None,
        checkpoint_selection: str | None = None,
    ) -> None:
        self.model_checkpoint = Path(model_checkpoint)
        self.checkpoint_selection = (
            None if checkpoint_selection is None else str(checkpoint_selection)
        )
        self.vision_encoder_path = Path(vision_encoder) if vision_encoder is not None else None
        self.device = torch.device(device)
        self.representation_source = representation_source
        self._vision_lazy_state_dict: dict[str, Any] | None = None
        self._vision_lazy_config: VisionConfig | None = None
        self.contract = _load_contract(self.model_checkpoint, model_contract_path)
        if representation_source not in self.contract.available_representations:
            raise KeyError(
                f"Representation '{representation_source}' is not exported. "
                f"Available: {self.contract.available_representations}"
            )
        _validate_online_extraction_contract(self.contract, representation_source)

        self.model = self._load_model()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.vision_encoder = self._load_vision_encoder()
        self._previous_action: int | None = None
        self._online_encoder: _OnlineEncoderRuntime | None
        if _uses_stateful_encoder_source(self.contract, representation_source):
            self._online_encoder = _OnlineEncoderRuntime(
                model=self.model,
                representation_source=representation_source,
                observation_delay_steps=self.contract.encoder_observation_delay_steps,
            )
        else:
            self._online_encoder = None

    @property
    def uses_stateful_encoder(self) -> bool:
        return self._online_encoder is not None

    @cached_property
    def head_row_norms(self) -> np.ndarray:
        """Per-neuron gain: L2 norm of each encoder code-head output row (read once, data-free)."""
        if self.representation_source != "encoder.place_codes":
            raise ValueError(
                "head_row_norm pre-scaling is only defined for representation_source="
                f"'encoder.place_codes'; got {self.representation_source!r}."
            )
        encoder_stack = getattr(self.model, "encoder_stack", None)
        encoder_head = getattr(encoder_stack, "encoder_head", None)
        linear = getattr(encoder_head, "linear", None)
        weight = getattr(linear, "weight", None)
        if weight is None:
            raise ValueError(
                "head_row_norm pre-scaling requires "
                "model.encoder_stack.encoder_head.linear.weight, "
                "which this place model does not expose."
            )
        norms = np.linalg.norm(weight.detach().to("cpu").numpy().astype(np.float32), axis=1)
        return np.where(norms <= 1e-8, 1.0, norms).astype(np.float32, copy=False)

    def _load_model(self) -> torch.nn.Module:
        if self.model_checkpoint.is_dir():
            if self.checkpoint_selection is None:
                raise ValueError(
                    "models.place_model_checkpoint is not set, and this model is a place-model "
                    "artifact directory, whose checkpoint the loader would otherwise pick on "
                    "its own. State 'best_primary' or 'last' in the config."
                )
            if self.checkpoint_selection not in {"best_primary", "last"}:
                raise ValueError(
                    "models.place_model_checkpoint must be 'best_primary' or 'last', got "
                    f"{self.checkpoint_selection!r}."
                )
            model, _auxiliary_heads, _contract, _payload = load_place_model_artifact(
                self.model_checkpoint,
                self.device,
                selection="best" if self.checkpoint_selection == "best_primary" else "last",
            )
            return model
        model, _auxiliary_heads, _payload = load_model_from_checkpoint(
            self.model_checkpoint, self.device
        )
        return model

    def _load_vision_encoder(self) -> torch.nn.Module | None:
        if self.vision_encoder_path is None:
            return None
        weights_path = (
            self.vision_encoder_path / "weights.pt"
            if self.vision_encoder_path.is_dir()
            else self.vision_encoder_path
        )
        payload = torch.load(weights_path, map_location=self.device, weights_only=False)
        if hasattr(payload, "encode"):
            model = payload
        elif (
            isinstance(payload, dict) and "model" in payload and hasattr(payload["model"], "encode")
        ):
            model = payload["model"]
        elif isinstance(payload, dict) and "model_state_dict" in payload:
            artifact_dir = weights_path.parent
            config_path = artifact_dir / "training_config.yaml"
            if not config_path.exists():
                raise RuntimeError(
                    "Vision encoder checkpoint only has state_dict, but "
                    "training_config.yaml is missing."
                )
            vision_config = materialize_dataclass(
                VisionConfig, yaml.safe_load(config_path.read_text()) or {}
            )
            self._vision_lazy_state_dict = payload["model_state_dict"]
            self._vision_lazy_config = vision_config
            return None
        else:
            raise RuntimeError("Vision encoder checkpoint does not expose an `encode` method.")
        model = model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _ensure_vision_encoder(self, rgb_tensor: torch.Tensor) -> torch.nn.Module | None:
        if self.vision_encoder is not None:
            return self.vision_encoder
        state_dict = getattr(self, "_vision_lazy_state_dict", None)
        config = getattr(self, "_vision_lazy_config", None)
        if state_dict is None or config is None:
            return None
        input_shape = tuple(int(value) for value in rgb_tensor.shape[-3:])
        model = build_vision_model(config, input_shape)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.vision_encoder = model
        self._vision_lazy_state_dict = None
        self._vision_lazy_config = None
        return model

    def reset(self, indices: np.ndarray | None = None) -> None:
        if indices is None:
            self._previous_action = None
        if self._online_encoder is not None:
            self._online_encoder.reset(indices)
        reset_method = getattr(self.model, "reset_state", None)
        if callable(reset_method) and indices is None:
            reset_method()

    def _encode_observation(
        self, rgb: np.ndarray
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        rgb_tensor = torch.as_tensor(rgb, dtype=torch.float32, device=self.device)
        if rgb_tensor.ndim == 3 and rgb_tensor.shape[0] not in {1, 3}:
            rgb_tensor = rearrange(rgb_tensor, "height width channels -> channels height width")
        rgb_tensor = rearrange(rgb_tensor, "channels height width -> 1 1 channels height width")
        vision_encoder = self._ensure_vision_encoder(rgb_tensor.squeeze(0))
        if vision_encoder is None:
            return rgb_tensor / 255.0, None
        with torch.no_grad():
            latent = vision_encoder.encode(
                (rgb_tensor.squeeze(0) / 255.0).reshape(-1, *rgb_tensor.shape[-3:])
            )
        if isinstance(latent, tuple):
            latent = latent[0]
        if latent.ndim == 1:
            latent = rearrange(latent, "features -> 1 1 features")
        else:
            latent = rearrange(latent, "batch features -> 1 batch features")
        return None, latent

    def _encode_observation_batch(
        self,
        rgb_batch: np.ndarray,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        rgb_array = np.asarray(rgb_batch)
        if rgb_array.ndim != 4:
            raise ValueError(
                f"Expected batched RGB frames with 4 dims, got {tuple(rgb_array.shape)}."
            )
        if rgb_array.shape[1] in {1, 3}:
            chw_batch = rgb_array
        else:
            chw_batch = np.transpose(rgb_array, (0, 3, 1, 2))
        rgb_tensor = torch.as_tensor(chw_batch, dtype=torch.float32, device=self.device)
        vision_encoder = self._ensure_vision_encoder(rgb_tensor)
        if vision_encoder is None:
            return rgb_tensor.unsqueeze(1) / 255.0, None
        with torch.inference_mode():
            latent = vision_encoder.encode(rgb_tensor / 255.0)
        if isinstance(latent, tuple):
            latent = latent[0]
        return None, latent.reshape(latent.shape[0], 1, -1)

    def _prepare_kinematics(self, kinematics: np.ndarray | None) -> torch.Tensor | None:
        if self.contract.requires_kinematics and kinematics is None:
            raise ValueError(
                "This model contract requires kinematics, but extract() did not receive them."
            )
        if kinematics is None:
            return None
        tensor = torch.as_tensor(kinematics, dtype=torch.float32, device=self.device)
        if tensor.shape[-1] < self.contract.required_kinematics_width:
            raise ValueError(
                "This model contract requires "
                f"{self.contract.required_kinematics_width} kinematics values, "
                "but extract() received width "
                f"{tensor.shape[-1]}."
            )
        return tensor.reshape(1, 1, -1)

    def _prepare_kinematics_batch(
        self,
        kinematics: np.ndarray | None,
        *,
        batch_size: int,
    ) -> torch.Tensor | None:
        if self.contract.requires_kinematics and kinematics is None:
            raise ValueError(
                "This model contract requires kinematics, but extract_batch() did not receive them."
            )
        if kinematics is None:
            return None
        tensor = torch.as_tensor(kinematics, dtype=torch.float32, device=self.device)
        if tensor.shape[-1] < self.contract.required_kinematics_width:
            raise ValueError(
                "This model contract requires "
                f"{self.contract.required_kinematics_width} kinematics values, "
                "but extract_batch() received width "
                f"{tensor.shape[-1]}."
            )
        return tensor.reshape(batch_size, 1, -1)

    def _extract_from_batch(
        self,
        *,
        rgb_tensor: torch.Tensor | None,
        latent_tensor: torch.Tensor | None,
        actions: torch.Tensor,
        kinematics_tensor: torch.Tensor | None,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._online_encoder is not None:
            observations = latent_tensor if latent_tensor is not None else rgb_tensor
            if observations is None:
                raise RuntimeError(
                    "Encoder extraction requires RGB observations or latent observations."
                )
            with torch.inference_mode():
                bundle = self._online_encoder.extract(
                    observations,
                    actions=actions,
                    kinematics=kinematics_tensor,
                    indices=indices,
                )
        else:
            batch_size = int(actions.shape[0])
            batch = {
                "rgb": rgb_tensor,
                "latent": latent_tensor,
                "actions": actions,
                "kinematics": kinematics_tensor,
                "position_xy": torch.zeros(
                    (batch_size, 1, 2),
                    dtype=torch.float32,
                    device=self.device,
                ),
                "valid_steps": torch.ones((batch_size, 1), dtype=torch.bool, device=self.device),
            }
            with torch.inference_mode():
                bundle = self.model.forward_sequence(
                    {key: value for key, value in batch.items() if value is not None}
                )
        feature_tensor = bundle.get_representation(self.representation_source)
        return feature_tensor[:, -1].detach().cpu().numpy().astype(np.float32, copy=False)

    def extract_batch(
        self,
        *,
        rgb_batch: np.ndarray,
        previous_actions: np.ndarray,
        kinematics_batch: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        batch_size = int(np.asarray(rgb_batch).shape[0])
        rgb_tensor, latent_tensor = self._encode_observation_batch(rgb_batch)
        actions = torch.as_tensor(previous_actions, dtype=torch.long, device=self.device).reshape(
            batch_size,
            1,
        )
        kinematics_tensor = self._prepare_kinematics_batch(
            kinematics_batch,
            batch_size=batch_size,
        )
        return self._extract_from_batch(
            rgb_tensor=rgb_tensor,
            latent_tensor=latent_tensor,
            actions=actions,
            kinematics_tensor=kinematics_tensor,
            indices=indices,
        )

    def extract(
        self,
        rgb: np.ndarray,
        previous_action: int | None = None,
        kinematics: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract one feature vector from an observation."""
        rgb_tensor, latent_tensor = self._encode_observation(rgb)
        kinematics_tensor = self._prepare_kinematics(kinematics)
        action_value = self._previous_action if previous_action is None else previous_action
        if action_value is None:
            action_value = 0
        features = self._extract_from_batch(
            rgb_tensor=rgb_tensor,
            latent_tensor=latent_tensor,
            actions=torch.tensor([[int(action_value)]], dtype=torch.long, device=self.device),
            kinematics_tensor=kinematics_tensor,
        )
        self._previous_action = int(action_value)
        return features[0].astype(np.float32, copy=False)

    @property
    def feature_dim(self) -> int:
        shape = self.contract.tensor_shapes.get(self.representation_source)
        if not shape:
            raise KeyError(f"Missing shape for '{self.representation_source}' in model contract.")
        return int(shape[-1])
