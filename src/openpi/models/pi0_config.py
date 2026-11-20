import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0
    from openpi.models.pi0_rtc import Pi0RTC


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # MEM short-horizon observation memory (https://arxiv.org/abs/2603.03596).
    # Historical image tokens are fused inside SigLIP and dropped before the
    # language backbone. Historical proprioceptive states become one continuous
    # backbone token per frame. Both paths are opt-in and independent.
    #
    # MEM pre-trains on six observations spaced one second apart. `memory_stride`
    # is counted in dataset frames — scale with fps (e.g. stride=10 at 10 fps).
    use_visual_memory: bool = False
    use_proprioceptive_memory: bool = False
    memory_frames: int = 6
    memory_stride: int = 30
    memory_temporal_attention_every: int = 4

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.use_proprioceptive_memory:
            object.__setattr__(self, "discrete_state_input", False)
        elif self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.memory_frames < 1:
            raise ValueError("memory_frames must be at least 1")
        if self.memory_stride < 1:
            raise ValueError("memory_stride must be at least 1")
        if self.memory_temporal_attention_every < 1:
            raise ValueError("memory_temporal_attention_every must be at least 1")

    @property
    def memory_delta_indices(self) -> list[int] | None:
        """Frame offsets (oldest → current) used for LeRobot delta_timestamps."""
        if not (self.use_visual_memory or self.use_proprioceptive_memory):
            return None
        horizon = (self.memory_frames - 1) * self.memory_stride
        return list(range(-horizon, 1, self.memory_stride))

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "Pi0":
        """Load params, remapping scanned SigLIP weights when MEM visual memory is enabled."""
        if self.use_visual_memory:
            from openpi.training.weight_loaders import maybe_remap_siglip_scan_params

            # Build a reference tree (shapes only) to decide whether remapping is needed.
            ref = nnx.eval_shape(self.create, jax.random.key(0))
            _, ref_state = nnx.split(ref)
            params = maybe_remap_siglip_scan_params(params, ref_state.to_pure_dict())
        return super().load(params, remove_extra_params=remove_extra_params)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        if self.use_visual_memory:
            image_spec = jax.ShapeDtypeStruct(
                [batch_size, self.memory_frames, *_model.IMAGE_RESOLUTION, 3], jnp.float32
            )
            image_mask_spec = jax.ShapeDtypeStruct([batch_size, self.memory_frames], jnp.bool_)
        else:
            image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
            image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        if self.use_proprioceptive_memory:
            state_spec = jax.ShapeDtypeStruct([batch_size, self.memory_frames, self.action_dim], jnp.float32)
            state_mask_spec = jax.ShapeDtypeStruct([batch_size, self.memory_frames], jnp.bool_)
        else:
            state_spec = jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32)
            state_mask_spec = None

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=state_spec,
                state_mask=state_mask_spec,
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class Pi0RTCConfig(Pi0Config):
    """Config for Pi0RTC (real-time control) model. Uses same architecture as Pi0/Pi05 but sample_actions supports
    prev_action_chunk, inference_delay, execute_horizon for RTC guidance. Use this config when serving
    for RTC inference (e.g. agilex_inference_openpi_rtc.py). Set pi05=True for Pi05-based RTC (model_type PI05_RTC)."""

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05_RTC if self.pi05 else _model.ModelType.PI0_RTC

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RTC":
        from openpi.models.pi0_rtc import Pi0RTC

        return Pi0RTC(self, rngs=nnx.Rngs(rng))

    @override
    def load_pytorch(self, train_config, weight_path: str):
        """RTC model is JAX-only; use a JAX checkpoint with serve_policy and Pi0RTCConfig."""
        raise NotImplementedError(
            "Pi0RTC is only supported with JAX checkpoints. Use a checkpoint saved from OpenPi JAX training "
            "(params directory, not model.safetensors) and serve with --policy.config=pi05_rtc_flatten_fold_inference (or your RTC config name)."
        )


@dataclasses.dataclass(frozen=True)
class AdvantageEstimatorConfig(Pi0Config):
    # * Custom
    loss_action_weight: float = 1.0
    loss_value_weight: float = 1.0