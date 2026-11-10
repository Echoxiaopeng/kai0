"""Explicit inference contract for the audited platform Pi05 joint14 recipe.

The platform keeps one history row while tokenizing state, normalizes with
min/max when present, and supplies a flat 14-dimensional state to Pi05. These
semantics are intentionally separate from the existing Agilex policy contract.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from openpi_client import base_policy

from openpi import transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy_config
from openpi.training import config as _config

STATE_DIM = 14
CAMERA_MAP = {
    "top_head": "base_0_rgb",
    "hand_left": "left_wrist_0_rgb",
    "hand_right": "right_wrist_0_rgb",
}


@dataclasses.dataclass(frozen=True)
class PlatformNormStats:
    """Keep raw endpoints that the generic local NormStats schema discards."""

    q01: np.ndarray
    q99: np.ndarray
    min: np.ndarray | None = None
    max: np.ndarray | None = None

    def __post_init__(self):
        for name in ("q01", "q99", "min", "max"):
            value = getattr(self, name)
            if value is None:
                if name in ("q01", "q99"):
                    raise ValueError(f"Platform stats require {name}")
                continue
            value = np.asarray(value, dtype=np.float64)
            if value.shape != (STATE_DIM,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Platform stats {name} must contain 14 finite values")
            object.__setattr__(self, name, value)
        if (self.min is None) != (self.max is None):
            raise ValueError("Platform stats min and max must be provided together")
        if np.any(self.q99 < self.q01):
            raise ValueError("Platform stats require q99 >= q01")
        if self.min is not None and np.any(self.max < self.min):
            raise ValueError("Platform stats require max >= min")

    def affine_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """Match remote Normalize._normalize_quantile, including constant dims."""
        input_min, input_max = (self.min, self.max) if self.min is not None else (self.q01, self.q99)
        input_range = input_max - input_min
        ignore_dim = input_range < 1e-7
        input_range_safe = input_range.copy()
        input_range_safe[ignore_dim] = 2.0
        scale = 2.0 / input_range_safe
        offset = -1.0 - scale * input_min
        offset[ignore_dim] = -input_min[ignore_dim]
        return scale, offset

    def normalize(self, values: np.ndarray) -> np.ndarray:
        scale, offset = self.affine_parameters()
        return values * scale + offset

    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        scale, offset = self.affine_parameters()
        return (values - offset) / scale


def load_platform_norm_stats(path: Path | str) -> dict[str, PlatformNormStats]:
    """Read raw JSON without dropping the training runtime's min/max fields."""
    payload = json.loads(Path(path).read_text())["norm_stats"]
    result = {}
    for key in ("state", "actions"):
        if key not in payload:
            raise ValueError(f"Platform stats are missing {key}")
        values = payload[key]
        result[key] = PlatformNormStats(
            q01=values.get("q01"),
            q99=values.get("q99"),
            min=values.get("min"),
            max=values.get("max"),
        )
    return result


def _finite_numeric(values, label: str) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError(f"Platform {label} must contain finite real values")
    return values


@dataclasses.dataclass(frozen=True)
class PlatformInputs(transforms.DataTransformFn):
    state_stats: PlatformNormStats

    def __call__(self, data: dict) -> dict:
        state = _finite_numeric(data["state"], "state")
        if state.shape == (STATE_DIM,):
            state = state.reshape(1, STATE_DIM)
        if state.shape != (1, STATE_DIM):
            raise ValueError(f"Platform state must have shape (14,) or (1, 14), got {state.shape}")

        source_images = data["images"]
        if set(source_images) != set(CAMERA_MAP):
            raise ValueError(f"Platform images must contain exactly {tuple(CAMERA_MAP)}")
        images = {}
        for camera, slot in CAMERA_MAP.items():
            image = np.asarray(source_images[camera])
            # ROS clients send CHW, while the audited training composition uses
            # HWC. Convert only unambiguous wire inputs before training checks.
            if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
                image = np.transpose(image, (1, 2, 0))
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Platform RGB image {camera} must have HWC layout")
            if not 1 <= image.shape[0] <= 4096 or not 1 <= image.shape[1] <= 4096:
                raise ValueError(f"Platform RGB image {camera} has invalid spatial dimensions")
            if image.dtype == np.uint8:
                pass
            elif image.dtype.kind == "f" and np.all(np.isfinite(image)) and np.all((image >= 0) & (image <= 1)):
                image = (image * 255).astype(np.uint8)
            else:
                raise ValueError(f"Platform RGB image {camera} must be uint8 or finite floats in [0, 1]")
            images[slot] = image

        result = {
            "state": self.state_stats.normalize(state),
            "image": images,
            "image_mask": dict.fromkeys(CAMERA_MAP.values(), np.True_),
        }
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, np.ndarray) and prompt.shape == ():
                prompt = prompt.item()
            if not isinstance(prompt, str):
                raise ValueError("Platform prompt must be scalar text")
            result["prompt"] = prompt
        return result


@dataclasses.dataclass(frozen=True)
class FlattenPlatformState(transforms.DataTransformFn):
    """Run after TokenizePrompt so its NumPy row string matches training."""

    def __call__(self, data: dict) -> dict:
        data["state"] = data["state"].flatten()
        return data


@dataclasses.dataclass(frozen=True)
class PlatformOutputs(transforms.DataTransformFn):
    action_stats: PlatformNormStats

    def __call__(self, data: dict) -> dict:
        actions = _finite_numeric(data["actions"], "actions")
        if actions.ndim != 2 or actions.shape[1] < STATE_DIM:
            raise ValueError(f"Platform actions must have shape (T, D) with D >= 14, got {actions.shape}")
        # The remote inverse expects raw width 14. Crop padded model outputs first.
        return {"actions": self.action_stats.unnormalize(actions[:, :STATE_DIM])}


def _validate_model(model_config: _model.BaseModelConfig) -> None:
    if model_config.model_type not in (_model.ModelType.PI05, _model.ModelType.PI05_RTC):
        raise ValueError("The platform profile requires a Pi05 or Pi05RTC model")
    if not getattr(model_config, "discrete_state_input", False):
        raise ValueError("The platform profile requires discrete_state_input=True")
    if (model_config.action_dim, model_config.action_horizon, model_config.max_token_len) != (32, 50, 200):
        raise ValueError("The audited platform profile requires action_dim=32, action_horizon=50, max_token_len=200")


@dataclasses.dataclass(frozen=True)
class PlatformDataConfig(_config.DataConfigFactory):
    stats: dict[str, PlatformNormStats] | None = None

    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        del assets_dirs
        _validate_model(model_config)
        if self.stats is None:
            raise ValueError("The platform profile requires explicit raw normalization statistics")
        return _config.DataConfig(
            data_transforms=transforms.Group(
                inputs=[PlatformInputs(self.stats["state"])],
                outputs=[PlatformOutputs(self.stats["actions"])],
            ),
            model_transforms=transforms.Group(
                inputs=[
                    transforms.ResizeImages(224, 224),
                    transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=True,
                    ),
                    FlattenPlatformState(),
                ],
            ),
        )


class PlatformPolicy(base_policy.BasePolicy):
    """Convert the client's physical RTC history into this model's action space."""

    def __init__(self, policy, action_stats: PlatformNormStats):
        self._policy = policy
        self._action_stats = action_stats

    @property
    def metadata(self) -> dict:
        return {
            **self._policy.metadata,
            "prev_action_chunk_input_space": "physical",
            "rtc_guidance_space": "normalized",
        }

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        previous = obs.get("prev_action_chunk")
        if bool(obs.get("enable_rtc", True)) and previous is not None:
            previous = _finite_numeric(previous, "prev_action_chunk")
            if (
                previous.ndim not in (2, 3)
                or previous.shape[-1] not in (14, 32)
                or previous.shape[-2] == 0
                or (previous.ndim == 3 and previous.shape[0] != 1)
            ):
                raise ValueError("Platform prev_action_chunk must have shape (T, 14/32) or (1, T, 14/32)")
            obs = dict(obs)
            obs["prev_action_chunk"] = self._action_stats.normalize(previous[..., :STATE_DIM])
        return self._policy.infer(obs, noise=noise)

    def reset(self) -> None:
        self._policy.reset()


def create_trained_platform_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: Path | str,
    *,
    repack_transforms: transforms.Group,
    default_prompt: str | None,
    norm_stats_path: Path | str,
):
    """Load weights with the explicitly selected, audited platform preprocessing.

    This covers this recipe's history=1, joint14, absolute-action composition,
    with all dimensions normalized and no robot tag, auxiliary fields or gripper
    range conversion. Physical RTC histories are normalized before guidance.
    """
    _validate_model(train_config.model)
    stats_path = Path(norm_stats_path).expanduser().resolve()
    stats = load_platform_norm_stats(stats_path)
    metadata = dict(train_config.policy_metadata or {})
    metadata.update(
        {
            "preprocessing_profile": "platform",
            "normalization": "minmax_if_available_else_q01_q99",
            "state_token_shape": [1, 14],
            "model_state_dim": 14,
            "training_image_layout": "HWC",
            "wire_image_layouts": ["HWC", "unambiguous_CHW"],
            "norm_stats_path": str(stats_path),
        }
    )
    platform_config = dataclasses.replace(
        train_config,
        data=PlatformDataConfig(stats=stats),
        policy_metadata=metadata,
    )
    policy = policy_config.create_trained_policy(
        platform_config,
        checkpoint_dir,
        repack_transforms=repack_transforms,
        default_prompt=default_prompt,
        # This profile handles its own raw stats in both directions. The generic
        # schema drops min/max and its quantile transform has different semantics.
        norm_stats={},
    )
    return PlatformPolicy(policy, stats["actions"])
