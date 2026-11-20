from collections import deque
from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import mem as _mem
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        # Optional MEM settings (taken from model attrs when omitted).
        use_visual_memory: bool | None = None,
        use_proprioceptive_memory: bool | None = None,
        memory_frames: int | None = None,
        memory_stride: int | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        self._use_visual_memory = (
            bool(getattr(model, "use_visual_memory", False)) if use_visual_memory is None else use_visual_memory
        )
        self._use_proprioceptive_memory = (
            bool(getattr(model, "use_proprioceptive_memory", False))
            if use_proprioceptive_memory is None
            else use_proprioceptive_memory
        )
        self._memory_frames = int(memory_frames or getattr(model, "memory_frames", 6) or 6)
        self._memory_stride = int(memory_stride or getattr(model, "memory_stride", 30) or 30)
        # Prefer config values when present on the model config object.
        cfg = getattr(model, "config", None)
        if cfg is not None:
            self._memory_frames = int(getattr(cfg, "memory_frames", self._memory_frames))
            self._memory_stride = int(getattr(cfg, "memory_stride", self._memory_stride))
            if use_visual_memory is None:
                self._use_visual_memory = bool(getattr(cfg, "use_visual_memory", self._use_visual_memory))
            if use_proprioceptive_memory is None:
                self._use_proprioceptive_memory = bool(
                    getattr(cfg, "use_proprioceptive_memory", self._use_proprioceptive_memory)
                )

        self._memory_queues: dict[str, deque] | None = None
        self._memory_steps_seen = 0
        self._memory_batch_size: int | None = None
        self.reset()

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def reset(self) -> None:
        """Clear MEM observation history. Call at each episode boundary.

        Does not clear client-side RTC prev_action_chunk state.
        """
        self._memory_steps_seen = 0
        self._memory_batch_size = None
        if not (self._use_visual_memory or self._use_proprioceptive_memory):
            self._memory_queues = None
            return
        history_length = (self._memory_frames - 1) * self._memory_stride + 1
        # Queues are created lazily on first infer once we know which image keys exist.
        self._memory_queues = {}
        self._memory_history_length = history_length

    def _ensure_memory_queues(self, obs: dict) -> None:
        assert self._memory_queues is not None
        if self._use_visual_memory:
            images = obs.get("images") or obs.get("image")
            if isinstance(images, dict):
                for key in images:
                    qkey = f"image/{key}"
                    if qkey not in self._memory_queues:
                        self._memory_queues[qkey] = deque(maxlen=self._memory_history_length)
        if self._use_proprioceptive_memory and "state" in obs and "state" not in self._memory_queues:
            self._memory_queues["state"] = deque(maxlen=self._memory_history_length)

    def _stack_inference_memory(self, obs: dict) -> dict:
        """Record one observation and attach MEM histories before transforms."""
        if not (self._use_visual_memory or self._use_proprioceptive_memory):
            return obs
        if self._memory_queues is None:
            self.reset()

        # Skip if the client already provided a temporal batch.
        images = obs.get("images") or obs.get("image")
        if isinstance(images, dict) and any(np.asarray(v).ndim >= 4 for v in images.values()):
            # Unbatched MEM sample is (T,H,W,C) ndim=4; single frame (H,W,C) ndim=3.
            # Batched would be rare at this stage. Treat ndim>=4 on unbatched as already temporal
            # only when first dim looks like memory_frames.
            already = False
            for v in images.values():
                arr = np.asarray(v)
                if arr.ndim == 4 and arr.shape[0] == self._memory_frames:
                    already = True
                    break
            if already:
                return obs
        if self._use_proprioceptive_memory and "state" in obs:
            state = np.asarray(obs["state"])
            if state.ndim == 2 and state.shape[0] == self._memory_frames:
                return obs

        self._ensure_memory_queues(obs)
        result = dict(obs)

        def _append_and_sample(queue_key: str, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            value = np.asarray(value)
            # Work with a leading batch dim of 1 for sample_observation_history.
            batched = value[None, ...]
            batch_size = batched.shape[0]
            if self._memory_batch_size is None:
                self._memory_batch_size = batch_size
            elif batch_size != self._memory_batch_size:
                raise ValueError(
                    "MEM inference batch size changed without policy.reset(); "
                    f"expected {self._memory_batch_size}, got {batch_size}"
                )
            queue = self._memory_queues[queue_key]
            if not queue:
                queue.extend([batched.copy()] * queue.maxlen)
            else:
                queue.append(batched.copy())
            values, padding = _mem.sample_observation_history(
                list(queue),
                num_frames=self._memory_frames,
                stride=self._memory_stride,
                steps_seen=self._memory_steps_seen + 1,
            )
            # Drop the helper batch dim for unbatched policy inputs.
            return values[0], padding[0]

        if self._use_visual_memory:
            image_key = "images" if "images" in result else "image"
            if image_key in result and isinstance(result[image_key], dict):
                stacked_images = {}
                stacked_masks = {}
                for cam, img in result[image_key].items():
                    if cam.endswith("_is_pad"):
                        continue
                    frames, padding = _append_and_sample(f"image/{cam}", img)
                    stacked_images[cam] = frames
                    stacked_masks[cam] = ~padding
                result[image_key] = stacked_images
                # Prefer model-side image_mask naming after transforms; also stash for raw paths.
                result.setdefault("image_mask", {})
                if isinstance(result["image_mask"], dict):
                    result["image_mask"].update(stacked_masks)
                # Also store per-camera is_pad for AgilexInputs-style transforms.
                for cam, padding_inv in stacked_masks.items():
                    result[image_key][f"{cam}_is_pad"] = ~padding_inv

        if self._use_proprioceptive_memory and "state" in result:
            frames, padding = _append_and_sample("state", result["state"])
            result["state"] = frames
            result["state_is_pad"] = padding
            result["state_mask"] = ~padding

        self._memory_steps_seen += 1
        return result

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        # MEM history first (single-frame client obs → temporal), then transforms, then RTC kwargs.
        inputs = self._stack_inference_memory(inputs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)

        # RTC passthrough: client may send delay / prev chunk / horizon for Pi0RTC.
        # Do NOT forward enable_rtc / mask_prefix_delay / prefix_attention_schedule into the
        # jitted sample_actions — those are Python bools/strs and cause TracerBoolConversionError.
        # Gate RTC in Python: omit prev_action_chunk when enable_rtc is false.
        enable_rtc = bool(obs.get("enable_rtc", True))
        if enable_rtc and "prev_action_chunk" in obs and obs["prev_action_chunk"] is not None:
            sample_kwargs["prev_action_chunk"] = np.asarray(
                obs["prev_action_chunk"], dtype=np.float32
            )
        if "inference_delay" in obs:
            sample_kwargs["inference_delay"] = int(obs["inference_delay"])
        if "execute_horizon" in obs:
            sample_kwargs["execute_horizon"] = int(obs["execute_horizon"])
        if "max_guidance_weight" in obs:
            sample_kwargs["max_guidance_weight"] = float(obs["max_guidance_weight"])
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()
