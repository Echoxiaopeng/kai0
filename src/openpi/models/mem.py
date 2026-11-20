"""Short-horizon visual and proprioceptive memory from MEM (arXiv:2603.03596).

JAX/NumPy port of the LeRobot pi05 MEM helpers. The SigLIP space-time path lives in
``siglip.py``; this module holds the shared math and inference history sampling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np


def sample_observation_history(
    history: Sequence[np.ndarray],
    *,
    num_frames: int,
    stride: int,
    steps_seen: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample a homogeneous-batch inference queue and mark pre-episode padding.

    Frames are addressed by age relative to ``history[-1]`` (newest), so the current
    frame is always the last sampled frame regardless of queue length.

    Returns:
        values: stacked history with a new time axis at dim 1, shape ``(B, T, ...)``.
        padding_mask: bool ``(B, T)`` where True means the frame is pre-episode padding
            (invalid). Callers typically invert this to get a validity ``frame_mask``.
    """
    required_ages = list(range((num_frames - 1) * stride, -1, -stride))
    if len(history) <= required_ages[0]:
        raise ValueError(
            f"observation history holds {len(history)} frames, need at least "
            f"{required_ages[0] + 1} to sample {num_frames} frames at stride {stride}"
        )
    values = np.stack([history[-1 - age] for age in required_ages], axis=1)
    valid = np.array([steps_seen > age for age in required_ages], dtype=bool)
    padding_mask = np.broadcast_to(~valid, (values.shape[0], num_frames)).copy()
    return values, padding_mask


def temporal_sinusoidal_embedding(num_frames: int, hidden_size: int, *, dtype=jnp.float32) -> jax.Array:
    """Return fixed temporal embeddings whose current position is exactly zero."""
    if hidden_size % 2:
        raise ValueError(f"hidden_size must be even, got {hidden_size}")
    positions = jnp.arange(1 - num_frames, 1, dtype=jnp.float32)[:, None]
    frequencies = jnp.exp(
        jnp.arange(0, hidden_size, 2, dtype=jnp.float32) * (-math.log(10_000.0) / hidden_size)
    )[None, :]
    angles = positions * frequencies
    sin = jnp.sin(angles)
    cos = jnp.cos(angles) - 1.0
    embedding = jnp.stack([sin, cos], axis=-1).reshape(num_frames, hidden_size)
    return embedding.astype(dtype)


def causal_temporal_mask(frame_mask: jax.Array, *, dtype, num_patches: int) -> jax.Array:
    """Build an additive causal + key-padding mask for temporal attention.

    Returns shape ``(B * num_patches, 1, T, T)`` suitable for multi-head attention
    where the leading dim is ``batch * patches`` (one temporal sequence per patch).
    """
    if frame_mask.ndim != 2:
        raise ValueError(f"frame_mask must have shape (batch, frames), got {tuple(frame_mask.shape)}")
    batch_size, num_frames = frame_mask.shape
    allowed = jnp.tril(jnp.ones((num_frames, num_frames), dtype=bool))
    allowed = allowed[None] & frame_mask[:, None, :].astype(bool)
    # Keep padded query rows numerically safe; valid queries still cannot see padded keys.
    allowed = allowed | jnp.eye(num_frames, dtype=bool)[None]
    neg_inf = jnp.finfo(dtype).min
    mask = jnp.where(allowed[:, None], jnp.asarray(0, dtype=dtype), jnp.asarray(neg_inf, dtype=dtype))
    # (B, 1, T, T) -> (B * P, 1, T, T)
    return jnp.repeat(mask, num_patches, axis=0)


def space_time_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    *,
    temporal_mask: jax.Array,
    num_heads: int,
    deterministic: bool = True,
    dropout_rate: float = 0.0,
) -> jax.Array:
    """MEM composed space-time attention (appendix C, eq. 3) on ``(B, T, P, D)`` Q/K/V.

    Applies temporal attention then spatial attention with the same projected Q/K,
    using temporally-mixed values for the spatial stage. Softmax over a single
    timestep is the identity, so ``T == 1`` reduces to stock spatial attention.
    """
    batch_size, num_frames, num_patches, dim = queries.shape
    head_dim = dim // num_heads
    if dim != num_heads * head_dim:
        raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
    scale = head_dim**-0.5

    def as_temporal(tensor: jax.Array) -> jax.Array:
        # (B,T,P,D) -> (B*P, H, T, d)
        tensor = tensor.reshape(batch_size, num_frames, num_patches, num_heads, head_dim)
        return tensor.transpose(0, 2, 3, 1, 4).reshape(batch_size * num_patches, num_heads, num_frames, head_dim)

    def as_spatial(tensor: jax.Array) -> jax.Array:
        # (B,T,P,D) -> (B*T, H, P, d)
        tensor = tensor.reshape(batch_size, num_frames, num_patches, num_heads, head_dim)
        return tensor.transpose(0, 1, 3, 2, 4).reshape(batch_size * num_frames, num_heads, num_patches, head_dim)

    def _attend(q, k, v, attn_bias=None):
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if attn_bias is not None:
            # attn_bias: (B*, 1, Q, K) -> broadcast over heads
            logits = logits + attn_bias
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(q.dtype)
        if dropout_rate > 0.0 and not deterministic:
            raise NotImplementedError("MEM space-time attention dropout is not used in openpi SigLIP")
        return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

    temporal = _attend(as_temporal(queries), as_temporal(keys), as_temporal(values), attn_bias=temporal_mask)
    temporal = temporal.reshape(batch_size, num_patches, num_heads, num_frames, head_dim)
    temporal = temporal.transpose(0, 3, 1, 2, 4).reshape(batch_size, num_frames, num_patches, dim)

    attended = _attend(as_spatial(queries), as_spatial(keys), as_spatial(temporal), attn_bias=None)
    attended = attended.reshape(batch_size, num_frames, num_heads, num_patches, head_dim)
    return attended.transpose(0, 1, 3, 2, 4).reshape(batch_size, num_frames, num_patches, dim)


def remap_siglip_scan_to_unstacked(params: dict, *, depth: int) -> dict:
    """Remap scanned SigLIP encoderblock params to unstacked ``encoderblock_{i}`` keys.

    OpenPI trains with ``scan=True`` (stacked leading axis). MEM vision uses
    ``scan=False`` so past frames can be dropped mid-network. Call this when loading
    a standard pi05 checkpoint into a MEM-enabled model.
    """
    # Works on a flattened or nested params tree focused on the vision Transformer.
    flat = {}
    from flax import traverse_util

    flat_in = traverse_util.flatten_dict(params, sep="/")
    for key, value in flat_in.items():
        if "/encoderblock/" in key or key.endswith("encoderblock") or "/encoderblock." in key:
            # Expected: .../Transformer/encoderblock/<subpath>
            parts = key.split("/")
            try:
                idx = parts.index("encoderblock")
            except ValueError:
                flat[key] = value
                continue
            # value has leading depth axis
            if value.shape[0] != depth:
                raise ValueError(
                    f"Expected scanned encoderblock leading dim {depth}, got {value.shape} for {key}"
                )
            prefix, suffix = parts[:idx], parts[idx + 1 :]
            for layer_i in range(depth):
                new_key = "/".join([*prefix, f"encoderblock_{layer_i}", *suffix])
                flat[new_key] = value[layer_i]
        else:
            flat[key] = value
    return traverse_util.unflatten_dict(flat, sep="/")
