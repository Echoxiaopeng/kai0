"""Shared MEM-aware prefix embedding for Pi0 / Pi0RTC."""

from __future__ import annotations

import einops
import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import array_typing as at


def embed_prefix_with_mem(
    model,
    obs: _model.Observation,
) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
    """Embed images (optional MEM video), optional proprio history, and language tokens."""
    input_mask = []
    ar_mask = []
    tokens = []

    use_visual_memory = bool(getattr(model, "use_visual_memory", False))

    for name in obs.images:
        image = obs.images[name]
        img_mask = obs.image_masks[name]
        if use_visual_memory and image.ndim == 5:
            # (B, T, H, W, C) — frame validity is (B, T)
            frame_mask = img_mask if img_mask.ndim == 2 else einops.repeat(img_mask, "b -> b t", t=image.shape[1])
            image_tokens, _ = model.PaliGemma.img(image, train=False, frame_mask=frame_mask)
            token_pad = frame_mask[:, -1]
        else:
            image_tokens, _ = model.PaliGemma.img(image, train=False)
            token_pad = img_mask[:, -1] if img_mask.ndim == 2 else img_mask

        tokens.append(image_tokens)
        input_mask.append(
            einops.repeat(
                token_pad,
                "b -> b s",
                s=image_tokens.shape[1],
            )
        )
        ar_mask += [False] * image_tokens.shape[1]

    proprio_proj = getattr(model, "proprio_history_proj", None)
    if proprio_proj is not None:
        state = obs.state
        if state.ndim == 2:
            state = state[:, None, :]
        state_tokens = proprio_proj(state)  # (B, T, emb)
        tokens.append(state_tokens)
        if obs.state_mask is not None:
            state_mask = obs.state_mask
        else:
            state_mask = jnp.ones(state.shape[:2], dtype=jnp.bool_)
        input_mask.append(state_mask)
        ar_mask += [False] * state_tokens.shape[1]

    if obs.tokenized_prompt is not None:
        tokenized_inputs = model.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]

    tokens = jnp.concatenate(tokens, axis=1)
    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)
    return tokens, input_mask, ar_mask


def init_mem_modules(model, config, *, paligemma_width: int, rngs: nnx.Rngs) -> None:
    """Attach MEM flags / proprio projection on a Pi0-like model instance."""
    model.config = config
    model.use_visual_memory = bool(config.use_visual_memory)
    model.use_proprioceptive_memory = bool(config.use_proprioceptive_memory)
    model.memory_frames = int(config.memory_frames)
    model.memory_stride = int(config.memory_stride)
    model.memory_temporal_attention_every = config.memory_temporal_attention_every
    if config.use_proprioceptive_memory:
        model.proprio_history_proj = nnx.Linear(config.action_dim, paligemma_width, rngs=rngs)
    else:
        model.proprio_history_proj = None
