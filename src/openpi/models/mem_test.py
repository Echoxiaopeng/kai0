"""Unit tests for MEM helpers and pi05 MEM wiring."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import mem as _mem
from openpi.models import pi0_config
from openpi.models import siglip as _siglip


def test_memory_delta_indices():
    baseline = pi0_config.Pi0Config(pi05=True)
    memory = pi0_config.Pi0Config(
        pi05=True,
        use_visual_memory=True,
        use_proprioceptive_memory=True,
        memory_frames=6,
        memory_stride=10,
    )
    assert baseline.memory_delta_indices is None
    assert memory.memory_delta_indices == [-50, -40, -30, -20, -10, 0]
    assert memory.discrete_state_input is False


@pytest.mark.parametrize("field", ["memory_frames", "memory_stride", "memory_temporal_attention_every"])
def test_mem_config_rejects_non_positive(field):
    with pytest.raises(ValueError, match=field):
        pi0_config.Pi0Config(pi05=True, **{field: 0})


def test_inference_history_order_and_padding():
    history = [np.full((2, 1), value, dtype=np.float32) for value in range(11)]
    values, padding = _mem.sample_observation_history(history, num_frames=3, stride=5, steps_seen=1)
    assert values[:, :, 0].tolist() == [[0, 5, 10], [0, 5, 10]]
    assert padding.tolist() == [[True, True, False], [True, True, False]]


def test_inference_history_anchors_on_newest_frame():
    history = [np.full((1, 1), value, dtype=np.float32) for value in range(14)]
    values, _ = _mem.sample_observation_history(history, num_frames=3, stride=5, steps_seen=99)
    assert values[:, :, 0].tolist() == [[3, 8, 13]]


def test_inference_history_rejects_too_short_queue():
    history = [np.zeros((1, 1), dtype=np.float32) for _ in range(10)]
    with pytest.raises(ValueError, match="need at least 11"):
        _mem.sample_observation_history(history, num_frames=3, stride=5, steps_seen=99)


def test_current_temporal_position_is_exactly_zero():
    embedding = _mem.temporal_sinusoidal_embedding(4, 16)
    np.testing.assert_allclose(np.asarray(embedding[-1]), np.zeros(16), atol=1e-6)
    assert np.count_nonzero(np.asarray(embedding[:-1])) > 0


def test_remap_siglip_scan_to_unstacked():
    depth = 3
    scanned = {
        "Transformer": {
            "encoderblock": {
                "MultiHeadDotProductAttention_0": {
                    "query": {"kernel": np.zeros((depth, 4, 2, 2), dtype=np.float32)},
                }
            }
        }
    }
    unstacked = _mem.remap_siglip_scan_to_unstacked(scanned, depth=depth)
    for i in range(depth):
        assert unstacked["Transformer"][f"encoderblock_{i}"]["MultiHeadDotProductAttention_0"]["query"][
            "kernel"
        ].shape == (4, 2, 2)


def test_tiny_siglip_mem_single_frame_matches_spatial():
    """T==1 MEM path should produce finite tokens with the single-frame token count."""
    module = _siglip.Module(
        num_classes=32,
        variant="Ti/16",
        pool_type="none",
        scan=False,
        temporal_attention_every=2,
        dtype_mm="float32",
    )
    rng = jax.random.key(0)
    image = jax.random.normal(rng, (2, 1, 32, 32, 3))
    frame_mask = jnp.ones((2, 1), dtype=bool)
    variables = module.init(rng, image, train=False, frame_mask=frame_mask)
    tokens, _ = module.apply(variables, image, train=False, frame_mask=frame_mask)
    assert tokens.shape[0] == 2
    assert tokens.shape[-1] == 32
    assert jnp.isfinite(tokens).all()


def test_tiny_siglip_mem_drops_to_single_frame_tokens():
    module = _siglip.Module(
        num_classes=32,
        variant="Ti/16",
        pool_type="none",
        scan=False,
        temporal_attention_every=2,
        dtype_mm="float32",
    )
    rng = jax.random.key(0)
    video = jax.random.normal(rng, (2, 3, 32, 32, 3))
    frame_mask = jnp.ones((2, 3), dtype=bool)
    variables = module.init(rng, video, train=False, frame_mask=frame_mask)
    tokens, _ = module.apply(variables, video, train=False, frame_mask=frame_mask)
    # Same token budget as a single frame (pool_type=none → patch tokens).
    single = jax.random.normal(rng, (2, 32, 32, 3))
    single_tokens, _ = module.apply(variables, single, train=False)
    assert tokens.shape == single_tokens.shape
    assert jnp.isfinite(tokens).all()


@pytest.mark.manual
def test_pi05_mem_embed_prefix_smoke():
    """Full pi05+MEM create + embed_prefix (expensive; marked manual)."""
    from openpi.shared import nnx_utils

    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        pi05=True,
        use_visual_memory=True,
        use_proprioceptive_memory=True,
        memory_frames=2,
        memory_stride=1,
        memory_temporal_attention_every=4,
        action_horizon=10,
        max_token_len=48,
    )
    model = config.create(key)
    obs = config.fake_obs(batch_size=1)
    tokens, pad, ar = model.embed_prefix(obs)
    assert tokens.ndim == 3
    assert pad.shape[:2] == tokens.shape[:2]
    assert ar.ndim == 1
