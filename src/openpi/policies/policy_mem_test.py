"""Tests for Policy MEM history buffer and RTC config inheritance."""

from types import SimpleNamespace

import numpy as np

from openpi.models import pi0_config
from openpi.policies import policy as _policy


def _make_policy_stub() -> _policy.Policy:
    model = SimpleNamespace(
        use_visual_memory=True,
        use_proprioceptive_memory=True,
        memory_frames=3,
        memory_stride=2,
        config=SimpleNamespace(
            use_visual_memory=True,
            use_proprioceptive_memory=True,
            memory_frames=3,
            memory_stride=2,
        ),
    )
    pol = _policy.Policy.__new__(_policy.Policy)
    pol._model = model
    pol._use_visual_memory = True
    pol._use_proprioceptive_memory = True
    pol._memory_frames = 3
    pol._memory_stride = 2
    pol.reset()
    return pol


def test_policy_stacks_mem_history():
    pol = _make_policy_stub()
    for step in range(5):
        obs = {
            "images": {"base_0_rgb": np.full((8, 8, 3), step, dtype=np.uint8)},
            "state": np.full((8,), float(step), dtype=np.float32),
        }
        stacked = pol._stack_inference_memory(obs)
        assert stacked["images"]["base_0_rgb"].shape == (3, 8, 8, 3)
        assert stacked["state"].shape == (3, 8)
        assert stacked["state_mask"].shape == (3,)
        assert "base_0_rgb_is_pad" in stacked["images"]

    # Newest frame is step 4.
    assert stacked["images"]["base_0_rgb"][-1, 0, 0, 0] == 4
    assert stacked["state"][-1, 0] == 4.0
    # Early history ages: stride=2, frames=3 → ages 4,2,0 relative; steps_seen=5 so all valid.
    assert stacked["state_mask"].tolist() == [True, True, True]

    pol.reset()
    assert pol._memory_steps_seen == 0
    assert pol._memory_queues == {}


def test_policy_skips_stack_when_already_temporal():
    pol = _make_policy_stub()
    obs = {
        "images": {"base_0_rgb": np.zeros((3, 8, 8, 3), dtype=np.uint8)},
        "state": np.zeros((3, 8), dtype=np.float32),
    }
    stacked = pol._stack_inference_memory(obs)
    assert stacked["images"]["base_0_rgb"].shape == (3, 8, 8, 3)
    assert pol._memory_steps_seen == 0


def test_pi0_rtc_config_inherits_mem_flags():
    cfg = pi0_config.Pi0RTCConfig(
        pi05=True,
        use_visual_memory=True,
        use_proprioceptive_memory=True,
        memory_frames=4,
        memory_stride=5,
    )
    assert cfg.model_type.value == "pi05_rtc"
    assert cfg.memory_delta_indices == [-15, -10, -5, 0]
    assert cfg.discrete_state_input is False


def test_platform_policy_reset_passthrough():
    from openpi.policies.platform_policy import PlatformPolicy

    class Inner:
        def __init__(self):
            self.resets = 0
            self.metadata = {}

        def reset(self):
            self.resets += 1

        def infer(self, obs, *, noise=None):
            return {"actions": np.zeros((4, 14))}

    inner = Inner()
    stats = SimpleNamespace(normalize=lambda x: x)
    wrapped = PlatformPolicy(inner, stats)
    wrapped.reset()
    assert inner.resets == 1
