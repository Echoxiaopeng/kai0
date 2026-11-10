import dataclasses
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from openpi import transforms
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import platform_policy
from openpi.training import config as training_config


def make_stats():
    return platform_policy.PlatformNormStats(
        q01=np.full(14, 1.0),
        q99=np.full(14, 7.0),
        min=np.zeros(14),
        max=np.full(14, 8.0),
    )


def make_observation():
    return {
        "state": np.full(14, 4.0, dtype=np.float32),
        "images": {name: np.zeros((224, 224, 3), dtype=np.uint8) for name in platform_policy.CAMERA_MAP},
        "prompt": "test",
    }


def test_raw_minmax_endpoints_take_precedence_over_quantiles():
    stats = make_stats()
    values = np.repeat(np.asarray([[0.0], [4.0], [8.0]]), 14, axis=1)

    normalized = stats.normalize(values)

    np.testing.assert_array_equal(normalized, np.repeat([[-1.0], [0.0], [1.0]], 14, axis=1))
    np.testing.assert_array_equal(stats.unnormalize(normalized), values)
    fallback = dataclasses.replace(stats, min=None, max=None)
    np.testing.assert_allclose(fallback.unnormalize(normalized)[[0, 2]], np.repeat([[1.0], [7.0]], 14, axis=1))


def test_constant_and_tiny_range_preserve_deviations_and_remain_invertible():
    minimum = np.full(14, 0.25)
    maximum = minimum.copy()
    maximum[1] += 5e-8
    stats = platform_policy.PlatformNormStats(q01=minimum, q99=maximum, min=minimum, max=maximum)
    values = np.asarray([minimum, minimum + 0.125])

    normalized = stats.normalize(values)

    np.testing.assert_array_equal(normalized[0], np.zeros(14))
    np.testing.assert_array_equal(normalized[1], np.full(14, 0.125))
    np.testing.assert_array_equal(stats.unnormalize(normalized), values)


@pytest.mark.parametrize("change", [{"q01": np.zeros(32)}, {"q01": np.full(14, np.nan)}, {"max": None}])
def test_rejects_incompatible_raw_stats(change):
    with pytest.raises(ValueError, match="Platform stats"):
        dataclasses.replace(make_stats(), **change)


def test_factory_preserves_history_row_in_prompt_then_flattens_state(monkeypatch):
    class RecordingSentencePiece:
        text = None

        def encode(self, text, *, add_bos):
            assert add_bos
            self.text = text
            return [2, 99]

    sentencepiece = RecordingSentencePiece()

    class RecordingTokenizer(_tokenizer.PaligemmaTokenizer):
        def __init__(self, max_len):
            self._tokenizer = sentencepiece
            self._max_len = max_len

    monkeypatch.setattr(_tokenizer, "PaligemmaTokenizer", RecordingTokenizer)
    stats = make_stats()
    data_config = platform_policy.PlatformDataConfig(stats={"state": stats, "actions": stats}).create(
        Path("/tmp"), pi0_config.Pi0RTCConfig(pi05=True)
    )
    observed = make_observation()
    normalized = transforms.compose(data_config.data_transforms.inputs)(observed)
    assert normalized["state"].shape == (1, 14)
    # A physical value above pi must remain 4.0 before normalization, rather than
    # triggering the legacy Agilex outlier-to-zero transform.
    np.testing.assert_array_equal(normalized["state"], np.zeros((1, 14)))

    result = transforms.compose(data_config.model_transforms.inputs)(normalized)

    expected_numbers = " ".join(["128"] * 14)
    assert sentencepiece.text == f"Task: test, State: [{expected_numbers}];\nAction: "
    assert result["state"].shape == (14,)
    np.testing.assert_array_equal(result["tokenized_prompt"][:2], [2, 99])
    assert result["tokenized_prompt_mask"].shape == (200,)
    assert result["tokenized_prompt_mask"].sum() == 2
    assert "prompt" not in result
    assert tuple(result["image"]) == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    assert all(result["image_mask"].values())


def test_output_crops_padding_before_using_platform_inverse():
    normalized = np.repeat(np.asarray([[-1.0], [0.0], [1.0]]), 32, axis=1)
    normalized[:, 14:] = 100.0

    result = platform_policy.PlatformOutputs(make_stats())({"actions": normalized, "state": np.zeros(14)})

    assert set(result) == {"actions"}
    np.testing.assert_array_equal(result["actions"], np.repeat([[0.0], [4.0], [8.0]], 14, axis=1))


def test_hwc_float_images_match_platform_uint8_conversion():
    observed = make_observation()
    observed["images"]["hand_left"] = np.full((224, 224, 3), 0.5, dtype=np.float32)

    result = platform_policy.PlatformInputs(make_stats())(observed)

    assert result["image"]["left_wrist_0_rgb"].dtype == np.uint8
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], np.full((224, 224, 3), 127, dtype=np.uint8))


@pytest.mark.parametrize("shape", [(32,), (2, 14), (14, 1)])
def test_rejects_incompatible_state_history_or_width(shape):
    observed = make_observation()
    observed["state"] = np.zeros(shape)
    with pytest.raises(ValueError, match="Platform state must have shape"):
        platform_policy.PlatformInputs(make_stats())(observed)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_chw_wire_boundary_matches_hwc_training_pixels(dtype):
    hwc = np.arange(4 * 5 * 3).reshape(4, 5, 3).astype(dtype)
    if dtype == np.float32:
        hwc /= 255.0
    chw = np.transpose(hwc, (2, 0, 1)).copy()
    observed_hwc = make_observation()
    observed_chw = make_observation()
    observed_hwc["images"]["top_head"] = hwc
    observed_chw["images"]["top_head"] = chw

    from_hwc = platform_policy.PlatformInputs(make_stats())(observed_hwc)
    from_chw = platform_policy.PlatformInputs(make_stats())(observed_chw)

    expected = hwc if dtype == np.uint8 else (hwc * 255).astype(np.uint8)
    np.testing.assert_array_equal(from_chw["image"]["base_0_rgb"], expected)
    np.testing.assert_array_equal(from_chw["image"]["base_0_rgb"], from_hwc["image"]["base_0_rgb"])
    np.testing.assert_array_equal(chw, np.transpose(hwc, (2, 0, 1)))


def test_hwc_with_height_three_keeps_declared_channel_axis():
    observed = make_observation()
    hwc = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    observed["images"]["top_head"] = hwc

    result = platform_policy.PlatformInputs(make_stats())(observed)

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], hwc)


@pytest.mark.parametrize(
    "image",
    [np.zeros((224, 224)), np.zeros((224, 224, 1)), np.full((224, 224, 3), 2.0), np.full((224, 224, 3), np.nan)],
)
def test_rejects_undeclared_image_layout_or_invalid_range(image):
    observed = make_observation()
    observed["images"]["top_head"] = image
    with pytest.raises(ValueError, match="Platform RGB image"):
        platform_policy.PlatformInputs(make_stats())(observed)


@pytest.mark.parametrize("field", ["state", "actions"])
def test_rejects_nonfinite_robot_values(field):
    if field == "state":
        observed = make_observation()
        observed["state"][0] = np.nan
        transform = platform_policy.PlatformInputs(make_stats())
    else:
        observed = {"actions": np.full((50, 32), np.inf)}
        transform = platform_policy.PlatformOutputs(make_stats())
    with pytest.raises(ValueError, match="finite real values"):
        transform(observed)


def test_explicit_policy_builder_preserves_legacy_config_and_disables_generic_normalization(tmp_path, monkeypatch):
    payload = {
        "q01": [1.0] * 14,
        "q99": [7.0] * 14,
        "min": [0.0] * 14,
        "max": [8.0] * 14,
    }
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"norm_stats": {"state": payload, "actions": payload}}))
    config = training_config.get_config("pi05_rtc_flatten_fold_inference")
    original_data = config.data
    captured = {}

    class DummyPolicy:
        metadata: ClassVar[dict] = {}

    expected = DummyPolicy()

    def create_policy(profile_config, checkpoint_dir, **kwargs):
        captured.update(config=profile_config, checkpoint=checkpoint_dir, **kwargs)
        expected.metadata = profile_config.policy_metadata
        return expected

    monkeypatch.setattr(platform_policy.policy_config, "create_trained_policy", create_policy)

    actual = platform_policy.create_trained_platform_policy(
        config,
        tmp_path / "checkpoint",
        repack_transforms=transforms.Group(),
        default_prompt="test",
        norm_stats_path=stats_path,
    )

    assert isinstance(actual, platform_policy.PlatformPolicy)
    assert actual.metadata["prev_action_chunk_input_space"] == "physical"
    assert actual.metadata["rtc_guidance_space"] == "normalized"
    assert captured["norm_stats"] == {}
    assert isinstance(captured["config"].data, platform_policy.PlatformDataConfig)
    np.testing.assert_array_equal(captured["config"].data.stats["state"].min, np.zeros(14))
    assert captured["config"].policy_metadata["model_state_dim"] == 14
    assert captured["config"].policy_metadata["state_token_shape"] == [1, 14]
    assert config.data is original_data


@pytest.mark.parametrize("width", [14, 32])
@pytest.mark.parametrize("batch_axis", [False, True])
def test_rtc_history_normalizes_physical_values_and_preserves_input(width, batch_axis):
    class DummyPolicy:
        metadata: ClassVar[dict] = {"checkpoint": "test"}

        def infer(self, obs, *, noise=None):
            self.observed = obs
            self.noise = noise
            return {"actions": np.zeros((50, 14))}

    inner = DummyPolicy()
    policy = platform_policy.PlatformPolicy(inner, make_stats())
    physical = np.repeat(np.asarray([[0.0], [4.0], [8.0]]), width, axis=1)
    if batch_axis:
        physical = physical[None]
    obs = {"prev_action_chunk": physical, "inference_delay": 2, "execute_horizon": 8}
    noise = np.zeros((50, 32))

    policy.infer(obs, noise=noise)

    expected = np.repeat([[-1.0], [0.0], [1.0]], 14, axis=1)
    np.testing.assert_array_equal(inner.observed["prev_action_chunk"], expected[None] if batch_axis else expected)
    assert inner.observed is not obs
    assert obs["prev_action_chunk"] is physical
    np.testing.assert_array_equal(physical.reshape(3, width)[:, 0], [0.0, 4.0, 8.0])
    assert inner.noise is noise


def test_rtc_disabled_does_not_read_or_normalize_history():
    class DummyPolicy:
        metadata: ClassVar[dict] = {}

        def infer(self, obs, *, noise=None):
            self.observed = obs
            return {}

    inner = DummyPolicy()
    policy = platform_policy.PlatformPolicy(inner, make_stats())
    obs = {"enable_rtc": False, "prev_action_chunk": "unused invalid history"}

    policy.infer(obs)

    assert inner.observed is obs
    assert obs["prev_action_chunk"] == "unused invalid history"
    policy.infer({})
    assert inner.observed == {}


def test_rtc_rejects_wrong_history_width_before_inference():
    class DummyPolicy:
        def infer(self, obs, *, noise=None):
            raise AssertionError("Invalid RTC history must be rejected before sampling")

    policy = platform_policy.PlatformPolicy(DummyPolicy(), make_stats())
    with pytest.raises(ValueError, match="prev_action_chunk must have shape"):
        policy.infer({"prev_action_chunk": np.zeros((50, 15))})


def test_platform_profile_rejects_pi0_model_before_loading_weights(tmp_path):
    config = dataclasses.replace(
        training_config.get_config("pi05_rtc_flatten_fold_inference"), model=pi0_config.Pi0Config()
    )
    with pytest.raises(ValueError, match="Pi05"):
        platform_policy.create_trained_platform_policy(
            config,
            tmp_path,
            repack_transforms=transforms.Group(),
            default_prompt="test",
            norm_stats_path=tmp_path / "missing",
        )
