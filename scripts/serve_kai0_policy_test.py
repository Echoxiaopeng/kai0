import json
from pathlib import Path
import re
import sys
from unittest.mock import Mock

import pytest

from scripts import serve_kai0_policy


def test_norm_stats_prefers_checkpoint_assets(tmp_path: Path):
    checkpoint = tmp_path / "10000"
    bundled = checkpoint / "assets" / "norm_stats.json"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("{}")
    (tmp_path / "norm_stats.json").write_text("{}")

    assert serve_kai0_policy.resolve_norm_stats_path(checkpoint) == bundled


def test_norm_stats_falls_back_to_checkpoint_parent(tmp_path: Path):
    checkpoint = tmp_path / "10000"
    legacy = tmp_path / "norm_stats.json"
    legacy.write_text("{}")

    assert serve_kai0_policy.resolve_norm_stats_path(checkpoint) == legacy


def test_norm_stats_explicit_path_takes_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint = tmp_path / "10000"
    bundled = checkpoint / "assets" / "norm_stats.json"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("{}")
    explicit = tmp_path / "custom" / "norm_stats.json"
    explicit.parent.mkdir()
    explicit.write_text("{}")
    monkeypatch.chdir(tmp_path)

    assert serve_kai0_policy.resolve_norm_stats_path(checkpoint, Path("custom/norm_stats.json")) == explicit


def test_norm_stats_missing_explicit_path_does_not_fall_back(tmp_path: Path):
    checkpoint = tmp_path / "10000"
    (tmp_path / "norm_stats.json").write_text("{}")
    missing = tmp_path / "missing" / "norm_stats.json"

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        serve_kai0_policy.resolve_norm_stats_path(checkpoint, missing)


def test_norm_stats_missing_defaults_reports_both_paths(tmp_path: Path):
    checkpoint = tmp_path / "10000"

    with pytest.raises(FileNotFoundError) as exc_info:
        serve_kai0_policy.resolve_norm_stats_path(checkpoint)

    assert str(checkpoint / "assets" / "norm_stats.json") in str(exc_info.value)
    assert str(tmp_path / "norm_stats.json") in str(exc_info.value)


def test_main_loads_custom_norm_stats_file_with_rtc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint = tmp_path / "10000"
    (checkpoint / "params").mkdir(parents=True)
    stats_path = tmp_path / "custom_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    key: {"mean": [1.0] * 14, "std": [2.0] * 14, "q01": [-3.0] * 14, "q99": [3.0] * 14}
                    for key in ("state", "actions")
                }
            }
        )
    )
    (tmp_path / "norm_stats.json").write_text("{}")
    policy = Mock(metadata={})
    create_policy = Mock(return_value=policy)
    create_platform_policy = Mock()
    server = Mock()
    create_server = Mock(return_value=server)
    monkeypatch.setattr(serve_kai0_policy.policy_config, "create_trained_policy", create_policy)
    monkeypatch.setattr(serve_kai0_policy.platform_policy, "create_trained_platform_policy", create_platform_policy)
    monkeypatch.setattr(serve_kai0_policy.websocket_policy_server, "WebsocketPolicyServer", create_server)
    monkeypatch.setattr(
        sys,
        "argv",
        ["serve_kai0_policy.py", "--checkpoint", str(checkpoint), "--norm-stats", str(stats_path), "--rtc"],
    )

    serve_kai0_policy.main()

    create_policy.assert_called_once()
    create_platform_policy.assert_not_called()
    config, loaded_checkpoint = create_policy.call_args.args
    assert config.name == serve_kai0_policy.RTC_CONFIG
    assert config.model.model_type.value == "pi05_rtc"
    assert loaded_checkpoint == checkpoint
    loaded_stats = create_policy.call_args.kwargs["norm_stats"]
    assert set(loaded_stats) == {"state", "actions"}
    assert loaded_stats["state"].mean.tolist() == [1.0] * 14
    assert loaded_stats["actions"].q99.tolist() == [3.0] * 14
    create_server.assert_called_once()
    assert create_server.call_args.kwargs["policy"] is policy
    assert create_server.call_args.kwargs["metadata"]["supports_rtc"] is True
    assert create_server.call_args.kwargs["metadata"]["rtc_config"] == serve_kai0_policy.RTC_CONFIG
    assert create_server.call_args.kwargs["metadata"]["preprocessing"] == "legacy"
    assert create_server.call_args.kwargs["metadata"]["norm_stats_path"] == str(stats_path)
    server.serve_forever.assert_called_once_with()


@pytest.mark.parametrize("rtc", [False, True])
@pytest.mark.parametrize("stats_location", ["explicit", "assets", "parent"])
def test_main_routes_platform_preprocessing_with_exact_norm_stats_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, rtc: bool, stats_location: str
):
    checkpoint = tmp_path / "10000"
    (checkpoint / "params").mkdir(parents=True)
    paths = {
        "explicit": tmp_path / "selected" / "custom_stats.json",
        "assets": checkpoint / "assets" / "norm_stats.json",
        "parent": tmp_path / "norm_stats.json",
    }
    stats_path = paths[stats_location]
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({"norm_stats": {"state": {"min": [0.0] * 14, "max": [1.0] * 14}}}))
    if stats_location == "explicit":
        # The selected file must win even when the checkpoint bundles statistics.
        paths["assets"].parent.mkdir(parents=True)
        paths["assets"].write_text("{}")
    policy = Mock(metadata={"platform_detail": "preserved"})
    create_policy = Mock()
    create_platform_policy = Mock(return_value=policy)
    deserialize_stats = Mock(side_effect=AssertionError("Platform helper must read the selected stats file itself"))
    server = Mock()
    create_server = Mock(return_value=server)
    monkeypatch.setattr(serve_kai0_policy.policy_config, "create_trained_policy", create_policy)
    monkeypatch.setattr(serve_kai0_policy.platform_policy, "create_trained_platform_policy", create_platform_policy)
    monkeypatch.setattr(serve_kai0_policy.normalize, "deserialize_json", deserialize_stats)
    monkeypatch.setattr(serve_kai0_policy.websocket_policy_server, "WebsocketPolicyServer", create_server)
    argv = [
        "serve_kai0_policy.py",
        "--checkpoint",
        str(checkpoint),
        "--preprocessing",
        "platform-joint14",
        "--default-prompt",
        "Fold the purple cloth.",
    ]
    if stats_location == "explicit":
        argv.extend(["--norm-stats", str(stats_path)])
    if rtc:
        argv.append("--rtc")
    monkeypatch.setattr(sys, "argv", argv)

    serve_kai0_policy.main()

    create_policy.assert_not_called()
    deserialize_stats.assert_not_called()
    create_platform_policy.assert_called_once()
    config, loaded_checkpoint = create_platform_policy.call_args.args
    assert config.name == (serve_kai0_policy.RTC_CONFIG if rtc else serve_kai0_policy.DEFAULT_CONFIG)
    assert config.model.model_type.value == ("pi05_rtc" if rtc else "pi05")
    assert loaded_checkpoint == checkpoint
    kwargs = create_platform_policy.call_args.kwargs
    assert kwargs["norm_stats_path"] == stats_path
    assert kwargs["default_prompt"] == "Fold the purple cloth."
    assert "norm_stats" not in kwargs
    raw_observation = {
        "observation.images.top_head": "top",
        "observation.images.hand_left": "left",
        "observation.images.hand_right": "right",
        "observation.state": "state14",
        "prompt": "instruction",
    }
    assert kwargs["repack_transforms"].inputs[0](raw_observation) == {
        "images": {"top_head": "top", "hand_left": "left", "hand_right": "right"},
        "state": "state14",
        "prompt": "instruction",
    }
    create_server.assert_called_once()
    assert create_server.call_args.kwargs["policy"] is policy
    metadata = create_server.call_args.kwargs["metadata"]
    assert metadata["platform_detail"] == "preserved"
    assert metadata["preprocessing"] == "platform-joint14"
    assert metadata["norm_stats_path"] == str(stats_path)
    assert metadata["supports_rtc"] is rtc
    assert metadata["rtc_config"] == (serve_kai0_policy.RTC_CONFIG if rtc else None)
    server.serve_forever.assert_called_once_with()
