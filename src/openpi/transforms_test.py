import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


@pytest.mark.parametrize("use_quantiles", [False, True])
@pytest.mark.parametrize("shape", [(14,), (32,), (50, 32), (2, 50, 32)])
def test_normalize_preserves_uncovered_dimensions(use_quantiles, shape):
    stats = _transforms.NormStats(
        mean=np.full(14, 2.0),
        std=np.full(14, 4.0),
        q01=np.full(14, -2.0),
        q99=np.full(14, 6.0),
    )
    values = np.full(shape, 4.0)
    values[..., 14:] = np.linspace(0.0, 1.0, shape[-1] - 14)
    original = values.copy()

    normalized = _transforms.Normalize({"state": stats}, use_quantiles=use_quantiles)({"state": values})

    assert normalized["state"].shape == shape
    np.testing.assert_allclose(normalized["state"][..., :14], 0.5, atol=1e-6)
    np.testing.assert_array_equal(normalized["state"][..., 14:], original[..., 14:])
    np.testing.assert_array_equal(values, original)
    restored = _transforms.Unnormalize({"state": stats}, use_quantiles=use_quantiles)(normalized)
    np.testing.assert_allclose(restored["state"], original, atol=1e-6)


@pytest.mark.parametrize("use_quantiles", [False, True])
@pytest.mark.parametrize("stats_dim", [32, 40])
def test_normalize_full_stats(use_quantiles, stats_dim):
    stats = _transforms.NormStats(
        mean=np.full(stats_dim, 2.0),
        std=np.full(stats_dim, 4.0),
        q01=np.full(stats_dim, -2.0),
        q99=np.full(stats_dim, 6.0),
    )

    normalized = _transforms.Normalize({"actions": stats}, use_quantiles=use_quantiles)(
        {"actions": np.full((50, 32), 4.0)}
    )

    assert normalized["actions"].shape == (50, 32)
    np.testing.assert_allclose(normalized["actions"], 0.5, atol=1e-6)


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
