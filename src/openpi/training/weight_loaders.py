import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        loaded_params = maybe_remap_siglip_scan_params(loaded_params, params)
        # Missing LoRA + MEM-only projections keep their freshly initialized values.
        return _merge_params(
            loaded_params,
            params,
            missing_regex=".*lora.*|.*proprio_history_proj.*",
        )


def maybe_remap_siglip_scan_params(loaded_params: at.Params, ref_params: at.Params) -> at.Params:
    """Unstack SigLIP ``encoderblock`` weights when the reference model uses MEM (scan=False)."""
    flat_ref = flax.traverse_util.flatten_dict(ref_params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    needs_unmap = any("encoderblock_0" in k for k in flat_ref) and any(
        "/encoderblock/" in k or k.endswith("/encoderblock") for k in flat_loaded
    )
    if not needs_unmap:
        # Also detect scanned key form: .../encoderblock/MultiHeadDotProductAttention_0/...
        needs_unmap = any("encoderblock_0" in k for k in flat_ref) and any(
            re.search(r"/encoderblock/", k) for k in flat_loaded
        )
    if not needs_unmap:
        return loaded_params

    # Infer depth from a scanned attention kernel's leading axis.
    depth = None
    for k, v in flat_loaded.items():
        if "/encoderblock/" in k and k.endswith("/kernel"):
            depth = int(v.shape[0])
            break
    if depth is None:
        return loaded_params

    from openpi.models import mem as _mem

    logger.info("Remapping scanned SigLIP encoderblock weights to unstacked MEM layout (depth=%d)", depth)
    return _mem.remap_siglip_scan_to_unstacked(loaded_params, depth=depth)

@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
