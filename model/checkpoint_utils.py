"""Checkpoint/index version coupling (fix #7).

Retraining the model without rebuilding the index fails silently — the
embedding count still matches, but every prediction is garbage. The index
stores a hash of the checkpoint's state dict at build time; CardIdentifier
recomputes it at load time and refuses to run on a mismatch.
"""

import hashlib
import json
from pathlib import Path

INDEX_INFO_FILENAME = "index_info.json"


def state_dict_hash(state_dict: dict) -> str:
    """Deterministic hash of a model state dict (keys, shapes, dtypes, values)."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        t = state_dict[key]
        h.update(key.encode())
        h.update(str(getattr(t, "dtype", type(t))).encode())
        h.update(str(tuple(getattr(t, "shape", ()))).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_index_info(index_dir: str | Path, checkpoint: dict) -> Path:
    """Write checkpoint provenance next to the index files."""
    info = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "state_dict_hash":  state_dict_hash(checkpoint["model_state_dict"]),
    }
    path = Path(index_dir) / INDEX_INFO_FILENAME
    path.write_text(json.dumps(info, indent=2))
    return path


def verify_index_info(index_dir: str | Path, checkpoint: dict) -> None:
    """Raise if the index was built from a different checkpoint.

    Legacy indexes (built before this coupling existed) have no info file —
    warn and instruct to rebuild rather than hard-failing.
    """
    path = Path(index_dir) / INDEX_INFO_FILENAME
    if not path.exists():
        import warnings
        warnings.warn(
            f"{path} not found — index predates checkpoint/version coupling. "
            "Rebuild with model/build_index.py to enable verification.",
            stacklevel=2,
        )
        return

    info = json.loads(path.read_text())
    current = state_dict_hash(checkpoint["model_state_dict"])
    if info.get("state_dict_hash") != current:
        raise RuntimeError(
            "Index/checkpoint mismatch: the index was built from a different "
            f"model (index epoch {info.get('checkpoint_epoch')}). Every "
            "prediction would be garbage. Rebuild the index: "
            "uv run python model/build_index.py --checkpoint <your checkpoint>"
        )
