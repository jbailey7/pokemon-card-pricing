import warnings

import pytest
import torch

from model.checkpoint_utils import (
    INDEX_INFO_FILENAME,
    state_dict_hash,
    verify_index_info,
    write_index_info,
)


def make_checkpoint(seed=0, epoch=7):
    g = torch.Generator().manual_seed(seed)
    return {
        "epoch": epoch,
        "model_state_dict": {
            "layer.weight": torch.randn(4, 4, generator=g),
            "layer.bias":   torch.randn(4, generator=g),
        },
    }


class TestStateDictHash:
    def test_deterministic(self):
        a = make_checkpoint(seed=1)["model_state_dict"]
        b = make_checkpoint(seed=1)["model_state_dict"]
        assert state_dict_hash(a) == state_dict_hash(b)

    def test_different_values_different_hash(self):
        a = make_checkpoint(seed=1)["model_state_dict"]
        b = make_checkpoint(seed=2)["model_state_dict"]
        assert state_dict_hash(a) != state_dict_hash(b)

    def test_key_rename_changes_hash(self):
        sd = make_checkpoint()["model_state_dict"]
        renamed = {k + "_x": v for k, v in sd.items()}
        assert state_dict_hash(sd) != state_dict_hash(renamed)

    def test_insensitive_to_key_order(self):
        sd = make_checkpoint()["model_state_dict"]
        reordered = dict(reversed(list(sd.items())))
        assert state_dict_hash(sd) == state_dict_hash(reordered)

    def test_non_contiguous_tensor(self):
        sd = {"w": torch.randn(4, 4).t()}  # transpose → non-contiguous
        assert isinstance(state_dict_hash(sd), str)


class TestWriteVerifyRoundtrip:
    def test_matching_checkpoint_passes_silently(self, tmp_path):
        ckpt = make_checkpoint()
        write_index_info(tmp_path, ckpt)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            verify_index_info(tmp_path, ckpt)

    def test_mismatched_checkpoint_raises(self, tmp_path):
        write_index_info(tmp_path, make_checkpoint(seed=1))
        with pytest.raises(RuntimeError, match="mismatch"):
            verify_index_info(tmp_path, make_checkpoint(seed=2))

    def test_legacy_index_warns_but_does_not_raise(self, tmp_path):
        # No info file written — legacy index
        with pytest.warns(UserWarning, match="predates"):
            verify_index_info(tmp_path, make_checkpoint())

    def test_info_file_records_epoch(self, tmp_path):
        import json
        path = write_index_info(tmp_path, make_checkpoint(epoch=42))
        assert path.name == INDEX_INFO_FILENAME
        assert json.loads(path.read_text())["checkpoint_epoch"] == 42
