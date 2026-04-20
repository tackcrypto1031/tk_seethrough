"""Tests for _assert_text_encoder_loaded — regression for issue #6.

See https://github.com/tackcrypto1031/tk_seethrough/issues/6

Diffusers may silently substitute nn.Identity for a missing text_encoder
(e.g. when model_index.json does not list it). The helper must fail fast
on any empty-placeholder module, not just nn.Identity.
"""
import sys
import pathlib
import types

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

folder_paths_stub = types.ModuleType("folder_paths")
folder_paths_stub.get_output_directory = lambda: str(ROOT / "output")
folder_paths_stub.get_input_directory = lambda: str(ROOT / "input")
folder_paths_stub.get_annotated_filepath = lambda name: name
folder_paths_stub.models_dir = str(ROOT / "models")
sys.modules.setdefault("folder_paths", folder_paths_stub)

comfy_mm_stub = types.ModuleType("comfy.model_management")
sys.modules.setdefault("comfy", types.ModuleType("comfy"))
sys.modules.setdefault("comfy.model_management", comfy_mm_stub)

from nodes import _assert_text_encoder_loaded


class _RealTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


def test_passes_on_module_with_parameters():
    _assert_text_encoder_loaded(_RealTextEncoder(), "text_encoder", "/fake/path")


def test_raises_on_identity_placeholder():
    with pytest.raises(RuntimeError) as exc:
        _assert_text_encoder_loaded(torch.nn.Identity(), "text_encoder", "/fake/path")
    msg = str(exc.value)
    assert "empty placeholder" in msg
    assert "/fake/path" in msg
    assert "issues/6" in msg


def test_raises_on_any_empty_placeholder():
    class CustomEmpty(torch.nn.Module):
        pass

    with pytest.raises(RuntimeError):
        _assert_text_encoder_loaded(CustomEmpty(), "text_encoder_2", "/fake/path")


def test_raises_on_none():
    with pytest.raises(RuntimeError):
        _assert_text_encoder_loaded(None, "text_encoder", "/fake/path")


def test_error_includes_component_name():
    with pytest.raises(RuntimeError) as exc:
        _assert_text_encoder_loaded(torch.nn.Identity(), "text_encoder_2", "/model")
    assert "text_encoder_2" in str(exc.value)
