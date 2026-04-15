import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import types
folder_paths_stub = types.ModuleType("folder_paths")
folder_paths_stub.get_output_directory = lambda: str(ROOT / "output")
folder_paths_stub.get_input_directory = lambda: str(ROOT / "input")
folder_paths_stub.get_annotated_filepath = lambda name: name
folder_paths_stub.models_dir = str(ROOT / "models")
sys.modules.setdefault("folder_paths", folder_paths_stub)

comfy_mm_stub = types.ModuleType("comfy.model_management")
sys.modules.setdefault("comfy", types.ModuleType("comfy"))
sys.modules.setdefault("comfy.model_management", comfy_mm_stub)

from nodes import _sanitize_filename


def test_empty_returns_empty():
    assert _sanitize_filename("") == ""
    assert _sanitize_filename(None) == ""


def test_strips_path_separators():
    assert _sanitize_filename("a/b\\c") == "a_b_c"


def test_strips_windows_reserved_chars():
    assert _sanitize_filename('a<b>c:d"e|f?g*h') == "a_b_c_d_e_f_g_h"


def test_preserves_unicode_and_spaces():
    assert _sanitize_filename("my 角色 01") == "my 角色 01"


def test_strips_null_byte():
    assert _sanitize_filename("a\x00b") == "a_b"


def test_rstrips_trailing_dot_and_space():
    assert _sanitize_filename("name.  ") == "name"
    assert _sanitize_filename("name.") == "name"
