import sys, pathlib, types
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse stubs (duplicated since tests may run in isolation)
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: str(ROOT / "output")
fp.get_input_directory = lambda: str(ROOT / "input")
fp.get_annotated_filepath = lambda name: name
fp.models_dir = str(ROOT / "models")
sys.modules.setdefault("folder_paths", fp)
sys.modules.setdefault("comfy", types.ModuleType("comfy"))
sys.modules.setdefault("comfy.model_management", types.ModuleType("comfy.model_management"))

import nodes

def test_loadsource_registered():
    assert "SeeThrough_LoadSource" in nodes.NODE_CLASS_MAPPINGS
    assert "SeeThrough_LoadSource" in nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert nodes.SeeThrough_LoadSource.RETURN_NAMES == ("image", "mask", "source_filename")
    assert nodes.SeeThrough_LoadSource.CATEGORY == "SeeThrough"
