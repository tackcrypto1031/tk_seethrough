import sys, pathlib, types
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: str(ROOT / "output")
fp.get_input_directory = lambda: str(ROOT / "input")
fp.get_annotated_filepath = lambda n: n
fp.models_dir = str(ROOT / "models")
sys.modules.setdefault("folder_paths", fp)
sys.modules.setdefault("comfy", types.ModuleType("comfy"))
sys.modules.setdefault("comfy.model_management", types.ModuleType("comfy.model_management"))
import nodes

def test_savepsd_has_optional_inputs():
    it = nodes.SeeThrough_SavePSD.INPUT_TYPES()
    assert "optional" in it
    assert "original_image" in it["optional"]
    assert "source_filename" in it["optional"]

def test_savepsd_signature():
    import inspect
    sig = inspect.signature(nodes.SeeThrough_SavePSD.save)
    assert "original_image" in sig.parameters
    assert "source_filename" in sig.parameters
