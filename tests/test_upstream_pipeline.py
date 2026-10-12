"""Pipeline regressions without downloading or running diffusion checkpoints."""
import ast
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

import nodes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cpu_runtime(monkeypatch):
    monkeypatch.setattr(nodes.mm, "get_torch_device", lambda: torch.device("cpu"), raising=False)
    monkeypatch.setattr(nodes.mm, "soft_empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(nodes, "_log_vram", lambda *args: None)
    monkeypatch.setattr(nodes, "_make_preview", lambda *args: torch.zeros(1, 1, 1, 3))


@pytest.mark.parametrize("mode", ["RGBA", "P", "RGB"])
def test_load_source_roundtrips_source_pixels_and_alpha(tmp_path, monkeypatch, mode):
    rgba = np.zeros((8, 12, 4), dtype=np.uint8)
    rgba[..., :3] = [123, 67, 201]
    rgba[2:6, 3:9, 3] = 127
    image = Image.fromarray(rgba)
    if mode == "P":
        image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=4)
    elif mode == "RGB":
        image = image.convert("RGB")
    source = tmp_path / "source.png"
    image.save(source)
    expected = np.array(Image.open(source).convert("RGBA"))
    monkeypatch.setattr(nodes.folder_paths, "get_annotated_filepath", lambda _: str(source))
    rgb, mask, _ = nodes.SeeThrough_LoadSource().load("source.png")
    np.testing.assert_array_equal(nodes._image_to_rgba(rgb, mask), expected)


def test_old_workflows_without_mask_still_accept_rgb():
    image = torch.ones((1, 8, 12, 3))
    assert np.all(nodes._image_to_rgba(image)[..., 3] == 255)


def test_mask_mismatch_fails_before_inference():
    with pytest.raises(ValueError, match="dimensions"):
        nodes._image_to_rgba(torch.zeros(1, 8, 12, 3), torch.ones(1, 7, 12))


def test_standard_load_image_opaque_mask_placeholder_is_compatible():
    rgba = nodes._image_to_rgba(torch.ones(1, 80, 120, 3), torch.zeros(1, 64, 64))
    assert rgba.shape == (80, 120, 4)
    assert np.all(rgba[..., 3] == 255)


class FakeComponent:
    device = torch.device("cpu")

    def to(self, *args, **kwargs):
        return self

    def get_tag_version(self):
        return "v3"


class DeterministicLayerPipeline:
    """Prescribed model output lets us compare actual upstream crop/reprojection."""
    def __init__(self):
        self.unet = FakeComponent()
        self.vae = FakeComponent()
        self.trans_vae = FakeComponent()
        self.text_encoder = FakeComponent()
        self.text_encoder_2 = FakeComponent()
        self.calls = []

    def encode_cropped_prompt_77tokens(self, tags):
        return list(tags), None

    def __call__(self, **kwargs):
        tags = kwargs.get("prompt", kwargs.get("prompt_embeds"))
        fullpage = kwargs["fullpage"]
        self.calls.append((list(tags), fullpage.copy(), kwargs))
        h, w = fullpage.shape[:2]
        output = []
        for i, tag in enumerate(tags):
            arr = np.zeros((h, w, 4), dtype=np.uint8)
            arr[..., :3] = [30 + i, 70 + i, 110 + i]
            if tag == "head":
                arr[h // 8:h // 3, w // 3:2 * w // 3, 3] = 255
            else:
                arr[h // 4:3 * h // 4, w // 4:3 * w // 4, 3] = 255
            output.append(arr)
        return SimpleNamespace(images=output)


@pytest.mark.parametrize("node_class", [nodes.SeeThrough_GenerateLayers, nodes.SeeThrough_GenerateLayers_Custom])
def test_v3_facial_crop_and_all_layers_match_upstream(tmp_path, cpu_runtime, node_class):
    source = np.zeros((256, 128, 4), dtype=np.uint8)
    source[..., :3] = [101, 149, 213]
    source[10:246, 5:123, 3] = 255
    srcp = tmp_path / "character.png"
    Image.fromarray(source).save(srcp)

    # Execute the repo's upstream apply_layerdiff function with a prescribed
    # model, then compare its saved PNGs with the public ComfyUI node outputs.
    path = ROOT / "see-through/common/utils/inference_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_layerdiff")
    upstream = DeterministicLayerPipeline()
    namespace = {
        "layerdiff_pipeline": upstream,
        "VALID_BODY_PARTS_V2": nodes.VALID_BODY_PARTS_V2,
        "np": np, "cv2": cv2, "torch": torch, "Image": Image,
        "os": os, "osp": os.path,
        "center_square_pad_resize": nodes.center_square_pad_resize,
        "smart_resize": nodes.smart_resize,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    namespace["apply_layerdiff"](str(srcp), "unused", seed=42, resolution=64, save_dir=str(tmp_path))

    pipeline = DeterministicLayerPipeline()
    image = torch.from_numpy(source[..., :3].astype(np.float32) / 255)[None]
    mask = 1 - torch.from_numpy(source[..., 3].astype(np.float32) / 255)[None]
    layers, _ = node_class().generate(image, pipeline, seed=42, resolution=64, mask=mask)
    assert len(layers.layer_dict) == 24
    assert len(pipeline.calls) == len(upstream.calls) == 2
    for call, ref in zip(pipeline.calls, upstream.calls):
        assert call[0] == ref[0]
        np.testing.assert_array_equal(call[1], ref[1])
        for key in ("num_inference_steps", "guidance_scale", "strength", "group_index"):
            assert call[2][key] == ref[2][key]
    for tag, layer in layers.layer_dict.items():
        np.testing.assert_array_equal(layer, np.array(Image.open(tmp_path / "character" / f"{tag}.png")))


class DeterministicDepthPipeline:
    def to(self, *args, **kwargs):
        return self

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        h, w = kwargs["img_list"][0].shape[:2]
        depth = np.full((20, h, w), 0.401, dtype=np.float32)
        return SimpleNamespace(depth_tensor=torch.from_numpy(depth))


@pytest.mark.parametrize("resolution_depth,expected", [(768, 768), (-1, 128)])
def test_depth_input_resolution_and_upstream_png_precision(cpu_runtime, resolution_depth, expected):
    img = np.full((128, 128, 4), 255, dtype=np.uint8)
    layers = nodes.SeeThrough_LayersData({"face": img}, img, img, 128, (128, 128), (0, 0))
    model = DeterministicDepthPipeline()
    result, _ = nodes.SeeThrough_GenerateDepth().generate(layers, model, resolution_depth=resolution_depth)
    assert {x.shape for x in model.kwargs["img_list"]} == {(expected, expected, 4)}
    assert model.kwargs["denoising_steps"] is None
    assert result.depth_dict["face"].shape == (128, 128)
    np.testing.assert_allclose(result.depth_dict["face"], 102 / 255)


def test_body_only_keeps_head_through_depth_and_postprocess(cpu_runtime):
    img = np.full((64, 64, 4), 255, dtype=np.uint8)
    layers = nodes.SeeThrough_LayersData({"head": img}, img, img, 64, (64, 64), (0, 0))
    depth, _ = nodes.SeeThrough_GenerateDepth().generate(layers, DeterministicDepthPipeline(), resolution_depth=64)
    assert "head" in depth.depth_dict and "face" not in depth.depth_dict
    parts, _ = nodes.SeeThrough_PostProcess().process(depth, tblr_split=False)
    assert "head" in parts["tag2pinfo"]


def test_facial_parts_survive_depth_and_left_right_postprocess(cpu_runtime):
    res = 64
    face = np.full((res, res, 4), 255, dtype=np.uint8)
    layers = {"face": face}
    for tag in nodes.VALID_BODY_PARTS_V3_HEAD:
        if tag == "face":
            continue
        arr = np.zeros_like(face)
        arr[12:18, 10:16] = 255
        arr[12:18, 42:48] = 255
        layers[tag] = arr
    data = nodes.SeeThrough_LayersData(layers, face, face, res, (res, res), (0, 0))
    depth, _ = nodes.SeeThrough_GenerateDepth().generate(data, DeterministicDepthPipeline(), resolution_depth=res)
    parts, _ = nodes.SeeThrough_PostProcess().process(depth)
    tags = set(parts["tag2pinfo"])
    assert {"face", "nose", "mouth", "headwear", "eyewear", "earwear"} <= tags
    for tag in ("irides", "eyebrow", "eyewhite", "eyelash", "ears"):
        assert {f"{tag}-l", f"{tag}-r"} <= tags


def test_v3_features_are_kept_above_face_and_split_ears_behind(cpu_runtime):
    res = 64
    opaque = np.full((res, res, 4), 255, dtype=np.uint8)
    paired = np.zeros_like(opaque)
    paired[10:16, 8:14] = 255
    paired[10:16, 42:48] = 255
    layer_dict = {"face": opaque, "irides": paired, "eyebrow": paired, "ears": paired}
    depth_dict = {
        "face": np.full((res, res), 0.4, dtype=np.float32),
        "irides": np.full((res, res), 0.7, dtype=np.float32),
        "eyebrow": np.full((res, res), 0.7, dtype=np.float32),
        "ears": np.full((res, res), 0.1, dtype=np.float32),
    }
    data = nodes.SeeThrough_LayersDepthData(layer_dict, depth_dict, opaque, res)
    parts, _ = nodes.SeeThrough_PostProcess().process(data)
    p = parts["tag2pinfo"]
    for tag in ("irides-l", "irides-r", "eyebrow-l", "eyebrow-r"):
        assert p[tag]["depth_median"] < p["face"]["depth_median"]
    for tag in ("ears-l", "ears-r"):
        assert p[tag]["depth_median"] > p["face"]["depth_median"]


def test_rename_then_filter_preserves_split_facial_parts_and_source_metadata():
    tags = {f"{feature}-{side}": {"tag": f"{feature}-{side}"} for feature in ("irides", "eyebrow", "eyewhite", "eyelash", "ears", "handwear") for side in ("l", "r")}
    source = np.zeros((8, 8, 4), dtype=np.uint8)
    parts = {"tag2pinfo": tags, "frame_size": (8, 8), "fullpage": source, "all_runs_layers": []}
    renamed, = nodes.SeeThrough_LayerRename().rename(parts)
    allowed = nodes.SeeThrough_LayerFilter.INPUT_TYPES()["required"]["tags"][1]["default"]
    filtered, = nodes.SeeThrough_LayerFilter().filter_layers(renamed, mode="include", tags=allowed)
    assert len(filtered["tag2pinfo"]) == 12
    assert filtered["fullpage"] is source
    assert filtered["all_runs_layers"] == []


def test_psd_reference_uses_same_padding_as_generated_parts(tmp_path, monkeypatch):
    source = np.full((16, 8, 4), 255, dtype=np.uint8)
    source[..., :3] = [25, 150, 200]
    image = torch.from_numpy(source[..., :3].astype(np.float32) / 255)[None]
    mask = torch.zeros(1, 16, 8)
    monkeypatch.setattr(nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))
    path, = nodes.SeeThrough_SavePSD().save(
        {"tag2pinfo": {}, "frame_size": (32, 32)},
        original_image=image, original_mask=mask,
    )
    info = json.loads(Path(path).read_text(encoding="utf-8"))
    actual = np.array(Image.open(tmp_path / info["source_filename"]))
    expected = nodes.center_square_pad_resize(source, 32)
    np.testing.assert_array_equal(actual, expected)


def test_workflows_have_valid_links_alpha_connection_and_fixed_comparison_settings():
    for filename in ("tk_see_throug_workflow.json", "tk_seethrough_upstream_parity.json"):
        workflow = json.loads((ROOT / "workflows" / filename).read_text(encoding="utf-8"))
        surfaces = [(workflow["nodes"], workflow["links"])]
        surfaces += [(g["nodes"], g["links"]) for g in workflow.get("definitions", {}).get("subgraphs", [])]
        all_nodes = {}
        for graph_nodes, links in surfaces:
            byid = {n["id"]: n for n in graph_nodes}
            all_nodes.update(byid)
            for link in links:
                if isinstance(link, dict):
                    link_id, origin, oslot, target, tslot, kind = (link[k] for k in ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type"))
                else:
                    link_id, origin, oslot, target, tslot, kind = link
                if origin >= 0:
                    out = byid[origin]["outputs"][oslot]
                    assert out["type"] == kind and link_id in out["links"]
                if target >= 0:
                    inp = byid[target]["inputs"][tslot]
                    assert inp["type"] == kind and inp["link"] == link_id
        gen = next(n for n in all_nodes.values() if n["type"].startswith("SeeThrough_GenerateLayers"))
        mask = next(x for x in gen["inputs"] if x["name"] == "mask")
        assert mask["link"] is not None
        assert gen["widgets_values"][:4] == [42, "fixed", 1280, 30]
        if gen["type"].endswith("_Custom"):
            assert gen["widgets_values"][4:6] == [True, False]
        depth = next(n for n in all_nodes.values() if n["type"] == "SeeThrough_GenerateDepth")
        assert depth["widgets_values"] == [42, "fixed", 768, -1]


def test_actual_browser_psd_writer_order_and_visible_composite(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node.js is needed to exercise the bundled browser PSD writer")
    psd_tools = pytest.importorskip("psd_tools")
    subprocess.run(["node", str(ROOT / "tests" / "psd_export_fixture.cjs"), str(tmp_path)], check=True)
    for filename, group_name in (("normal.psd", "Parts"), ("depth.psd", "Parts"), ("runs.psd", "Runs")):
        psd = psd_tools.PSDImage.open(tmp_path / filename)
        assert [layer.name for layer in psd] == ["Original", group_name]
        assert psd.composite(force=True).getpixel((0, 0))[:3] == (0, 255, 0)
        if group_name == "Parts":
            assert [layer.name for layer in psd[1]] == ["back", "front"]
            assert psd.composite().getpixel((0, 0))[:3] == (0, 255, 0)
            psd[1].visible = True
            assert psd.composite(force=True).getpixel((0, 0))[:3] == (0, 0, 255)

