# Issue #5 — PSD Base Layer & Preserve Filename — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the original input image as a visible base layer in generated PSDs, preserve the user's source filename in output filenames and PSD structure, and group output layers (`Parts`/`Runs`/`Original`) for easier editing.

**Architecture:** Python side adds one new ComfyUI node (`SeeThrough_LoadSource`) and extends `SeeThrough_PostProcess` + `SeeThrough_SavePSD` to carry/persist the original image and a sanitized source filename. JS side (`web/seethrough_psd.js`) reads extended JSON metadata and builds a grouped PSD tree with the original as the visible bottom layer.

**Tech Stack:** Python 3 (ComfyUI custom node, numpy, Pillow), JavaScript (ag-psd in-browser PSD writer).

**Spec:** `docs/superpowers/specs/2026-04-16-issue5-psd-base-layer-design.md`

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `nodes.py` | modify | New `SeeThrough_LoadSource` class; `_sanitize_filename` helper; `PostProcess` pipes `input_img`; `SavePSD` new inputs, filename rule, base-layer save, JSON schema fields; registry entries. |
| `web/seethrough_psd.js` | modify | `createPSD` builds grouped tree with `Parts` (hidden) + `Original` (visible bottom). `createAllRunsPSD` adds `Original` base. Download filename uses server-computed `base`. |
| `workflows/tk_see_throug_workflow.json` | modify | Swap `LoadImage` for `SeeThrough_LoadSource`; wire `source_filename` output to `SavePSD`. |
| `README.md`, `README_ZH.md` | modify | Document new node + behavior changes under v1.2.8. |
| `tests/test_sanitize_filename.py` | create | Unit tests for the sanitize helper (only pure-Python unit we can easily test outside ComfyUI). |

Manual verification in ComfyUI is required for end-to-end checks; record results in commit messages.

---

## Task 1: Add `_sanitize_filename` helper with unit tests

**Files:**
- Modify: `nodes.py` (add helper near top of file, after imports)
- Create: `tests/test_sanitize_filename.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sanitize_filename.py`:
```python
import os
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub ComfyUI-only imports so nodes.py is importable in unit tests.
import types
folder_paths_stub = types.ModuleType("folder_paths")
folder_paths_stub.get_output_directory = lambda: str(ROOT / "output")
folder_paths_stub.get_input_directory = lambda: str(ROOT / "input")
folder_paths_stub.get_annotated_filepath = lambda name: name
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sanitize_filename.py -v`
Expected: FAIL with `ImportError: cannot import name '_sanitize_filename' from 'nodes'`

- [ ] **Step 3: Write the helper**

In `nodes.py`, directly after the `import traceback` line (around line 14), add:
```python
import re


def _sanitize_filename(name):
    """Windows-safe filename sanitization; preserves Unicode and spaces.

    Replaces `<>:"|?*\\/\x00` with underscore; strips trailing dots/spaces.
    """
    if not name:
        return ""
    cleaned = re.sub(r'[<>:"|?*\\/\x00]', '_', str(name))
    return cleaned.rstrip(' .')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sanitize_filename.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add nodes.py tests/test_sanitize_filename.py
git commit -m "feat(nodes): add _sanitize_filename helper with unit tests"
```

---

## Task 2: Add `SeeThrough_LoadSource` node

**Files:**
- Modify: `nodes.py` (add class before `NODE_CLASS_MAPPINGS`; register in both mappings)

- [ ] **Step 1: Add the class**

In `nodes.py`, immediately before the `NODE_CLASS_MAPPINGS` dict (around line 1427), add:
```python
class SeeThrough_LoadSource:
    """Loads an image like ComfyUI's LoadImage but also outputs the source filename.

    Used to feed `SeeThrough_SavePSD.source_filename` so the final PSD keeps the
    user's original filename instead of a timestamp.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        try:
            files = [
                f for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
            ]
        except FileNotFoundError:
            files = []
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "source_filename")
    FUNCTION = "load"
    CATEGORY = "SeeThrough"

    def load(self, image):
        from PIL import Image, ImageOps
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        rgb = img.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None,]  # [1, H, W, 3]

        if "A" in img.getbands():
            mask_arr = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask_arr)
        else:
            mask = torch.zeros((tensor.shape[1], tensor.shape[2]), dtype=torch.float32)

        basename = os.path.splitext(os.path.basename(image))[0]
        source_filename = _sanitize_filename(basename)

        if tensor.shape[0] > 1:
            print(f"[SeeThrough] LoadSource: batch>1 not supported, using [0]", flush=True)
            tensor = tensor[:1]

        return (tensor, mask.unsqueeze(0) if mask.ndim == 2 else mask, source_filename)
```

- [ ] **Step 2: Register the node**

Modify `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` (around lines 1428 and 1441) by adding one entry each:

```python
NODE_CLASS_MAPPINGS = {
    "SeeThrough_LoadSource": SeeThrough_LoadSource,  # NEW
    "SeeThrough_LoadLayerDiffModel": SeeThrough_LoadLayerDiffModel,
    # ... existing entries unchanged ...
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeeThrough_LoadSource": "SeeThrough Load Source",  # NEW
    "SeeThrough_LoadLayerDiffModel": "SeeThrough Load LayerDiff Model",
    # ... existing entries unchanged ...
}
```

- [ ] **Step 3: Smoke-test import**

Run: `python -c "import sys; sys.path.insert(0,'.'); import types; sys.modules['folder_paths']=types.SimpleNamespace(get_input_directory=lambda:'.', get_annotated_filepath=lambda n:n, get_output_directory=lambda:'.'); sys.modules['comfy']=types.ModuleType('comfy'); sys.modules['comfy.model_management']=types.ModuleType('comfy.model_management'); import nodes; print(nodes.SeeThrough_LoadSource.RETURN_NAMES)"`
Expected: `('image', 'mask', 'source_filename')`

- [ ] **Step 4: Commit**

```bash
git add nodes.py
git commit -m "feat(nodes): add SeeThrough_LoadSource node (outputs source_filename)"
```

---

## Task 3: Pipe `input_img` through `SeeThrough_PostProcess`

**Files:**
- Modify: `nodes.py:1041-1042` (PostProcess `parts_data` construction)

- [ ] **Step 1: Add `input_img` to parts_data**

In `nodes.py`, find the `parts_data = {"tag2pinfo": tag2pinfo, "frame_size": frame_size, "all_runs_layers": all_runs_layers}` line (around 1041) and change it to:
```python
parts_data = {
    "tag2pinfo": tag2pinfo,
    "frame_size": frame_size,
    "all_runs_layers": all_runs_layers,
    "input_img": layers_data.input_img,  # RGBA numpy, for PSD base layer
}
```

- [ ] **Step 2: Smoke-test import**

Run: `python -c "import sys,types; sys.modules['folder_paths']=types.SimpleNamespace(get_input_directory=lambda:'.', get_annotated_filepath=lambda n:n, get_output_directory=lambda:'.'); sys.modules['comfy']=types.ModuleType('comfy'); sys.modules['comfy.model_management']=types.ModuleType('comfy.model_management'); import nodes; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add nodes.py
git commit -m "feat(postprocess): pipe input_img through parts_data for PSD base layer"
```

---

## Task 4: Extend `SeeThrough_SavePSD` — optional inputs, filename rule, base-layer PNG, JSON fields

**Files:**
- Modify: `nodes.py:1053-1163` (SavePSD class)

- [ ] **Step 1: Update `INPUT_TYPES`**

Replace the existing `INPUT_TYPES` classmethod (around line 1054) with:
```python
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parts": ("SEETHROUGH_PARTS",),
                "filename_prefix": ("STRING", {"default": "seethrough"}),
            },
            "optional": {
                "original_image": ("IMAGE",),
                "source_filename": ("STRING", {"default": ""}),
            },
        }
```

- [ ] **Step 2: Update `save()` signature and add filename helper at top**

Replace the `def save(self, parts, filename_prefix="seethrough"):` line with:
```python
    def save(self, parts, filename_prefix="seethrough", original_image=None, source_filename=""):
        from PIL import Image
        import json

        tag2pinfo = parts["tag2pinfo"]
        frame_size = parts["frame_size"]
        canvas_h, canvas_w = frame_size

        output_dir = folder_paths.get_output_directory()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]

        src = _sanitize_filename(source_filename)
        if src and filename_prefix:
            base = f"{filename_prefix}_{src}_{uid}"
        elif src:
            base = f"{src}_{uid}"
        else:
            base = f"{filename_prefix}_{ts}_{uid}"
```

Keep the remainder of the function body but rename every existing `{filename_prefix}_{ts}_{uid}` f-string to use `{base}` instead. Concretely, in the current function, the lines to update are:

- `layer_filename = f"{filename_prefix}_{ts}_{uid}_{tag}.png"` → `layer_filename = f"{base}_{tag}.png"`
- `depth_filename = f"{filename_prefix}_{ts}_{uid}_{tag}_depth.png"` → `depth_filename = f"{base}_{tag}_depth.png"`
- `run_filename = f"{filename_prefix}_{ts}_{uid}_run{run_idx}_{tag}.png"` → `run_filename = f"{base}_run{run_idx}_{tag}.png"`
- `info_filename = f"{filename_prefix}_{ts}_{uid}_layers.json"` → `info_filename = f"{base}_layers.json"`

- [ ] **Step 3: Save the base-layer PNG and add JSON fields**

Between the `uid = ...` / `base = ...` block and the existing `sorted_tags = ...` line, insert:
```python
        base_img_np = None
        if original_image is not None:
            src_tensor = original_image[0] if original_image.ndim == 4 else original_image
            base_img_np = (src_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            if base_img_np.ndim == 3 and base_img_np.shape[2] == 3:
                alpha = np.full(base_img_np.shape[:2] + (1,), 255, dtype=np.uint8)
                base_img_np = np.concatenate([base_img_np, alpha], axis=2)
            if base_img_np.shape[0] != canvas_h or base_img_np.shape[1] != canvas_w:
                print(
                    f"[SeeThrough] WARNING: original_image size "
                    f"({base_img_np.shape[1]}x{base_img_np.shape[0]}) mismatches "
                    f"frame_size ({canvas_w}x{canvas_h}). Resizing.",
                    flush=True,
                )
                pil = Image.fromarray(base_img_np, mode="RGBA")
                pil = pil.resize((canvas_w, canvas_h), Image.LANCZOS)
                base_img_np = np.array(pil)
        elif parts.get("input_img") is not None:
            base_img_np = parts["input_img"]
            if base_img_np.ndim == 3 and base_img_np.shape[2] == 3:
                alpha = np.full(base_img_np.shape[:2] + (1,), 255, dtype=np.uint8)
                base_img_np = np.concatenate([base_img_np, alpha], axis=2)

        source_filename_saved = None
        if base_img_np is not None:
            source_filename_saved = f"{base}_original.png"
            Image.fromarray(base_img_np).save(os.path.join(output_dir, source_filename_saved))
```

Then update the `info_data` assembly. Replace:
```python
        info_data = {"prefix": filename_prefix, "timestamp": f"{ts}_{uid}",
                     "layers": layer_info_list, "width": int(canvas_w), "height": int(canvas_h)}
```
with:
```python
        info_data = {
            "prefix": filename_prefix,
            "timestamp": f"{ts}_{uid}",
            "layers": layer_info_list,
            "width": int(canvas_w),
            "height": int(canvas_h),
            "base": base,
            "source_name": src,
            "source_filename": source_filename_saved,
        }
```

- [ ] **Step 4: Smoke-test import**

Run: `python -c "import sys,types; sys.modules['folder_paths']=types.SimpleNamespace(get_input_directory=lambda:'.', get_annotated_filepath=lambda n:n, get_output_directory=lambda:'.'); sys.modules['comfy']=types.ModuleType('comfy'); sys.modules['comfy.model_management']=types.ModuleType('comfy.model_management'); import nodes; print(nodes.SeeThrough_SavePSD.INPUT_TYPES()['optional'])"`
Expected: prints dict containing `original_image` and `source_filename` keys.

- [ ] **Step 5: Commit**

```bash
git add nodes.py
git commit -m "feat(savepsd): accept original_image/source_filename, write base PNG, new filename rule"
```

---

## Task 5: JS — build grouped PSD with Original base layer

**Files:**
- Modify: `web/seethrough_psd.js` (rewrite `createPSD`, update `createAllRunsPSD`)

- [ ] **Step 1: Rewrite `createPSD`**

Replace the whole `createPSD` function (currently lines 49-118) with:
```javascript
async function createPSD(layerInfo, psdType) {
    await ensureAgPsdLoaded();

    const {
        layers, width, height, prefix, timestamp,
        base, source_name, source_filename,
    } = layerInfo;
    const isDepth = psdType === "depth";
    const suffix = isDepth ? "_depth" : "";

    console.log(`[SeeThrough] Creating ${isDepth ? "depth " : ""}PSD: ${width}x${height}, ${layers.length} layers`);

    const compositeCanvas = document.createElement("canvas");
    compositeCanvas.width = width;
    compositeCanvas.height = height;
    const compositeCtx = compositeCanvas.getContext("2d");

    // Build Parts group (hidden by default).
    const partLayers = [];
    for (const layer of layers) {
        const filenameKey = isDepth ? "depth_filename" : "filename";
        const filename = layer[filenameKey];
        if (!filename) continue;

        const url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=output&t=${Date.now()}`);
        const img = await loadImage(url);

        const lw = layer.right - layer.left;
        const lh = layer.bottom - layer.top;

        const layerCanvas = document.createElement("canvas");
        layerCanvas.width = lw;
        layerCanvas.height = lh;
        const layerCtx = layerCanvas.getContext("2d");
        layerCtx.drawImage(img, 0, 0, lw, lh);

        compositeCtx.drawImage(img, layer.left, layer.top, lw, lh);

        partLayers.push({
            name: layer.name,
            canvas: layerCanvas,
            left: layer.left,
            top: layer.top,
            right: layer.right,
            bottom: layer.bottom,
            blendMode: "normal",
            opacity: 1,
        });
    }

    const children = [];
    children.push({
        name: "Parts",
        hidden: true,
        opened: false,
        children: partLayers,
    });

    // Base layer (colored original, visible by default, bottom).
    if (source_filename) {
        const origUrl = api.apiURL(`/view?filename=${encodeURIComponent(source_filename)}&type=output&t=${Date.now()}`);
        try {
            const origImg = await loadImage(origUrl);
            const origCanvas = document.createElement("canvas");
            origCanvas.width = width;
            origCanvas.height = height;
            origCanvas.getContext("2d").drawImage(origImg, 0, 0, width, height);
            // Draw behind composite so the exported thumbnail shows the original too.
            const prevComposite = compositeCtx.globalCompositeOperation;
            compositeCtx.globalCompositeOperation = "destination-over";
            compositeCtx.drawImage(origImg, 0, 0, width, height);
            compositeCtx.globalCompositeOperation = prevComposite;
            children.push({
                name: "Original",
                hidden: false,
                canvas: origCanvas,
                left: 0,
                top: 0,
                right: width,
                bottom: height,
                blendMode: "normal",
                opacity: 1,
            });
        } catch (e) {
            console.warn("[SeeThrough] Failed to load original base layer:", e);
        }
    }

    const psd = { width, height, canvas: compositeCanvas, children };
    const psdBuffer = window.AgPsd.writePsd(psd);
    const blob = new Blob([psdBuffer], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const downloadBase = base || `${prefix}_${timestamp}`;
    a.href = url;
    a.download = `${downloadBase}${suffix}.psd`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`[SeeThrough] PSD downloaded: ${a.download}`);
}
```

- [ ] **Step 2: Update `createAllRunsPSD`**

In `createAllRunsPSD`, after the existing `for (const runData of all_runs)` loop that populates `runGroups`, add an Original base layer before building the final `psd` object. Replace the block starting at `// Create group (folder) for this run` through the final download section with:

```javascript
        runGroups.push({
            name: `Run ${runData.run} (seed=${runData.seed}, ${runData.layer_count} layers)`,
            children: groupLayers,
            opened: false,
            hidden: true,
        });
    }

    const children = [{
        name: "Runs",
        hidden: true,
        opened: false,
        children: runGroups,
    }];

    if (layerInfo.source_filename) {
        const origUrl = api.apiURL(`/view?filename=${encodeURIComponent(layerInfo.source_filename)}&type=output&t=${Date.now()}`);
        try {
            const origImg = await loadImage(origUrl);
            const origCanvas = document.createElement("canvas");
            origCanvas.width = width;
            origCanvas.height = height;
            origCanvas.getContext("2d").drawImage(origImg, 0, 0, width, height);
            const prev = compositeCtx.globalCompositeOperation;
            compositeCtx.globalCompositeOperation = "destination-over";
            compositeCtx.drawImage(origImg, 0, 0, width, height);
            compositeCtx.globalCompositeOperation = prev;
            children.push({
                name: "Original",
                hidden: false,
                canvas: origCanvas,
                left: 0, top: 0, right: width, bottom: height,
                blendMode: "normal",
                opacity: 1,
            });
        } catch (e) {
            console.warn("[SeeThrough] Failed to load original base layer:", e);
        }
    }

    const psd = { width, height, canvas: compositeCanvas, children };
    const psdBuffer = window.AgPsd.writePsd(psd);
    const blob = new Blob([psdBuffer], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const downloadBase = layerInfo.base || `${prefix}_${timestamp}`;
    a.href = url;
    a.download = `${downloadBase}_all_runs.psd`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`[SeeThrough] All Runs PSD downloaded: ${a.download}`);
}
```

- [ ] **Step 3: Manual verification checklist**

In the commit message, include results of:
1. Open PSD in Photoshop → `Original` visible at bottom, `Parts` group collapsed/hidden.
2. Toggle `Parts` visibility on → composite matches previous behavior.
3. Depth PSD → `Original` visible at bottom, Parts group hidden.
4. All-runs PSD → `Runs` group hidden, `Original` visible.
5. Console shows no errors; download filename matches `{base}.psd`.

- [ ] **Step 4: Commit**

```bash
git add web/seethrough_psd.js
git commit -m "feat(web): grouped PSD (Parts hidden + Original visible base layer)"
```

---

## Task 6: Update example workflow + README

**Files:**
- Modify: `workflows/tk_see_throug_workflow.json`
- Modify: `README.md`, `README_ZH.md`

- [ ] **Step 1: Update workflow JSON**

Open `workflows/tk_see_throug_workflow.json`. Find the `LoadImage` node. Change its `type` field from `"LoadImage"` to `"SeeThrough_LoadSource"`. In the node's `outputs` array, add a third output entry mirroring the existing two, with:
```json
{ "name": "source_filename", "type": "STRING", "links": [<NEW_LINK_ID>] }
```
In the top-level `links` array, add a new link from this new output to the `SeeThrough_SavePSD` node's `source_filename` input:
```json
[<NEW_LINK_ID>, <LOADSOURCE_NODE_ID>, 2, <SAVEPSD_NODE_ID>, <SAVEPSD_SOURCE_FILENAME_INPUT_INDEX>, "STRING"]
```
On the `SeeThrough_SavePSD` node, add `source_filename` and `original_image` entries to its `inputs` array with `link` set to the new link id for `source_filename` and `null` for `original_image`.

(If this manual JSON edit is risky, an alternative is: load workflow in ComfyUI, rewire visually, export JSON, replace file.)

- [ ] **Step 2: Update READMEs**

Append a section under "What's new" (or create one) in both `README.md` and `README_ZH.md`:

**English (`README.md`):**
```markdown
### v1.2.8 — Issue #5

- New node **SeeThrough Load Source**: same dropdown as ComfyUI's LoadImage, plus outputs `source_filename` for preserving the original filename in PSD output.
- **SeeThrough Save PSD** now accepts optional `original_image` + `source_filename` inputs, automatically includes the original input image as a visible base layer in the generated PSD, and uses the source filename in output filenames when available.
- PSD layer structure is now grouped: `Original` (visible, bottom), `Parts` (hidden), `Runs` (hidden, grouped-PSD mode only) — opens the PSD on the original so you can toggle groups to edit specific parts.
```

**Chinese (`README_ZH.md`):**
```markdown
### v1.2.8 — Issue #5

- 新增節點 **SeeThrough Load Source**:行為與 ComfyUI LoadImage 相同,額外輸出 `source_filename`,讓最終 PSD 保留原始檔名。
- **SeeThrough Save PSD** 新增 optional 輸入 `original_image` 與 `source_filename`,自動將原始輸入圖作為可見底圖放入 PSD,且輸出檔名會保留原檔名。
- PSD 圖層結構改為分組:`Original`(底層可見)/`Parts`(隱藏)/`Runs`(隱藏,僅 grouped PSD 模式),開啟 PSD 即見原圖,依需要展開 group 修局部。
```

- [ ] **Step 3: Commit**

```bash
git add workflows/tk_see_throug_workflow.json README.md README_ZH.md
git commit -m "docs+workflow: switch example to LoadSource + document issue #5 changes"
```

---

## Task 7: End-to-end manual verification in ComfyUI

**Files:** none (verification only)

- [ ] **Step 1: Restart ComfyUI and reload the example workflow.** Confirm node registrations: `SeeThrough Load Source` appears in the `SeeThrough` category.

- [ ] **Step 2: Run workflow with a source image named `my_char.png`.** Confirm:
  - `output/my_char_<uid>_original.png` exists.
  - `output/my_char_<uid>_layers.json` contains `source_name: "my_char"` and `source_filename: "my_char_<uid>_original.png"`.
  - Downloaded PSD is named `my_char_<uid>.psd`.
  - In Photoshop: `Original` visible at bottom, `Parts` group hidden. Toggling `Parts` visibility reveals composite.

- [ ] **Step 3: Run workflow using the old `LoadImage` node (no `source_filename`).** Confirm:
  - PSD still generates.
  - Filename falls back to legacy `seethrough_<ts>_<uid>.psd`.
  - `Original` still appears in PSD because `parts["input_img"]` auto-piped through PostProcess.

- [ ] **Step 4: Run grouped PSD mode (auto_fill with multiple runs).** Confirm `Runs` group and `Original` coexist; download filename is `<base>_all_runs.psd`.

- [ ] **Step 5: Download Depth PSD.** Confirm colored `Original` is bottom visible, Parts hidden.

- [ ] **Step 6: Record verification results.**

```bash
git commit --allow-empty -m "chore: verify issue #5 end-to-end (base layer, grouped PSD, filename)"
```

---

## Self-Review Notes

- **Spec coverage:** Q1-Q8 each map to a task: Q1 auto-pipe → Task 3; Q1 override → Task 4; Q2 group structure → Task 5; Q3 filename rule → Task 4; Q4 LoadSource → Task 2; Q5 batch warning → Task 2; Q6 Depth base → Task 5; Q7 sanitize → Task 1; Q8 resize+warn → Task 4.
- **Backward compatibility:** legacy `LoadImage` path preserved — Task 4 resolution order falls through to `parts["input_img"]`; Task 5 JS skips `Original` when `source_filename` absent.
- **No placeholders:** all code blocks self-contained; no TBD / "similar to above".
- **Tests:** only `_sanitize_filename` has unit tests (pure function). Node-level testing is manual in ComfyUI per Task 7 — this repo has no existing pytest harness for ComfyUI nodes, and bootstrapping one is out of scope.
