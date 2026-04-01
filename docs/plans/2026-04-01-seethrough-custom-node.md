# SeeThrough_GenerateLayers_Custom Node Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `SeeThrough_GenerateLayers_Custom` node to the ComfyUI-See-through plugin that lets users toggle which semantic tags to generate via boolean checkboxes.

**Architecture:** Clone the existing repo, add a new node class in `nodes.py` that copies the original `generate()` logic but replaces hardcoded tag lists with user-selected tags filtered by runtime `tag_version`. Register in existing mappings.

**Tech Stack:** Python, ComfyUI custom node API, PyTorch, NumPy, OpenCV

---

### Task 1: Clone the repository

**Files:**
- Create: entire repo at `D:/tack_project/tk_seethrough/`

**Step 1: Clone ComfyUI-See-through**

```bash
cd D:/tack_project/tk_seethrough
git clone https://github.com/jtydhr88/ComfyUI-See-through .
```

Note: Clone into current directory (which already has `docs/`). If git complains about non-empty directory, use:
```bash
git init
git remote add origin https://github.com/jtydhr88/ComfyUI-See-through.git
git fetch origin
git checkout -b main origin/main
```

**Step 2: Verify clone**

```bash
ls nodes.py __init__.py
```
Expected: both files exist.

**Step 3: Commit docs that were already in place**

```bash
git add docs/
git commit -m "docs: add custom node spec and design documents"
```

---

### Task 2: Add tag constants for the Custom node

**Files:**
- Modify: `nodes.py` (after the existing `VALID_BODY_PARTS_V2` constant, around line 78)

**Step 1: Add the three tag list constants and the unified parameter map**

Insert after `VALID_BODY_PARTS_V2 = [...]` (line 78) and before `SEETHROUGH_MODELS_DIR`:

```python
VALID_BODY_PARTS_V3_BODY = [
    "front hair", "back hair", "head", "neck", "neckwear",
    "topwear", "handwear", "bottomwear", "legwear", "footwear",
    "tail", "wings", "objects",
]

VALID_BODY_PARTS_V3_HEAD = [
    "headwear", "face", "irides", "eyebrow", "eyewhite",
    "eyelash", "eyewear", "ears", "earwear", "nose", "mouth",
]

# All unique tags across v2 + v3 (24 total), used for boolean INPUT_TYPES
ALL_TAGS = list(dict.fromkeys(
    VALID_BODY_PARTS_V2 + VALID_BODY_PARTS_V3_BODY + VALID_BODY_PARTS_V3_HEAD
))
```

**Step 2: Verify constant correctness**

Manually confirm:
- `VALID_BODY_PARTS_V2` has 19 items
- `VALID_BODY_PARTS_V3_BODY` has 13 items
- `VALID_BODY_PARTS_V3_HEAD` has 11 items
- `ALL_TAGS` has 24 unique items (union of all three)

**Step 3: Commit**

```bash
git add nodes.py
git commit -m "feat: add v3 body/head tag constants and ALL_TAGS union for custom node"
```

---

### Task 3: Implement SeeThrough_GenerateLayers_Custom class

**Files:**
- Modify: `nodes.py` (insert new class after `SeeThrough_GenerateLayers` class, before `SeeThrough_GenerateDepth` class — approximately after line 330)

**Step 1: Add the new class with INPUT_TYPES**

Insert the following class after the closing of `SeeThrough_GenerateLayers`:

```python
class SeeThrough_GenerateLayers_Custom:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "layerdiff_model": ("SEETHROUGH_LAYERDIFF_MODEL",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                "resolution": ("INT", {"default": 1280, "min": 512, "max": 2048, "step": 64}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            },
        }
        for tag in ALL_TAGS:
            param_name = tag.replace(" ", "_")
            inputs["required"][param_name] = ("BOOLEAN", {"default": True, "tooltip": tag})
        return inputs

    RETURN_TYPES = ("SEETHROUGH_LAYERS", "IMAGE")
    RETURN_NAMES = ("layers", "preview")
    FUNCTION = "generate"
    CATEGORY = "SeeThrough"

    def generate(self, image, layerdiff_model, seed=42, resolution=1280, num_inference_steps=30, **kwargs):
        pipeline = layerdiff_model

        # Collect user-selected tags
        selected_tags = set()
        for tag in ALL_TAGS:
            param_name = tag.replace(" ", "_")
            if kwargs.get(param_name, True):
                selected_tags.add(tag)

        seed_everything(seed)

        # Convert ComfyUI IMAGE to numpy RGBA
        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if img_np.shape[-1] == 3:
            img_np = np.concatenate([img_np, np.full((*img_np.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
        input_img = img_np.copy()

        fullpage, pad_size, pad_pos = center_square_pad_resize(input_img, resolution, return_pad_info=True)
        scale = pad_size[0] / resolution
        rng = torch.Generator(device=pipeline.unet.device).manual_seed(seed)

        tag_version = pipeline.unet.get_tag_version()
        layer_dict = {}

        print(f"[SeeThrough] GenerateLayers_Custom: tag_version={tag_version}, resolution={resolution}, steps={num_inference_steps}", flush=True)

        if tag_version == "v2":
            active_tags = [t for t in VALID_BODY_PARTS_V2 if t in selected_tags]
            if not active_tags:
                raise ValueError("At least one valid tag must be selected for the current model's tag version (v2).")

            out = pipeline(strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                           generator=rng, guidance_scale=1.0, prompt=active_tags,
                           negative_prompt="", fullpage=fullpage)
            for rst, tag in zip(out.images, active_tags):
                layer_dict[tag] = rst

        elif tag_version == "v3":
            active_body_tags = [t for t in VALID_BODY_PARTS_V3_BODY if t in selected_tags]
            active_head_tags = [t for t in VALID_BODY_PARTS_V3_HEAD if t in selected_tags]

            if not active_body_tags and not active_head_tags:
                raise ValueError("At least one valid tag must be selected for the current model's tag version (v3).")

            if active_body_tags:
                out = pipeline(strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                               generator=rng, guidance_scale=1.0, prompt=active_body_tags,
                               negative_prompt="", fullpage=fullpage, group_index=0)
                for rst, tag in zip(out.images, active_body_tags):
                    layer_dict[tag] = rst

            # Head detail stage: only if "head" was selected AND generated, AND there are head tags
            if "head" in active_body_tags and active_head_tags and "head" in layer_dict:
                head_img = layer_dict["head"]
                nz = cv2.findNonZero((head_img[..., -1] > 15).astype(np.uint8))
                if nz is not None:
                    hx0, hy0, hw, hh = cv2.boundingRect(nz)
                    hx = int(hx0 * scale) - pad_pos[0]
                    hy = int(hy0 * scale) - pad_pos[1]
                    input_head, (hx1, hy1, hx2, hy2) = _crop_head(input_img, [hx, hy, int(hw * scale), int(hh * scale)])
                    hx1 = int(hx1 / scale + pad_pos[0] / scale)
                    hy1 = int(hy1 / scale + pad_pos[1] / scale)
                    ih, iw = input_head.shape[:2]
                    input_head, head_pad_size, head_pad_pos = center_square_pad_resize(input_head, resolution, return_pad_info=True)

                    out = pipeline(strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                                   generator=rng, guidance_scale=1.0, prompt=active_head_tags,
                                   negative_prompt="", fullpage=input_head, group_index=1)

                    canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)
                    coords = np.array([head_pad_pos[1], head_pad_pos[1] + ih, head_pad_pos[0], head_pad_pos[0] + iw])
                    py1, py2, px1, px2 = (coords / scale).astype(np.int64)
                    scale_size = (int(head_pad_size[0] / scale), int(head_pad_size[1] / scale))

                    for rst, tag in zip(out.images, active_head_tags):
                        rst = smart_resize(rst, scale_size)[py1:py2, px1:px2]
                        full = canvas.copy()
                        full[hy1:hy1 + rst.shape[0], hx1:hx1 + rst.shape[1]] = rst
                        layer_dict[tag] = full
        else:
            raise ValueError(f"Unknown tag version: {tag_version}")

        print(f"[SeeThrough] GenerateLayers_Custom complete: {len(layer_dict)} layers: {list(layer_dict.keys())}", flush=True)

        layers_data = SeeThrough_LayersData(layer_dict, fullpage, input_img, resolution, pad_size, pad_pos)

        preview_dict = {}
        for tag, img in layer_dict.items():
            mask = img[..., -1] > 10
            if np.any(mask):
                preview_dict[tag] = {"img": img, "xyxy": [0, 0, resolution, resolution]}
        preview = _make_preview(preview_dict, resolution)

        return (layers_data, preview)
```

**Step 2: Verify the class is syntactically correct**

```bash
python -c "import ast; ast.parse(open('nodes.py').read()); print('Syntax OK')"
```
Expected: `Syntax OK`

**Step 3: Commit**

```bash
git add nodes.py
git commit -m "feat: add SeeThrough_GenerateLayers_Custom node with boolean tag toggles"
```

---

### Task 4: Register the new node in mappings

**Files:**
- Modify: `nodes.py` (the `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` dicts at the end of the file)

**Step 1: Add the new node to NODE_CLASS_MAPPINGS**

In the `NODE_CLASS_MAPPINGS` dict (around line 1107), add:

```python
    "SeeThrough_GenerateLayers_Custom": SeeThrough_GenerateLayers_Custom,
```

**Step 2: Add the new node to NODE_DISPLAY_NAME_MAPPINGS**

In the `NODE_DISPLAY_NAME_MAPPINGS` dict (around line 1116), add:

```python
    "SeeThrough_GenerateLayers_Custom": "SeeThrough Generate Layers (Custom)",
```

**Step 3: Verify registration**

```bash
python -c "
import ast
with open('nodes.py') as f:
    content = f.read()
assert 'SeeThrough_GenerateLayers_Custom' in content
print('Registration found')
tree = ast.parse(content)
print('Syntax OK')
"
```
Expected: Both lines printed.

**Step 4: Commit**

```bash
git add nodes.py
git commit -m "feat: register SeeThrough_GenerateLayers_Custom in node mappings"
```

---

### Task 5: Final verification

**Files:**
- Read: `nodes.py` (verify complete file)

**Step 1: Verify all tag counts**

```bash
python -c "
exec(open('nodes.py').read().split('class SeeThrough_LoadLayerDiffModel')[0])
print(f'V2 tags: {len(VALID_BODY_PARTS_V2)} (expect 19)')
print(f'V3 body tags: {len(VALID_BODY_PARTS_V3_BODY)} (expect 13)')
print(f'V3 head tags: {len(VALID_BODY_PARTS_V3_HEAD)} (expect 11)')
print(f'ALL_TAGS: {len(ALL_TAGS)} (expect 24)')
assert len(VALID_BODY_PARTS_V2) == 19
assert len(VALID_BODY_PARTS_V3_BODY) == 13
assert len(VALID_BODY_PARTS_V3_HEAD) == 11
assert len(ALL_TAGS) == 24
print('All counts correct!')
"
```
Expected: All assertions pass.

**Step 2: Verify INPUT_TYPES has all 24 booleans + 5 original params**

```bash
python -c "
# Quick check: count BOOLEAN occurrences in INPUT_TYPES
with open('nodes.py') as f:
    content = f.read()
# Find the Custom class
idx = content.index('class SeeThrough_GenerateLayers_Custom')
class_code = content[idx:content.index('\nclass ', idx+1)] if content.find('\nclass ', idx+1) != -1 else content[idx:]
bool_count = class_code.count('\"BOOLEAN\"')
print(f'BOOLEAN params in Custom node: {bool_count}')
# Booleans are generated dynamically via the loop, so check ALL_TAGS length
print('(Booleans are added dynamically from ALL_TAGS in the for loop)')
print('Verification: check that for loop over ALL_TAGS exists')
assert 'for tag in ALL_TAGS' in class_code
print('Dynamic boolean generation confirmed!')
"
```

**Step 3: Verify original node is untouched**

```bash
python -c "
with open('nodes.py') as f:
    content = f.read()
# Original class should still exist with hardcoded tags
assert 'class SeeThrough_GenerateLayers:' in content
# Original should NOT reference ALL_TAGS
idx = content.index('class SeeThrough_GenerateLayers:')
end_idx = content.index('class SeeThrough_GenerateLayers_Custom')
original_class = content[idx:end_idx]
assert 'ALL_TAGS' not in original_class
print('Original SeeThrough_GenerateLayers is untouched!')
"
```

**Step 4: Verify node count in mappings**

```bash
python -c "
with open('nodes.py') as f:
    content = f.read()
assert content.count('SeeThrough_GenerateLayers_Custom') >= 3  # class def + 2 mappings
print('Node registered in both mappings!')
"
```

**Step 5: Commit (if any fixes were needed)**

If all checks pass, no commit needed. If fixes were made:
```bash
git add nodes.py
git commit -m "fix: correct any issues found during verification"
```
