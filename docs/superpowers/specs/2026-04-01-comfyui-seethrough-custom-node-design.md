# Design: SeeThrough_GenerateLayers_Custom Node

## Overview

Add a new ComfyUI node `SeeThrough_GenerateLayers_Custom` to the existing ComfyUI-See-through plugin that exposes 24 boolean toggles for semantic layer tags, allowing users to select which layers to generate instead of always generating all layers.

## Prerequisites

- Clone https://github.com/jtydhr88/ComfyUI-See-through into the working directory
- All modifications happen inside the cloned repo

## Node Definition

- **Class name:** `SeeThrough_GenerateLayers_Custom`
- **Display name:** `SeeThrough Generate Layers (Custom)`
- **Category:** `SeeThrough`
- **File:** `nodes.py` (append after existing `SeeThrough_GenerateLayers` class)

## INPUT_TYPES

Retains the original 5 required parameters, plus 24 BOOLEAN toggles (all default `True`):

### Original parameters (unchanged)

| Parameter | Type | Default | Constraints |
|-----------|------|---------|-------------|
| image | IMAGE | - | - |
| layerdiff_model | SEETHROUGH_LAYERDIFF_MODEL | - | - |
| seed | INT | 42 | 0 ~ 2^32-1 |
| resolution | INT | 1280 | 512 ~ 2048, step 64 |
| num_inference_steps | INT | 30 | 1 ~ 100 |

### Boolean tag toggles (24 total)

Parameter naming: spaces replaced with underscores. Tooltip shows original tag name.

| Parameter name | Tooltip | V2 (19) | V3 body (13) | V3 head (11) |
|----------------|---------|---------|--------------|--------------|
| hair | hair | x | | |
| headwear | headwear | x | | x |
| face | face | x | | x |
| eyes | eyes | x | | |
| eyewear | eyewear | x | | x |
| ears | ears | x | | x |
| earwear | earwear | x | | x |
| nose | nose | x | | x |
| mouth | mouth | x | | x |
| neck | neck | x | x | |
| neckwear | neckwear | x | x | |
| topwear | topwear | x | x | |
| handwear | handwear | x | x | |
| bottomwear | bottomwear | x | x | |
| legwear | legwear | x | x | |
| footwear | footwear | x | x | |
| tail | tail | x | x | |
| wings | wings | x | x | |
| objects | objects | x | x | |
| front_hair | front hair | | x | |
| back_hair | back hair | | x | |
| head | head | | x | |
| irides | irides | | | x |
| eyebrow | eyebrow | | | x |
| eyewhite | eyewhite | | | x |
| eyelash | eyelash | | | x |

## RETURN_TYPES

Identical to the original node:
- `("SEETHROUGH_LAYERS", "IMAGE")`
- Return names: `("layers", "preview")`

## Execution Logic

### Tag constants

Three lists defined at module level (or as class constants):

```
V2_TAGS = ["hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear",
           "nose", "mouth", "neck", "neckwear", "topwear", "handwear",
           "bottomwear", "legwear", "footwear", "tail", "wings", "objects"]

V3_BODY_TAGS = ["front hair", "back hair", "head", "neck", "neckwear",
                "topwear", "handwear", "bottomwear", "legwear", "footwear",
                "tail", "wings", "objects"]

V3_HEAD_TAGS = ["headwear", "face", "irides", "eyebrow", "eyewhite",
                "eyelash", "eyewear", "ears", "earwear", "nose", "mouth"]
```

### Step 1: Collect selected tags

Read all 24 boolean kwargs. For each, map parameter name back to original tag name (underscore -> space). Collect tags where value is `True`.

### Step 2: Filter by tag_version

At runtime, get `tag_version = pipeline.unet.get_tag_version()`.

- **v2:** `active_tags = [t for t in V2_TAGS if t in selected_tags]`
- **v3 body:** `active_body_tags = [t for t in V3_BODY_TAGS if t in selected_tags]`
- **v3 head:** `active_head_tags = [t for t in V3_HEAD_TAGS if t in selected_tags]`

Tags selected but not in the current version's list are silently ignored.

### Step 3: Validate

If no valid tags remain after filtering, raise an error: `"At least one valid tag must be selected for the current model's tag version."`

### Step 4: Inference

- **v2 path:** Replace hardcoded tag list with `active_tags`. All other logic (center_square_pad_resize, pipeline call, preview generation) identical to original.
- **v3 body path:** Replace hardcoded body tag list with `active_body_tags`.
- **v3 head path:** If `"head"` is NOT in `active_body_tags`, skip the entire head detail generation stage. Otherwise, replace hardcoded head tag list with `active_head_tags`. If `active_head_tags` is empty but head was selected, skip head detail stage as well (no head tags to process).

### Step 5: Output

Package results into `SeeThrough_LayersData` and generate preview, identical to original node. Only generated layers are included in the layer_dict.

## Registration

Append to existing mappings in `nodes.py`:

```python
NODE_CLASS_MAPPINGS["SeeThrough_GenerateLayers_Custom"] = SeeThrough_GenerateLayers_Custom
NODE_DISPLAY_NAME_MAPPINGS["SeeThrough_GenerateLayers_Custom"] = "SeeThrough Generate Layers (Custom)"
```

## What stays unchanged

- Original `SeeThrough_GenerateLayers` node: no modifications
- `__init__.py`: no modifications (imports mappings from nodes.py)
- Downstream nodes (`GenerateDepth`, `PostProcess`, `SavePSD`): no modifications, fully compatible
- All preprocessing utilities (`center_square_pad_resize`, etc.): no modifications

## Implementation approach

The new class copies the `generate()` method logic from the original rather than inheriting, because tag filtering needs to be inserted inside the inference loop. This avoids coupling with the original class while keeping both nodes independently maintainable.

## Error handling

- Zero valid tags after filtering: raise descriptive error
- Tags not matching current tag_version: silently ignored
- All other error handling inherits from original node behavior
