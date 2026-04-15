# Design: Issue #5 — Original file as base layer + preserve source filename

**Date:** 2026-04-16
**Issue:** [#5](https://github.com/tackcrypto1031/tk_seethrough/issues/5) by @vivi-gomez
**Scope:** ComfyUI path only (`nodes.py` + `web/seethrough_psd.js`). CLI path (`inference_psd.py`) out of scope.

## Intent

Issue asks for two things:
1. Save the original input image as a **background base layer** in the final PSD, so users can quickly compare / fix specific parts in Photoshop.
2. **Preserve the original filename** in PSD and per-layer filenames, instead of timestamp/UUID-only names that make batch output untraceable.

Bundled into this spec (scope option B): light cleanup of the PSD layer structure (grouping) to make the resulting PSD easier to navigate.

## Design decisions (Q&A resolved)

| # | Decision |
|---|---|
| Q1 | Original image is sourced via **both**: `parts_data` auto-pipes `layers_data.input_img` (zero-config for existing workflows); optional `original_image` input on `SavePSD` overrides when provided. |
| Q2 | PSD layer structure: `Parts` group (hidden by default), `Runs` group (hidden, only in grouped PSD mode), `Original` single layer at the bottom (**visible by default**). |
| Q3 | Filename rule: `source_filename` + `filename_prefix` interaction per user choice (D+B): prefix empty → `{source}_{uid}.psd`; both set → `{prefix}_{source}_{uid}.psd`; source empty → legacy `{prefix}_{ts}_{uid}.psd`. |
| Q4 | Source filename discovery: new `SeeThrough_LoadSource` node (wraps LoadImage, outputs `IMAGE, MASK, STRING`) + `SavePSD` accepts optional `source_filename` string. Both may be used. |
| Q5 | Batch: single-image only (matches upstream pipeline). `LoadSource` returns `[1, H, W, C]`; warn if batch>1. |
| Q6 | Depth PSD also gets colored original as base layer (same PNG reused). |
| Q7 | Filename sanitization: Windows-safe — strip `<>:"|?*\\/\x00` + trailing dot/space, preserve Unicode and spaces. |
| Q8 | External `original_image` with size mismatching `frame_size`: resize to `frame_size` + emit console warning. |

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐     ┌──────────┐
│ LoadSource      │─►IMG─►GenerateLayers│─►LD─►PostProcess  │─►PD─►SavePSD       │─►JSON─►JS PSD  │
│ (new)           │─►STR──────────────────┐   │ + input_img │     │ + orig_image │     │ builder  │
└─────────────────┘                       │   │ into PD     │     │ + src_fname  │     └──────────┘
                                          └──►(source_fname)──────►(override chain)
```

- `LD` = `SeeThrough_LayersData` (unchanged; already carries `input_img`)
- `PD` = parts_data dict — **extended** with `input_img` key
- JS reads extended JSON metadata to insert Original layer + build groups.

## Components

### 1. `SeeThrough_LoadSource` (new node, `nodes.py`)

- **Inputs:** `image` (same dropdown as ComfyUI `LoadImage`, lists files in `input/`)
- **Outputs:** `IMAGE`, `MASK`, `STRING (source_filename)`
- **source_filename:** basename without extension, passed through `_sanitize_filename`.
- **Batch handling:** if input tensor has N>1, take `[0]` and `print` a warning.
- **Category:** `SeeThrough`

### 2. `SeeThrough_PostProcess` modification (`nodes.py`)

Add one line — include `input_img` (RGBA numpy) from `layers_data` into the returned `parts_data` dict:

```python
parts_data = {
    "tag2pinfo": tag2pinfo,
    "frame_size": frame_size,
    "all_runs_layers": all_runs_layers,
    "input_img": layers_data.input_img,   # NEW
}
```

### 3. `SeeThrough_SavePSD` modification (`nodes.py`)

**New optional inputs:**
```python
"optional": {
    "original_image": ("IMAGE",),
    "source_filename": ("STRING", {"default": ""}),
}
```

**Resolution order for base layer image:**
1. `original_image` if provided (resize + warn if mismatch with `frame_size`)
2. else `parts["input_img"]` (auto-piped from PostProcess)
3. else skip base layer (legacy behavior)

**Filename rule (`_resolve_names`):**
```
src = sanitize(source_filename)
if src and prefix:   base = f"{prefix}_{src}_{uid}"
elif src:            base = f"{src}_{uid}"
else:                base = f"{prefix}_{ts}_{uid}"   # legacy
```

`base` is used for: `{base}_{tag}.png`, `{base}_original.png`, `{base}_layers.json`.
JS builds PSD as `{base}.psd` / `{base}_depth.psd`.

**Sanitize helper:**
```python
def _sanitize_filename(name: str) -> str:
    if not name:
        return ""
    cleaned = re.sub(r'[<>:"|?*\\/\x00]', '_', name)
    return cleaned.rstrip(' .')
```

**Base layer save:**
- Write `{base}_original.png` to `output_dir` — full canvas RGBA matching `frame_size`.
- Add `source_filename: "{base}_original.png"` and `source_name: src` into JSON.

### 4. JSON schema extension

```jsonc
{
  "prefix": "...",
  "source_name": "my_char",                   // NEW, for display
  "source_filename": "..._original.png",      // NEW, PNG reference
  "timestamp": "...",
  "width": 1024, "height": 1536,
  "layers": [...],
  "all_runs": [...]
}
```

Absent fields ⇒ JS skips base layer (backward compatible).

### 5. `web/seethrough_psd.js` rewrite of `createPSD`

Build tree:
```
root
├─ Parts (group, hidden=true)
│   └─ layer per part (depth_median sorted, existing logic)
├─ Runs (group, hidden=true, only if all_runs present)
│   └─ Run N (seed=...) (group)
│       └─ layer per part
└─ Original (layer, hidden=false, bottom)
```

Base layer is loaded via same `/view?filename=...` pattern, drawn at canvas size (no offset — sizes already match).

Downloaded filename derived from JSON `source_name` / `prefix` — JS no longer builds its own; reuses the `base` computed on the server.

### 6. Workflow + docs

- `workflows/tk_see_throug_workflow.json`: swap `LoadImage` → `SeeThrough_LoadSource`, route its STRING output into `SavePSD.source_filename`.
- README (EN + ZH): one paragraph describing new node + behavior change under a "v1.2.8" note.

## Testing

1. Run existing workflow without changes → base layer appears automatically (input_img auto-pipe); filename still legacy.
2. Swap in `LoadSource` → PSD named after source file, PSD structure grouped.
3. Open PSD in Photoshop → Original visible, Parts group hidden; toggle Parts on to compose.
4. Run grouped PSD mode → Runs group appears.
5. Edge cases: empty source_filename, Unicode/space filename, Windows reserved chars (`a?b:c.png` → `a_b_c.png`).
6. Depth PSD contains colored Original at bottom.
7. External `original_image` override with mismatched size → resize + warning printed.

## Risk & rollback

- Fully backward-compatible: absent JSON fields and absent optional inputs reproduce legacy behavior.
- No change to inference code paths, preprocessing, or depth computation.
- CLI path (`inference_psd.py`) untouched — future issue if needed.

## Out of scope

- Batch (N>1 images in one run).
- CLI `save_psd` parity.
- Custom group-name / visibility UI knobs (grouped layout is hardcoded per Q2).
- Smart positioning when external original has different size (resize only).
