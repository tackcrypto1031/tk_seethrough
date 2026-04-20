# Issue #6 — `'Identity' object has no attribute 'device'` on ComfyUI v0.19.3

**Reported:** works on ComfyUI v0.16.3, fails on ComfyUI v0.19.3 with:

```
File ".../see-through/common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py", line 143, in encode_cropped_prompt_77tokens
    device = self.text_encoder.device
AttributeError: 'Identity' object has no attribute 'device'
```

Upstream issue: https://github.com/tackcrypto1031/tk_seethrough/issues/6

## Root cause

Newer versions of `diffusers` (shipped with ComfyUI v0.19.3) silently substitute
`torch.nn.Identity()` as a placeholder when a component listed in the pipeline's
`_optional_components` cannot be resolved from the checkpoint's `model_index.json`.

- `KDiffusionStableDiffusionXLPipeline._optional_components` lists `text_encoder`
  and `text_encoder_2` (inherited from `StableDiffusionXLImg2ImgPipeline`).
- When the model is loaded, diffusers treats them as skippable and substitutes
  `nn.Identity()` if loading fails or the manifest doesn't list the sub-folder.
- `nn.Identity` inherits from `nn.Module` but does **not** expose a `.device`
  property — only `transformers.CLIPTextModel` and friends do — so the inference
  path that reads `self.text_encoder.device` crashes with `AttributeError`.

Diagnostic signal: the VRAM log `Text encoders loaded to GPU: allocated=0.02GB`
is far too small for real SDXL CLIP encoders (~0.7 GB), confirming the
placeholder substitution.

## Fix

Two-layer defence, landing in v1.2.9.

### 1. Fail fast at load time (`nodes.py`)

New helper `_assert_text_encoder_loaded` is invoked immediately after
`from_pretrained` in both `SeeThrough_LoadLayerDiffModel.load_model` and
`SeeThrough_LoadDepthModel.load_model`. It raises `RuntimeError` with a clear,
actionable message when the text encoder has no parameters — covering
`nn.Identity` and any future placeholder type.

### 2. Defensive device lookup (pipeline files)

- `see-through/common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py::encode_cropped_prompt_77tokens`
- `see-through/common/modules/marigold/marigold_depth_pipeline.py::MarigoldDepthPipeline.encode_empty_text`

Both now use `next(self.text_encoder.parameters()).device` instead of
`self.text_encoder.device`. On a real CLIP encoder the returned device is
identical; on an empty placeholder the helper raises `RuntimeError` pointing to
this issue instead of the cryptic `AttributeError`.

### 3. Regression test

`tests/test_text_encoder_check.py` covers the helper against:

- real `nn.Module` with parameters (must pass)
- `nn.Identity` placeholder (must raise)
- arbitrary empty custom `nn.Module` (must raise — forward-compat)
- `None` (must raise)

## Backwards compatibility

Both layers are backwards compatible:

- `next(parameters()).device` returns the same device as `.device` on real
  `CLIPTextModel` instances — no behavioural change for working installs.
- The load-time assert only fires when the text encoder was already broken
  (would have crashed during inference anyway).

## User-facing guidance

If users encounter the new `RuntimeError`, the message instructs them to:

1. Re-download the model checkpoint (ensuring `text_encoder/` and
   `text_encoder_2/` sub-folders are present and referenced in
   `model_index.json`), or
2. Downgrade `diffusers` to a version compatible with their ComfyUI build.
