# Upstream parity and facial layers

The recommended baseline is [tk_seethrough_upstream_parity.json](../workflows/tk_seethrough_upstream_parity.json). It uses the original Generate Layers node, enables the full V3 body/head pipeline, and preserves the existing depth, post-process, PSD, and optional Spine outputs. The existing Custom workflow also remains available.

## What is preserved from See-through

- LayerDiff model: `layerdifforg/seethroughv0.0.2_layerdiff3d` (V3).
- Marigold model: `layerdifforg/seethroughv0.0.1_marigold`.
- BF16 model weights; LayerDiff strength 1.0, CFG 1.0, 30 steps, 1280 square inference.
- Original RGBA source, center-square padding, original-resolution head crop, 1280 head inference, and projection back to the same canvas.
- Original body/head tag lists and their order. Both Generate Layers nodes now share one implementation of that existing upstream algorithm.
- Marigold's fixed 19-part frame layout plus the source frame, with the V3 eye/hair composites in the original slots.
- Depth inference at 768, resized back to the layer canvas, using model-default denoising steps.
- Upstream's 8-bit depth PNG precision before calculating part medians, plus its tag iteration order for equal-depth ties.

Reference: [upstream apply_layerdiff/apply_marigold/further_extr](https://github.com/shitagaki-lab/see-through/blob/df019de5129d6c4b406587a14c3501669441a783/common/utils/inference_utils.py).
The depth resolution was corrected by upstream in [90a77a4](https://github.com/shitagaki-lab/see-through/commit/90a77a4e5394b9cbd3c6216c289fa0e9c1288364).

The model architecture and diffusion sampler are unchanged. This is not a new segmentation model.

## Facial outputs and compatibility fixes

V3 produces these 11 head-stage layers: `headwear`, `face`, `irides`, `eyebrow`, `eyewhite`, `eyelash`, `eyewear`, `ears`, `earwear`, `nose`, and `mouth`. Empty layers are still omitted during post-processing; not every character has every accessory.

- Connect the loader's **MASK** output to Generate Layers' optional **mask** input. ComfyUI MASK uses 1 for transparent pixels. Without that connection, a three-channel IMAGE has no recoverable original alpha.
- V2 does not have V3's independent iris/eyewhite/eyelash tags. Use the V3 checkpoint above and confirm the log reports `tag_version=v3`.
- With the Custom node, keep `enable_head_detail=true` for facial decomposition.
- When head detail is deliberately disabled, the generated `head` now receives depth through Marigold's existing face slot and survives post-processing.
- Extend the existing face-order guard to V3 facial tags and their actual `-l`/`-r` names. Split ears use those same names when placed behind the face. Other parts retain the depth-based order.
- Layer Rename understands the hyphenated split names, so a subsequent include filter no longer drops the independent eye features. Rename/Filter preserve the source metadata used for the PSD base layer.
- The browser writer receives `Original` first (bottom), followed by `Parts` or `Runs`. Individual part order remains back-to-front. The embedded normal/depth PSD composite agrees with the initially visible Original layer.
- A separately connected `original_image` is square-padded like the generator, instead of being stretched. Connect `original_mask` to preserve its transparency.

All new inputs are optional; existing workflows can still load. Unconnected mask inputs retain the historical opaque-RGB behavior. To reproduce legacy depth resolution, set `resolution_depth=-1`; the new default is 768.

## Compare with upstream

Use the same source file, actual checkpoint revision/weights, GPU, and package versions on both sides. Matching a repository name or numeric seed alone does not establish identical inputs or random-number state.

Baseline ComfyUI settings:

| Parameter | Value |
| --- | --- |
| LayerDiff seed/control | 42 / fixed |
| Layer resolution / steps | 1280 / 30 |
| Head details | Full V3 (original node), or true (Custom) |
| Auto-fill | false |
| Depth seed/control | 42 / fixed |
| Depth resolution / steps | 768 / -1 (model default) |
| Left/right split | true |
| Source MASK | connected |

Comparable upstream command:

```sh
python inference/scripts/inference_psd.py \
  --srcp character.png \
  --repo_id_layerdiff layerdifforg/seethroughv0.0.2_layerdiff3d \
  --repo_id_depth layerdifforg/seethroughv0.0.1_marigold \
  --seed 42 --resolution 1280 --resolution_depth 768 \
  --inference_steps 30 --inference_steps_depth -1 \
  --tblr_split --save_to_psd
```

Compare raw per-tag RGBA layers first, then depth maps, then the PSD with `Original` hidden and `Parts` visible. Check independent eyes/brows/nose/mouth at full size, their alpha edges, and alignment against the source. This separates generation problems from export visibility/order problems.

Auto-fill remains an optional experimental heuristic: it compares RGB with the visible source and can mix parts from different seeds. Hidden-region completion can legitimately differ from the source, and small facial parts can fall below its coverage threshold. It is disabled in both supplied comparison baselines; this change does not claim to solve semantic auto-fill scoring.

## Verification scope

The regression suite compares the public nodes' V3 crop/reprojection and all 24 raw outputs against the bundled upstream `apply_layerdiff`, using identical prescribed model outputs. It also verifies RGBA/palette transparency, 768/-1 depth inputs and depth precision, facial layer preservation, split-name Rename/Filter, body-only head retention, workflow socket links, and native PSD order/composites using the actual bundled browser writer.

These tests run without diffusion checkpoints. A same-image GPU A/B is still required to establish perceptual quality or runtime/memory parity for a specific user image.

