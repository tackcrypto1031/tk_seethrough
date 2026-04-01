# ComfyUI-See-through (Custom Fork)

![Preview](https://raw.githubusercontent.com/tackcrypto1031/tk_seethrough/main/workflows/img_1.png)

A fork of [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through) by [@jtydhr88](https://github.com/jtydhr88), adding a custom node with the option to skip the head detail inference stage for faster processing.

[中文說明](README_ZH.md)

## What's New in This Fork

### SeeThrough Generate Layers (Custom)

A new node `SeeThrough_GenerateLayers_Custom` that adds one parameter compared to the original `SeeThrough Generate Layers`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_head_detail` | true | v3 models only: toggle the head detail inference stage on/off |

#### How It Works

The v3 See-through model runs in **two inference stages**:

1. **Body stage** — Generates 13 body-level layers (front hair, back hair, head, neck, neckwear, topwear, handwear, bottomwear, legwear, footwear, tail, wings, objects)
2. **Head stage** — Crops the head region from stage 1, upscales it, and runs a second inference pass to generate 11 fine-grained head layers (headwear, face, irides, eyebrow, eyewhite, eyelash, eyewear, ears, earwear, nose, mouth)

Each stage is a full diffusion pipeline call. By setting `enable_head_detail = false`, the entire head stage is **skipped** (no GPU computation), saving approximately **50% of the total inference time**.

This is useful when you only need body-level decomposition and don't require fine-grained facial features.

> **Note:** For v2 models, this toggle has no effect since v2 uses a single-stage inference.

## All Nodes

| Node | Description |
|------|-------------|
| **SeeThrough Load LayerDiff Model** | Load the LayerDiff SDXL pipeline |
| **SeeThrough Load Depth Model** | Load the Marigold depth estimation pipeline |
| **SeeThrough Generate Layers** | Original layer generation (all stages, all layers) |
| **SeeThrough Generate Layers (Custom)** | Layer generation with `enable_head_detail` toggle |
| **SeeThrough Generate Depth** | Depth map estimation per layer |
| **SeeThrough Post Process** | Left/right splitting, hair clustering, color restoration |
| **SeeThrough Save PSD** | Export layers as PNGs + metadata; download PSD via browser |

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tackcrypto1031/tk_seethrough.git
```

Install dependencies:

```bash
cd tk_seethrough
pip install -r requirements.txt
```

Restart ComfyUI. The nodes will appear under the `SeeThrough` category.

### Models

Models are downloaded automatically from HuggingFace on first use:

| Model | HuggingFace Repo | Purpose |
|-------|-------------------|---------|
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | SDXL-based transparent layer generation |
| Marigold Depth | `24yearsold/seethroughv0.0.1_marigold` | Fine-tuned monocular depth for anime |

You can also download models manually and place them in `ComfyUI/models/SeeThrough/`.

## Usage

1. Add **SeeThrough Load LayerDiff Model** and **SeeThrough Load Depth Model**
2. Add **SeeThrough Generate Layers (Custom)** — connect both models and a **Load Image** node
3. Uncheck `enable_head_detail` if you want faster processing without head detail layers
4. Connect to **SeeThrough Generate Depth** → **SeeThrough Post Process** → **SeeThrough Save PSD**
5. Run the workflow and click **Download PSD** to export

## Acknowledgements

This project is a fork of [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through) by [@jtydhr88](https://github.com/jtydhr88). Huge thanks for creating the original ComfyUI integration.

The underlying research is [See-through](https://github.com/shitagaki-lab/see-through) by [shitagaki-lab](https://github.com/shitagaki-lab).
Paper: [arxiv:2602.03749](https://arxiv.org/abs/2602.03749) (Conditionally accepted to ACM SIGGRAPH 2026)

PSD generation uses [ag-psd](https://github.com/nicasiomg/ag-psd) in the browser.

## License

MIT
