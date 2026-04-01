# ComfyUI-See-through（自訂 Fork）

![預覽](https://raw.githubusercontent.com/tackcrypto1031/tk_seethrough/main/workflows/img_1.png)

基於 [@jtydhr88](https://github.com/jtydhr88) 的 [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through) 修改，新增可跳過 head 細節推理階段的自訂節點，加速處理流程。

[English](README.md)

## 本 Fork 新增功能

### SeeThrough Generate Layers (Custom)

新增節點 `SeeThrough_GenerateLayers_Custom`，相比原始的 `SeeThrough Generate Layers` 多一個參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `enable_head_detail` | true | 僅 v3 模型：開關頭部細節推理階段 |

#### 運作原理

v3 See-through 模型分**兩個推理階段**執行：

1. **Body 階段** — 生成 13 個身體層級圖層（前髮、後髮、頭部、頸部、頸飾、上衣、手套、下裝、腿飾、鞋子、尾巴、翅膀、物件）
2. **Head 階段** — 從第一階段裁切頭部區域，放大至原始解析度，再進行第二次推理，生成 11 個精細頭部圖層（頭飾、臉部、虹膜、眉毛、眼白、睫毛、眼鏡、耳朵、耳飾、鼻子、嘴巴）

每個階段都是一次完整的擴散管線呼叫。將 `enable_head_detail` 設為 `false` 時，整個 head 階段會被**完全跳過**（不佔用 GPU 運算），節省大約 **50% 的推理時間**。

適用於只需要身體層級拆解、不需要精細面部特徵的場景。

> **注意：** v2 模型為單階段推理，此開關無效果。

## 所有節點

| 節點 | 說明 |
|------|------|
| **SeeThrough Load LayerDiff Model** | 載入 LayerDiff SDXL 管線 |
| **SeeThrough Load Depth Model** | 載入 Marigold 深度估計管線 |
| **SeeThrough Generate Layers** | 原始圖層生成（全部階段、全部圖層）|
| **SeeThrough Generate Layers (Custom)** | 圖層生成，附帶 `enable_head_detail` 開關 |
| **SeeThrough Generate Depth** | 逐圖層深度估計 |
| **SeeThrough Post Process** | 左右拆分、頭髮聚類、色彩還原 |
| **SeeThrough Save PSD** | 匯出圖層 PNG + 中繼資料；透過瀏覽器下載 PSD |

## 安裝

將此倉庫克隆到 ComfyUI 的 `custom_nodes` 目錄：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tackcrypto1031/tk_seethrough.git
```

安裝相依套件：

```bash
cd tk_seethrough
pip install -r requirements.txt
```

重啟 ComfyUI，節點會出現在 `SeeThrough` 分類下。

### 模型

首次使用時自動從 HuggingFace 下載：

| 模型 | HuggingFace 倉庫 | 用途 |
|------|-------------------|------|
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | 基於 SDXL 的透明圖層生成 |
| Marigold Depth | `24yearsold/seethroughv0.0.1_marigold` | 動漫微調的單目深度估計 |

也可手動下載模型放到 `ComfyUI/models/SeeThrough/` 目錄。

## 使用方法

1. 新增 **SeeThrough Load LayerDiff Model** 和 **SeeThrough Load Depth Model**
2. 新增 **SeeThrough Generate Layers (Custom)** — 連接兩個模型和 **Load Image** 節點
3. 取消勾選 `enable_head_detail` 可跳過頭部細節，加速處理
4. 連接至 **SeeThrough Generate Depth** → **SeeThrough Post Process** → **SeeThrough Save PSD**
5. 執行工作流，點擊 **Download PSD** 匯出

## 致謝

本專案是 [@jtydhr88](https://github.com/jtydhr88) 的 [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through) 的 Fork 版本。非常感謝原作者建立了 ComfyUI 整合。

底層研究為 [shitagaki-lab](https://github.com/shitagaki-lab) 的 [See-through](https://github.com/shitagaki-lab/see-through)。
論文：[arxiv:2602.03749](https://arxiv.org/abs/2602.03749)（已被 ACM SIGGRAPH 2026 條件接收）

PSD 生成使用瀏覽器端的 [ag-psd](https://github.com/nicasiomg/ag-psd) 套件。

## 授權

MIT
