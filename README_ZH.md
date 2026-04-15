# ComfyUI-See-through（自訂 Fork）

![預覽](https://raw.githubusercontent.com/tackcrypto1031/tk_seethrough/main/workflows/sample.png)

基於 [@jtydhr88](https://github.com/jtydhr88) 的 [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through) 修改，新增可跳過 head 細節推理階段的自訂節點，加速處理流程。

[English](README.md)

## 本 Fork 新增功能

### v1.2.8 — Issue #5

- 新增節點 **SeeThrough Load Source**:行為與 ComfyUI LoadImage 相同,額外輸出 `source_filename`,讓最終 PSD 保留原始檔名。
- **SeeThrough Save PSD** 新增 optional 輸入 `original_image` 與 `source_filename`,自動將原始輸入圖作為可見底圖放入 PSD,且輸出檔名會保留原檔名。
- PSD 圖層結構改為分組:`Original`(底層可見)/`Parts`(隱藏)/`Runs`(隱藏,僅 grouped PSD 模式),開啟 PSD 即見原圖,依需要展開 group 修局部。

### v1.2.3 — 問題修復

- **修復：PSD 下載失敗「Failed to load ag-psd bundle from any path」** — ComfyUI 新版前端以 ES module `import()` 載入擴充套件，導致 `document.currentScript` 為 `null`。改用 `import.meta.url` 可靠地解析 bundle 路徑，不受安裝資料夾名稱影響。（[#1](https://github.com/tackcrypto1031/tk_seethrough/issues/1)）
- **修復：全新環境下節點無法載入** — 上游 Marigold 模組在頂層匯入 `matplotlib`（僅用於可視化功能），導致未安裝 `matplotlib` 的環境出現 `ModuleNotFoundError`。改為延遲匯入，節點載入不再依賴 `matplotlib`。

### v1.2 — Spine 匯出、自動補圖 & 全輪次 PSD

- **Spine 匯出節點** — 新增 `Layer Rename`、`Layer Filter`、`Export Spine` 節點，用於 Spine 2D 動畫前置拆圖。
- **自動補齊缺失圖層** — 在 GenerateLayers (Custom) 開啟 `auto_fill`，自動重跑推理最多 5 次，補齊缺失圖層並透過原圖相似度比對升級品質差的圖層。
- **全輪次 PSD** — 開啟 `auto_fill` 後，Save PSD 新增「Download All Runs PSD」按鈕，將所有輪次以群組資料夾方式包進同一個 PSD，方便手動比較選擇。
- **PSD 下載按鈕** — Save PSD 現在有 3 個按鈕：
  - **Download PSD**（綠色）— auto-fill 篩選後的最佳圖層
  - **Download Depth PSD**（紫色）— 深度圖
  - **Download All Runs PSD**（橘色）— 所有輪次依群組分開（需開啟 `auto_fill`）

### SeeThrough Generate Layers (Custom)

新增節點 `SeeThrough_GenerateLayers_Custom`，相比原始的 `SeeThrough Generate Layers` 多一個參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `enable_head_detail` | true | 僅 v3 模型：開關頭部細節推理階段 |
| `auto_fill` | false | 自動補圖：開啟後自動重跑推理（最多 5 次）直到所有預期圖層生成完畢（v3+head=24, v3 body=13, v2=19）|
| `min_alpha_coverage` | 0.01 | 圖層有效的最小 alpha 覆蓋率。僅在 `auto_fill` 開啟時使用 |

#### 運作原理

v3 See-through 模型分**兩個推理階段**執行：

1. **Body 階段** — 生成 13 個身體層級圖層（前髮、後髮、頭部、頸部、頸飾、上衣、手套、下裝、腿飾、鞋子、尾巴、翅膀、物件）
2. **Head 階段** — 從第一階段裁切頭部區域，放大至原始解析度，再進行第二次推理，生成 11 個精細頭部圖層（頭飾、臉部、虹膜、眉毛、眼白、睫毛、眼鏡、耳朵、耳飾、鼻子、嘴巴）

每個階段都是一次完整的擴散管線呼叫。將 `enable_head_detail` 設為 `false` 時，整個 head 階段會被**完全跳過**（不佔用 GPU 運算），節省大約 **50% 的推理時間**。

適用於只需要身體層級拆解、不需要精細面部特徵的場景。

#### 多輪推理自動補圖

擴散模型具有隨機性 — 每次推理可能產生不同結果。有時某個圖層（如臉部或手部）在一次推理中缺失，但在另一次中卻正常生成。

開啟 `auto_fill` 可自動重跑推理，直到所有預期圖層都以良好品質生成：

1. **第 1 輪** 使用原始 seed — 作為主要結果
2. 每個圖層會與原圖進行比對，計算**相似度分數**（0~1）
3. **缺失**的圖層（alpha 覆蓋率低於閾值）或**相似度低**（< 0.85）的圖層會觸發後續輪次
4. **第 2 輪** 使用 `seed + 1`，**第 3 輪** 使用 `seed + 2`，依此類推
5. 每個圖層自動保留**與原圖相似度最高**的版本
6. 最多重跑 **5 輪**，或所有圖層達到良好相似度即停止

這表示即使第 1 輪已生成臉部圖層，若第 2 輪產生的臉部與原圖更相似，會自動使用第 2 輪的版本。

模型在所有輪次中只載入 GPU 一次 — 額外開銷僅為擴散推理時間，不包含模型載入。

> **注意：** v2 模型為單階段推理，此開關無效果。

### v1.1 — 同步上游更新

本 Fork 已同步 [上游 v0.2.2](https://github.com/jtydhr88/ComfyUI-See-through) 的改進：

- **VRAM 卸載** — 模型載入後留在 CPU，推理時才搬到 GPU，推理完立刻搬回 CPU。大幅降低顯存佔用，讓 8GB 以下的顯卡也能順利運行。
- **Text Encoder 卸載** — Text Encoder 在 GPU 上完成 prompt encoding 後立刻卸載，再載入 UNet+VAE 進行擴散，兩者不會同時佔用顯存。
- **Marigold 相容性修復** — 修復 torchvision >= 0.23 的 `InterpolationMode` 嚴格檢查導致的 resize 錯誤。
- **Web 修復** — 支援子路徑部署；修復大小寫敏感檔案系統上 ag-psd bundle 404 問題。
- **自訂節點優化** — 當 `enable_head_detail = false` 時，連 head 的 text encoding 也一併跳過（不只跳過擴散），進一步降低 GPU 記憶體使用。

## 所有節點

| 節點 | 說明 |
|------|------|
| **SeeThrough Load LayerDiff Model** | 載入 LayerDiff SDXL 管線 |
| **SeeThrough Load Depth Model** | 載入 Marigold 深度估計管線 |
| **SeeThrough Generate Layers** | 原始圖層生成（全部階段、全部圖層）|
| **SeeThrough Generate Layers (Custom)** | 圖層生成，附帶 `enable_head_detail` 開關 |
| **SeeThrough Generate Depth** | 逐圖層深度估計 |
| **SeeThrough Post Process** | 左右拆分、頭髮聚類、色彩還原 |
| **SeeThrough Save PSD** | 匯出圖層 PNG + 中繼資料；可下載最佳 PSD、深度 PSD 或全輪次 PSD |
| **SeeThrough Layer Rename** | 將圖層標籤重新命名為 Spine 友善名稱（可自訂）|
| **SeeThrough Layer Filter** | 依標籤名稱包含/排除特定圖層 |
| **SeeThrough Export Spine** | 匯出為 Spine 2D 骨架專案（JSON + 圖片）|

### Spine 匯出工作流

用於 [Spine](http://esotericsoftware.com/) 動畫前置拆圖，連接方式：

```
Post Process → Layer Rename（可選）→ Layer Filter（可選）→ Export Spine
```

#### Layer Rename（圖層重命名）

將內部標籤映射為 Spine 友善名稱。內建預設涵蓋所有標籤。`custom_mapping_json` 欄位為**可選** — 留空即使用預設值。

**使用時機：**
- 你希望在 Spine 中看到乾淨易讀的名稱（如 `front-hair` 而非 `hairf`）
- 你的團隊有命名規範，需要自訂名稱

**內建預設映射（部分列表）：**

| 原始標籤 | → 重命名為 |
|----------|-----------|
| `hairf` | `front-hair` |
| `hairb` | `back-hair` |
| `eyel` | `eye-left` |
| `eyer` | `eye-right` |
| `handwearl` | `handwear-left` |
| `handwearr` | `handwear-right` |
| `earl` | `ear-left` |
| `earr` | `ear-right` |
| `topwear` | `topwear`（不變）|
| `face` | `face`（不變）|

> 已經是乾淨名稱的標籤（如 `face`、`head`、`nose`）會保持原樣。

**自訂映射範例：** 在 `custom_mapping_json` 中輸入 JSON 物件來覆蓋特定名稱：

```json
{
  "hairf": "bangs",
  "hairb": "back-hair",
  "topwear": "shirt",
  "bottomwear": "skirt",
  "handwearl": "left-glove",
  "handwearr": "right-glove"
}
```

只有你在 JSON 中指定的標籤會被覆蓋 — 其餘標籤仍使用內建預設值。JSON 格式錯誤時會忽略並顯示警告。

#### Layer Filter（圖層篩選）

移除不需要的圖層，支援 include 或 exclude 模式。預設已填入所有可用標籤 — 刪除不需要的即可。每行一個標籤名。

> **提示：** 如果 Layer Rename 接在 Layer Filter 前面，請使用**重命名後**的標籤名（如 `front-hair`）。若未使用 Layer Rename，請用原始標籤（如 `hairf`）。

#### Export Spine（Spine 匯出）

輸出一個資料夾，可自訂匯出路徑（預設為 ComfyUI output 資料夾），包含：

- `{prefix}.json` — Spine 骨架檔（可直接在 Spine 編輯器中開啟）
- `images/` — 各圖層裁切後的 PNG 檔案
- 設定 `output_path` 可匯出到指定目錄（如 `D:/my_project/spine_assets`）

座標自動從圖片空間（Y 向下）轉換為 Spine 空間（Y 向上，原點在畫布底部中央）。繪製順序依照 Post Process 的深度排序。

#### PSD 匯入 vs JSON 匯出 — 該用哪個？

Spine Professional（3.6+）可以直接匯入 PSD 檔案，因此你可能會疑惑是否還需要 JSON 匯出。以下是比較：

| | Save PSD → Spine PSD 匯入 | Export Spine（JSON + 圖片）|
|---|---|---|
| **Spine 版本** | 僅 Professional 3.6+ | **所有版本**（Essential + Professional）|
| **圖層定位** | 自動 | 自動（座標已預轉換）|
| **圖層命名** | 取決於 PSD 圖層名稱 | 可透過 LayerRename 控制 |
| **圖層篩選** | 需先在 PSD 中隱藏/刪除 | 內建 LayerFilter 節點 |
| **迭代更新** | 重新匯入 PSD 更新圖片 | 重新匯出更新 |
| **骨架層級** | 不會自動建立 | 不會自動建立 |
| **適合** | Spine Professional 用戶，快速開始 | Spine Essential 用戶，或需要自動化前處理的團隊 |

**建議：**
- **Spine Professional 用戶** → 使用 **Save PSD** 後透過 Spine 內建的 PSD 匯入功能。這是最簡單的工作流程。
- **Spine Essential 用戶** → 使用 **Export Spine**，因為 Essential 版本不支援 PSD 匯入。
- **自動化工作流程** → 使用 **Export Spine** 搭配 LayerRename + LayerFilter，取得一致且預處理過的輸出。

<details>
<summary>可用圖層標籤（經 LayerRename 後，共 38 個）</summary>

| 分類 | 標籤 |
|------|------|
| 頭髮 | `front-hair`, `back-hair` |
| 頭部 | `head`, `headwear` |
| 臉部 | `face`, `nose`, `mouth` |
| 眼睛 | `eye-left`, `eye-right`, `eyewear` |
| 眼部細節 | `irides`, `irides-left`, `irides-right`, `eyebrow`, `eyebrow-left`, `eyebrow-right`, `eye-white`, `eye-white-left`, `eye-white-right`, `eyelash`, `eyelash-left`, `eyelash-right` |
| 耳朵 | `ears`, `ear-left`, `ear-right`, `earwear` |
| 身體 | `neck`, `neckwear`, `topwear`, `bottomwear` |
| 四肢 | `handwear`, `handwear-left`, `handwear-right`, `legwear`, `footwear` |
| 其他 | `tail`, `wings`, `objects` |

若未使用 LayerRename，請用原始標籤：`hairf`、`hairb`、`eyel`、`eyer`、`handwearl`、`handwearr`、`earl`、`earr` 等。

</details>

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

**Spine 匯出：** 將步驟 4 的 **Save PSD** 替換為 **Layer Rename** → **Layer Filter** → **Export Spine**。在 Spine 編輯器中開啟輸出的 JSON 檔案即可。

## 致謝

本專案是 [@jtydhr88](https://github.com/jtydhr88) 的 [ComfyUI-See-through](https://github.com/jtydhr88/ComfyUI-See-through) 的 Fork 版本。非常感謝原作者建立了 ComfyUI 整合。

底層研究為 [shitagaki-lab](https://github.com/shitagaki-lab) 的 [See-through](https://github.com/shitagaki-lab/see-through)。
論文：[arxiv:2602.03749](https://arxiv.org/abs/2602.03749)（已被 ACM SIGGRAPH 2026 條件接收）

PSD 生成使用瀏覽器端的 [ag-psd](https://github.com/nicasiomg/ag-psd) 套件。

## 授權

MIT
