# 需求：ComfyUI-See-through 自訂圖層選擇節點

## 背景

ComfyUI-See-through 外掛（https://github.com/jtydhr88/ComfyUI-See-through）可以將一張動漫插圖拆成多個語義透明圖層。目前 `SeeThrough Generate Layers` 節點的 prompt tag 列表是硬編碼的，使用者無法選擇只生成自己想要的圖層。

## 目標

基於現有的 `SeeThrough_GenerateLayers` 節點，新增一個自訂版節點 `SeeThrough_GenerateLayers_Custom`，將原本寫死的語義 tag 改為 ComfyUI 節點介面上的 **布林勾選框**，使用者打勾的 tag 才會送入擴散模型生成圖層，不打勾的就跳過。

## 技術細節

### 1. 節點基本資訊

- **節點名稱：** `SeeThrough_GenerateLayers_Custom`
- **顯示名稱：** `SeeThrough Generate Layers (Custom)`
- **分類：** `SeeThrough`

### 2. 需要支援的 tag 勾選框（共 24 個）

節點需要根據模型的 `tag_version` 對應不同的 tag 集合：

#### v2 模式（19 個 tag）

```
hair, headwear, face, eyes, eyewear, ears, earwear,
nose, mouth, neck, neckwear, topwear, handwear,
bottomwear, legwear, footwear, tail, wings, objects
```

#### v3 模式 — body 階段（13 個 tag，group_index=0）

```
front hair, back hair, head, neck, neckwear,
topwear, handwear, bottomwear, legwear, footwear,
tail, wings, objects
```

#### v3 模式 — head 階段（11 個 tag，group_index=1）

```
headwear, face, irides, eyebrow, eyewhite,
eyelash, eyewear, ears, earwear, nose, mouth
```

每個 tag 在 `INPUT_TYPES` 中以 `BOOLEAN` 型別呈現，預設全部為 `True`。參數命名將空格替換為底線（例如 `front hair` → `front_hair`），tooltip 顯示原始 tag 名稱。

### 3. 運作邏輯

- 讀取所有布林參數，收集使用者勾選為 `True` 的 tag 組成列表
- `tag_version` 在 runtime 才能從 `pipeline.unet.get_tag_version()` 取得，所以 **節點 UI 上先全部列出所有 24 個 tag 的勾選框**，執行時再根據實際 `tag_version` 過濾出有效的 tag
- 如果使用者勾選了一個在當前 `tag_version` 下不存在的 tag，靜默忽略即可（不報錯）
- v3 模式下，如果 body 階段的 `head` tag 沒被勾選，則跳過整個 head 細節生成階段（因為沒有 head 就無法裁切頭部區域）
- 其餘推理邏輯（`center_square_pad_resize`、pipeline 呼叫、v3 頭部二階段裁切等）保持與原始 `SeeThrough_GenerateLayers` 完全一致

### 4. 輸出

與原節點一致，輸出 `SEETHROUGH_LAYERS` 和 `IMAGE`（preview），下游的 `GenerateDepth`、`PostProcess`、`SavePSD` 無需修改，直接相容。

### 5. 程式碼位置

在原本的 `nodes.py` 中新增這個 class，並在 `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS` 中註冊，**不要修改或刪除原有的 `SeeThrough_GenerateLayers` 節點**。

## 注意事項

- 至少要勾選 1 個 tag，如果一個都沒勾選就拋出明確的錯誤訊息提示使用者
- 不要發明新 tag 或修改可用的 tag 列表，嚴格使用上述原始碼中已定義的 tag
- 原有的 `seed`、`resolution`、`num_inference_steps` 參數保留不變
