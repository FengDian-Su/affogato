# Molmo Prompt Format — Bimanual Affordance Data Generation

設計 Molmo (`allenai/MolmoPoint-8B`) query 的 reference，用於生成 role-conditioned bimanual affordance heatmap。

---

## 1. Molmo I/O 規格

- **Model:** `allenai/MolmoPoint-8B`（Qwen3-8B 為 base）
- **輸入:** image + 一段 text query
- **輸出:** `<points coords="frame_id idx x y ..."/>`
  - 座標是 0–1000 的千分比，要再 scale 回影像尺寸
  - 一次回應可包含多個點（`idx` 列舉），但這些點都對應**同一個 query**，沒有 role label
- **Implication:** role-conditioned bimanual data **每個 role 跑一次推論**。不要把多個 role 塞到同一句 query — 解析回來分不清誰是誰。

---

## 2. Prompt 模板

每個 role 用下列其中一個模板。**優先順序視情境而定**：

- **單 role / single-region 任務** → 優先 A (action-verb)，落空再退 B → C
- **多 role / role-conditioned 並行任務** → 優先 **B (part-name)**，落空再退 A → C
  - 多個 role 各自獨立跑 Molmo 時，verb-based prompt 兩個 query 的 attention 容易塌到同一個顯著部位（例：playing machine 上 "push to move" 跟 "press to perform action" 都被吸到同一個按鈕）。Part-name 把區分壓力丟到 Molmo 最強的「定位特定 part」這件事，更穩。

### Template A — Action verb（首選）

```
Point to where you would <verb> to <functional outcome>.
```

範例：
- `Point to where you would push to move the character.`
- `Point to where you would press to perform an action.`
- `Point to where you would grip to hold the bottle steady.`
- `Point to where you would twist to open the cap.`

**為什麼**：動詞錨定操作意圖，比單純 part name 更能聚焦在 functional region。實測對複雜物件（例如 arcade machine 的搖桿 vs 按鈕）較不混淆。

### Template B — Part name（fallback）

```
Point to the <part name> on this <object>.
```

範例：
- `Point to the joystick on this arcade machine.`
- `Point to the action button on this game machine.`
- `Point to the cap of this bottle.`

**何時用**：動詞太抽象、或 Template A 在多個 view 上回傳 0 點時。

### Template C — Functional region（fallback）

```
Point to the part of this object you would use with one hand to <function>.
```

範例：
- `Point to the part of this object you would use with one hand to control character movement.`

**何時用**：A、B 都失敗。語意最寬鬆，但點的精確度也最低。

---

## 3. Role decomposition recipe

Pipeline 上游用 LLM（目前是 [bimanual_annotation/get_affordance.py](../bimanual_annotation/get_affordance.py) 的 Gemma）把 high-level user query 拆成多個 role。建議 schema：

```json
{
  "user_query": "How do I play a game on this machine?",
  "roles": {
    "role_move": {
      "molmo_prompt": "Point to where you would push to move the character.",
      "fallback_prompts": [
        "Point to the joystick on this arcade machine."
      ]
    },
    "role_action": {
      "molmo_prompt": "Point to where you would press to perform an action.",
      "fallback_prompts": [
        "Point to the action button on this game machine."
      ]
    }
  }
}
```

**Role 命名原則**：用功能性名稱（`support`, `actuate`, `move`, `action`, `grip`...），**不要**用 `left_hand`/`right_hand` — spec 是 hand-agnostic。

---

## 4. Tested example sets

### Playing machine (arcade-style, joystick + button)

**首選 (Template B, part-name)** — 兩 role 部位在空間上接近，verb-based 兩個 query 會塌到同一個按鈕：
- `role_move`: `Point to the joystick on this arcade machine.`
- `role_action`: `Point to the action button on this arcade machine.`

Action-verb 版本（不推薦但留作對照）：
- `role_move`: `Point to where you would push to move the character.`
- `role_action`: `Point to where you would press to perform an action.`

### Bottle (twist-open)
- `role_support`: `Point to where you would grip to hold the bottle steady.`
- `role_actuate`: `Point to where you would twist to open the cap.`

### Drawer
- `role_support`: `Point to where you would hold the drawer body steady.`
- `role_actuate`: `Point to where you would pull to open the drawer.`

### Mug (pouring)
- `role_support`: `Point to where you would grip the handle.`
- `role_actuate`: `Point to where you would tilt the rim to pour.`

> 隨著測試新物件持續補充。每筆條目應該有實際跑過 Molmo 並 visually 驗證過。

---

## 5. Runtime fallback strategy

對每個 (view, role)，如果 primary prompt 回傳 0 點：

1. 改用 Template B（part name）再 query 一次
2. 還是 0 點 → 改用 Template C（functional region）
3. 仍然 0 點 → 標記這個 view 對該 role 為 missing。heatmap 用全 0，3D voting 時這個 view 不貢獻給該 role。

**追蹤每筆 dataset 的 fallback 使用率** — 高 fallback rate 通常代表 primary prompt 設計不好。

---

## 6. Open questions / TODO

- 哪些 role 用 part-name (B) 系統性地比 action-verb (A) 好？
- 3+ roles（例如 trimanual 或多步驟任務）時，每個 role 一次 Molmo call 是否還夠用？或需要 step-level 分解？
- 動詞選擇（push vs press vs tap）對 Molmo 的敏感度？需要做 ablation。
- 是否要在 prompt 中明示 "with one hand"（強調是單手可達區域）？實測對某些大型物件（例如 drawer 的整個 body）有用。
