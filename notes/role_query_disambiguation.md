# Disambiguating Role Queries in Bimanual Affordance Generation

## Context

`data_generation/point_cloud_from_depth.ipynb` 目前對每個 role 跑一次 Molmo，query 用 action-verb 形式：

```python
ROLE_QUERIES = {
    "role_move":   "Point to where you would push to move the character.",
    "role_action": "Point to where you would press to perform an action.",
}
```

實測 playing machine 上兩個 query 點到**接近 / 同一個區域**，role assignment 失效 — 兩張 heatmap 在 3D 上重疊，bimanual 分解就破功。

## Decision

採用 **Approach A: part-name query**（已與 user 確認）。

根因是 verb-based 講得不夠具體，Molmo 兩個 query 的 attention 落在同一個顯著部位。改用 part name 把區分壓力丟到 Molmo 最擅長的「定位特定 part」這件事 — 對應 `prompt/molmo_prompt_format.md` 的 Template B。

不加 distance safety net；目視 cell 27 的 combined grid 就足以發現問題。

不採 user 提案：
- 提案 #1（明示 left/right）違反 hand-agnostic spec、3D rotated view 會反鏡像
- 提案 #2（overlay context）Molmo 沒被訓練看 annotated overlay、ordering bias、2× inference

## Changes

### 1. `data_generation/point_cloud_from_depth.ipynb` cell 25

把 ROLE_QUERIES 從 verb-based 改為 part-name based（針對當前 playing machine asset）：

```python
ROLE_QUERIES = {
    "role_move":   "Point to the joystick on this arcade machine.",
    "role_action": "Point to the action button on this arcade machine.",
}
```

註解保留其他 asset 的 verb-based 範例（bottle / drawer / mug）作為參考；之後對其他 asset 也建議優先用 part-name，verb-based 當 fallback。

### 2. `prompt/molmo_prompt_format.md`

**§ 2（Template 排序）**：把目前的 A → B → C 優先順序改成「視情境而定 — 多 role 並行時 B 優於 A」並補一句解釋。

**§ 4 (Tested example sets)**：
- 補一筆「Playing machine (arcade-style, joystick + button)」的 part-name 版本
- 加一行 note：「兩 role 部位在空間上接近時（例：playing machine 的 joystick 與 button），實測 Template B (part name) 比 Template A (action verb) 穩 — verb-based prompt 兩個 query attention 容易塌到同一個顯著部位」

## Files to modify

- `data_generation/point_cloud_from_depth.ipynb` cell 25
- `prompt/molmo_prompt_format.md` §§ 2 + 4

## Verification

1. **不需 restart kernel** — Molmo 仍在記憶體
2. 重 run cell 25（更新 ROLE_QUERIES）→ cell 26（per-role inference，~80s）→ cell 27（combined grid）
3. **目視 check**：cell 27 的 grid 上兩個顏色星星應落在**不同**區域；> 80% view 應如此
4. 接下來的 cell（SAM2 / 3D voting）不用改。最終 cell 39 的 3D plot 兩個 role 高分區應分離
5. 若仍有相當比例 view 兩點重疊，下一輪可再評估 Approach E (post-filter retry) 或調整 part name 用詞（例：把 "action button" 改成 "red button" 或更獨特的描述）

## Out of scope（之後再做）

- 把 part name 接到 `bimanual_annotation/get_affordance.py` 的 `affordance_parts` 自動產生 ROLE_QUERIES — 這是 batch 跑很多 asset 時的 scaling step
- distance-based post-filter / retry 機制
