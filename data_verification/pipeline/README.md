# Data Verification Pipeline — 規格 v2.0

依據《Dual Affordance Dataset 資料品質評估計畫書（新版）》（2026-07-27）＋ 同日定案的本地決策。
**本文件是實作層規格；與計畫書衝突時以本文件的「本地決策（L*）」為準，其餘一律以計畫書為準。**

舊版（v2 計畫書 + D1–D14 spec）實作已歸檔於 `data_verification/0723_prev/pipeline/`，
其中被新計畫書推翻的決定見 §13，**不得沿用**。

---

## 0. 本地決策摘要（相對計畫書的差異）

| # | 決策 | 影響 |
|---|---|---|
| **L1** | **完全省略 geometry 審查**（節省時間與算力） | 不渲染 geometry 六視圖、不跑 object-level geometry usability pass、**`geometry_usable` 這個欄位整個不存在**（不進 manifest、不進 gating、不進人工評分、不進論文表）。詳見 §6.1 |
| **L2** | **硬體只有 1×RTX 3090 (24GB)** | Pilot 用 **`Qwen3-VL-8B-Instruct`（bf16 可直上，免量化）**，Stage 1/Stage 2 共用；等大卡空出再換 30B-A3B 重跑。詳見 §11 |
| **L3** | 資料規模 280,064 pairs → **完整資料 VLM 審核不可行**，Stage 1/2 改為分層抽樣審核 | 待確認抽樣規模。詳見 §12 |

---

## 1. 資料與路徑

### 1.1 正式輸入

```
data_verification/data/
└── stage1/{daily_used,electronics}/stage1_all.json    # ★ 正式全量 stage1 結果
                                                        #   md5 與 data_generation/outputs/stage1/*/ 完全相同

data_generation/outputs/
└── stage2/{daily_used,electronics}/<object_id>/<qN_task_slug>/
                                     ├── scores.npz     # 含 xyz（= point cloud）
                                     └── meta.json      # 無獨立 point cloud 檔
```

規模（2026-07-27 實測）：

| dataset | objects | queries | stage2 已完成 objects |
|---|---:|---:|---:|
| daily_used | 57,441 | 251,390 | 21,793（仍在跑） |
| electronics | 5,528 | 28,674 | 1 |
| **合計** | **62,969** | **280,064** | — |

> Verification Stage 0/2 只能對 stage2 已完成的 object 執行 → **必須維護 coverage 對帳表並可重跑**。
> Stage 1 不受影響（stage1 已全量完成）。

### 1.2 Pipeline 測試資料（**只供 plumbing，不可用於凍結任何門檻**）

`data_verification/data/stage2_example_data/` — 1,000 objects，**取自 2026-07-16/17 的舊 stage2 批次**。
與正式資料有兩處實質差異：

| | example（舊） | 正式（新 run） |
|---|---|---|
| npz keys | `xyz, gt, counts, scoreA_raw, scoreB_raw, scoreA, scoreB, ptsA, ptsB, molmo_queries, task, partition_axis, sel_version` | `xyz, scoreA_raw, scoreB_raw, scoreA, scoreB, ptsA, ptsB, ptsA_multi, ptsB_multi, molmo_queries, task, pexistA, pexistB` |
| score 分布 | 抽樣 `scoreA.max() ≈ 0.996` | 抽樣 `scoreA.max() ≈ 0.833` |

⚠ **後果**：(1) Stage 0 的 schema 驗證不得硬性要求 `gt`/`counts`/`sel_version`/`partition_axis`，
需把 `ptsA_multi`/`ptsB_multi`/`pexistA`/`pexistB` 列為已知 optional key；
(2) **τ_vis 絕對不能在 example data 上凍結**，score scale 不同 → Phase 2 pilot 必須從
**新 run 已完成的 21,793 objects** 抽樣。1,000 objects 中僅 692 個在新 run 也存在。

### Stage 1 JSON schema（實測）

object 層：`object_id`, `object_name`, `views_used`, `components`, `queries`
query 層：`task`, `query`, `goal`, `why_bimanual`, `category`, `mechanism`, `operation`,
`coordination`, `roles`, `pattern`, `relation`, `symmetric`, `hand_agnostic`,
`molmo_queries`, `answer`

- `views_used` = **固定 8 張** G-Objaverse RGB 絕對路徑（`/nfs_drive/gobjaverse/...`），實測 3,000 objects 皆為 8。
- `roles` = list of 2 dicts：`{id: "A"|"B", role, target, contact_region, function}`。
- `pattern`（如 `"slide + hold"`）**非權威欄位**，一律由 `role_A + " + " + role_B` 重建。

---

## 2. 評估單位與 enum

### 2.1 sample_id

```
{object_id}/q{N}_{task_slug}
```
`qN` 前綴**必須保留**（目錄名截斷 43 字元、slug 有碰撞實例）；`task_slug` 不得單獨作為 key。

### 2.2 Primitive role set（core-8）

來源 `data_generation/pipeline/stage1_v2.py:60`：

```python
ROLE_VERBS = ["hold", "lift", "push", "pull", "press", "slide", "rotate", "squeeze"]
```

Stage 0 直接檢查 `role in ROLE_VERBS`。非法值 → `metadata_correction`，**不直接 Reject**。
實測（5,000 objects / ~22k queries）非法 role：`tilt` 71、`none` 23、`roll` 11、
`shake` 3、`fold` 3、`press, pull` 1、`trim` 1、`scrub` 1（約 0.2%）。

### 2.3 凍結 enum（實測驗證）

- `category ∈ {inter, intra, pose}` — 實測無其他值。
- `coordination ∈ {whole-body, stabilize+actuate, co-actuate}` — 實測另有 `"none"` 81 筆、
  `null` 38 筆（約 0.5%）→ `metadata_correction`。
- Dataset Card 需補三種 coordination 的文字定義。

---

## 3. Verification Stage 0：Programmatic Audit

對 **100% samples** 執行（純程式，唯一負擔得起全量的階段）。

**不負責**判斷 task 是否合理、contact region 是否符合物件、heatmap 是否落在正確部件、
雙手配置是否能完成 task。

### 3.1 檔案與 schema

| 檢查 | 規則 |
|---|---|
| `meta.json` | 存在、可讀、可解析 |
| `scores.npz` | 存在、可載入 |
| point cloud | **= `scores.npz` 內的 `xyz`（無獨立檔案）**，shape `(N,3)`、有限值、N > 0（實測 N = 16,384） |
| `sample_id` | 符合 `object_id/qN_task_slug` |
| `object_id` | 與 folder / mapping / point cloud 一致 |
| roles | 恰好 2 個，`id` 為 A 與 B |
| required fields | `role`/`target`/`contact_region`/`function` 不可為空 |
| primitive roles | A、B 皆 ∈ core-8 |
| category | ∈ 凍結 enum |
| coordination | ∈ 凍結 enum |

NPZ 欄位分級（更新自舊 D3）：
- **必要**：`xyz`、`scoreA`、`scoreB`（長度必須一致）
- **正式 run 應有**：`scoreA_raw`、`scoreB_raw`（§3.6 soft overlap 需要；缺 → warning 並跳過該指標）
- **optional/diagnostic，缺少不構成 schema failure**：`ptsA`、`ptsB`、`ptsA_multi`、`ptsB_multi`、
  `pexistA`、`pexistB`、`molmo_queries`、`task`、以及舊批次才有的 `gt`、`counts`、
  `sel_version`、`partition_axis`
- meta.json 必要欄位：`object_id`、`object_name`、`task`、`query`、`category`、`coordination`、
  `roles`（2 筆）；診斷欄位 `n_views`、`hitA`、`hitB`、`coverage`、`engine`、`seconds`。

### 3.2 Heatmap 數值檢查（A、B 分別）

NaN / Inf / 超出 `[0,1]` / 全零 / 長度 ≠ point count / A 與 B 完全相同 /
active region 異常大或異常小。

### 3.3 Hard Stop

必要檔案缺失或不可讀、point count ≠ heatmap 長度、NaN/Inf、score 越界、
A 或 B heatmap 全零、必要 metadata 缺失、roles 數量 ≠ 2。

Hard stop ≠ 永久 Reject，代表必須先做 metadata correction 或 heatmap regeneration。

### 3.4 Metadata Correction Flags

role ∉ core-8、category 不合法、coordination 不合法、
`role`/`target`/`contact_region`/`function` 缺失、sample_id/task index/object mapping 不一致。

> **若修正改動了 role / target / contact region / function / Molmo query，
> 對應的 Generation Stage 2 heatmap 必須重新生成，不可沿用舊 heatmap。**

### 3.5 Warning（不 Reject）

A/B 完全或近乎相同、raw A/B overlap 過高、center distance 極端、`hitA`/`hitB` 偏低、
`coverage` 偏低、active region 過大或過小、final heatmap 過度集中於極少數 points。
（門檻 pilot 後凍結，現值 PROVISIONAL。）

> `active_region_large_B` 是**已知資料性質的 diagnostic**（§4.1.2 第 4 點）：
> 不進 hard failure、不進 Reject gate、不計入正式錯誤率。

`hitA`/`hitB`/`coverage` 定義（來源 `stage2_v2.py:213-214`）：`hitA` = 40 views 中 Molmo 對
Hand A 回傳 finite 點的 view 數；`coverage` = `(counts > 0).mean()`。皆為生成時計算、
事後不可重算，**只作 warning，不作 hard gate，不提供給 judge**。

### 3.6 Dual Geometry 統計（只留兩個）

**A/B Soft Overlap — 用 `scoreA_raw` / `scoreB_raw`**（final 已 disjoint partition）：

```
O_AB = Σ min(s_raw^A, s_raw^B) / (Σ max(s_raw^A, s_raw^B) + ε)
```

**Normalized Center Distance — 用 final `scoreA` / `scoreB`**：

```
c_X = Σ s^X p / (Σ s^X + ε)        D_AB = ‖c_A − c_B‖₂ / (d_bbox + ε)
```

計畫書明文**不再計算**：pattern-specific geometry、region-size ratio、
connected components、multi-view stability。兩者皆只產生 warning。

> ⚠ 舊 spec D2「一律使用 scoreA/scoreB」已被計畫書 §4.6.1 部分推翻：
> **soft overlap 改用 `*_raw`**，其餘（渲染、Soft IoU/SIM/MAE、最終 supervision）維持 final。

> **【L1】不做任何 geometry usability 判定。** 除了 §3.1 對 `xyz` 的基本可讀性檢查
> （shape/有限值/長度一致）之外，不再有幾何品質評分——沒有 geometry sheet、沒有
> object-level pass、也沒有程式化的 geometry 品質 warning。點雲若有問題，只能透過
> Stage 2 grounding 間接反映。

### 3.7 輸出

```json
{
  "schema_version": "verification-stage0-v1.0",
  "sample_id": "0a3a2273.../q0_pick_up_the_teapot",
  "schema_pass": true,
  "heatmap_A_valid": true, "heatmap_B_valid": true,
  "role_A_valid": true, "role_B_valid": true,
  "raw_soft_overlap": 0.18,
  "normalized_center_distance": 0.41,
  "hard_failures": [], "metadata_correction_flags": [], "warning_flags": []
}
```

---

## 4. Canonical Active-Region Visualization Protocol

> ### 【L4 — 2026-08-05 使用者決定：canonical render 改用 `render_eval.py`】
>
> 呈現層改用 `pipeline/render_eval.py`（原 `data_generation/outputs/stage2_eval/render_eval.py`），
> **取代** `render.py` 於 2026-07-28 凍結的六視圖 τ=0.20 協定。差異：
>
> | | render.py（舊，07-28 凍結） | **render_eval.py（新 canonical）** |
> |---|---|---|
> | 視角 | 6（front/left/upper-front/back/right/upper-back） | **8**：6 個 orbit 方位 + top-down + underside |
> | 門檻 | τ_vis = 0.20（A/B 共用） | **0.15**（A/B 共用，仍是 absolute、仍不做 per-sample normalization） |
> | 配色 | A 紅 / B 藍 / 重疊紫 / inactive 灰 | A 橘 / B 蒂芬妮綠，**深色 = 高分**，物體淺灰 #DBDBDB |
> | 重疊處理 | 紫色 | 取分數較大者上色（A/B 互斥） |
> | 強度 | 原始線性 score | 正規化到 [門檻, p98] 後映射到色階 |
>
> 理由：`claude_pilot_1000/` 的 1,000 筆 heatmap 標註是在這個協定下做出來的，且 8 視角能涵蓋
> top/underside，heatmap 錯在頂面或底面時無法藏。§4.1 的原則（absolute threshold、A/B 共用、
> 不做 per-sample normalization、mask 只決定上色範圍而強度仍連續）**全部保留**，改的是門檻值、
> 視角數與配色。
>
> **必須重跑的東西**：Stage 2 的 probe（29 筆）、judge smoke（24 筆）與 `corrupt_stage2.py`
> 的 160 筆 corruption 判定全部是在舊六視圖 τ=0.20 下做的，**不可與新協定的結果混用**；
> 要沿用其結論必須用 `render_eval.py` 重跑。`render.py` 保留供該批歷史結果重現，不再是 canonical。
> 另注意新協定的紅藍語意也變了 —— Stage 2 prompt 裡「red = Hand A, blue = Hand B」的措辭
> 必須同步改成橘/蒂芬妮綠，否則判官會被顏色名稱誤導。

### 4.1 固定絕對門檻 τ_vis（**取代舊 D7 的 top-k%**）

```
M^A_i = 1[s^A_i ≥ τ_vis]        M^B_i = 1[s^B_i ≥ τ_vis]
```

- τ_vis 為**全資料集統一**的 absolute threshold，A/B 共用。
- **不對每個 sample 調整、不做 per-sample min-max normalization。**
- 計畫書 §5.2 明文**不採 Top-k**：top-k 會強制每個 sample 顯示近似固定比例的點，
  遮蔽 score scale 差異，使 `over_expanded` / `under_localized` 無法判讀。
- binary mask **只決定哪些點上色**；active region 內顏色強度仍用原始 continuous score
  → 最終顯示仍是連續 heatmap，不是 binary segmentation。
### 【已凍結 2026-07-28】

```json
{
  "mask_method": "absolute_threshold",
  "threshold": 0.20,
  "per_hand_threshold": false,
  "intensity_mapping": "original_score_[0,1]"
}
```

- **τ_vis = 0.20**，A、B **共用同一個 absolute threshold**。
- mask 內顏色強度 = **原始 [0,1] score**（線性），不做 gamma、不做 floor、
  不做 per-sample min-max → 顯示仍是連續 heatmap，不是 binary segmentation。
- `render.py` 仍保留 `--tau` 與 `--intensity gamma` 供事後重新比較，
  但**釋出協定就是上面那組值**。
- ⚠ 若要重跑比較，樣本必須抽自新 run（`outputs/stage2/`），不可用
  `stage2_example_data`——兩批 score scale 不同（§1.2）。

**τ_vis = 0.20 的實測後果（400 samples）**：active fraction 中位數 A 0.339 / B 0.515
（mean A 0.321 / B 0.593，p90 A 0.647 / B 0.977）。區域偏大是**刻意接受**的——
面積不再是 `over_expanded` 的判準（§6.4.1），改由語意判斷。

**線性強度的可讀性已驗證**：active 點中只有 **5.7%** 的 score < 0.40，
其餘都落在高分區間 → 用原始 score 當強度不會讓多數點褪成灰色。
（`--intensity gamma` 保留為事後可讀性比對工具，非釋出協定。）

#### 4.1.1 實測 τ 掃描（600 samples，daily_used 新 run，2026-07-27）

active region 佔全部點的比例（median）與被門檻清空的比例：

| τ | median A | median B | A 全空 | B 全空 |
|---:|---:|---:|---:|---:|
| 0.3 | 0.220 | 0.517 | 2.0% | 0.7% |
| 0.5 | 0.141 | 0.497 | 7.2% | 3.5% |
| 0.7 | 0.028 | 0.315 | 23.3% | 13.0% |
| 0.8 | 0.010 | 0.125 | 34.7% | 20.3% |
| 0.9 | 0.000 | 0.011 | 52.3% | 36.8% |

⚠ **Hand B 系統性比 Hand A 寬得多**，且不是 peak-scale 假象
（score max median：A 0.897 / B 0.947，僅 57.5% 的 sample B>A）——B 的高分點**數量**本來就多，
與 B 多半是 hold/stabilize、接觸大面積 body 的物理事實一致。

**後果：計畫書 §5.3「A、B 使用相同 threshold」在實測下沒有一個好解**——
τ 低則 B 過度擴張（τ=0.5 時 B 佔一半點雲），τ 高則 A 被清空（τ=0.7 時 23% 的 A 全空）。
Phase 2 pilot 必須正面處理，選項：(a) 接受單一 τ 並選在折衷點；(b) 允許 per-hand τ
（偏離計畫書，需明確記錄）；(c) 回頭檢查 stage2 的 partition/prune 是否讓 B 過寬。
**目前 `TAU_VIS = 0.30` 只是 placeholder，警告數量（61% 的 sample 觸發
`active_region_large_B`）反映的是門檻未校準，不是資料壞掉。**

> **【指示 2026-07-27】現階段不凍結任何 threshold。**
> `render.py` 的 `τ_vis` 必須是可調參數，且要能**對同一批 sample、同一組相機與配色
> 產生多個 τ 的圖**供並排比較（不是每次重挑樣本）。

#### 4.1.2 已知特性：凍結版 Generation 輸出的 A/B 寬度不對稱

**【決策 2026-07-28】Generation Stage 1／2 的輸出視為凍結版本。不修改
`is_body_target()`、Molmo、SAM2、partition 或 prune。** 以下是必須被視為**資料的已知性質**、
而非待修 bug 的行為，由 `check_partition.py`（300 samples，重算鏈與實際輸出 300/300 完全一致）測得：

1. **`body` target 的 raw heatmap 系統性比 named-part target 寬**：
   `target == "body"` 的 raw active ratio 0.849、final 0.651；named part 為 0.288 / 0.200。
2. **Hand B 經常對應 `hold` + `body`，因此 B 的 active region 通常較大**。
   這是 role 語意造成的，不是 A/B 本身有偏差——對稱 pair 完全平衡
   （`lift+lift` A 0.389 / B 0.421、`push+push` 0.519 / 0.479、`rotate+rotate` 0.312 / 0.306）。
3. **Partition 會把 A/B 差距由約 1.27× 放大到 1.82×**（raw 1.27× → refined 1.27× →
   partitioned 1.82× → pruned 1.82×；mass 保留率中位數 A 0.786 / B 0.964），
   **但主要寬度在 raw grounding 階段就已存在**，partition 只是放大而非製造。
   All-zero 同理繼承自 raw（raw B 全零 1.3%），partition 不製造 all-zero。
4. **`active_region_large_A/B` 只是 diagnostic warning**：不納入 hard failure、不進 Reject gate、
   不計入正式錯誤率，也不因此觸發 heatmap regeneration。它描述的是已知的資料性質。
   **`hold` + `body` 本來就可能覆蓋較大表面，因此 active-region ratio 大不直接視為
   `over_expanded`。**
   ⚠ **但 body target 並非自動豁免**——判定改由 Stage 2 依 `contact_region` 與 `function`
   作語意判斷，見 §6.4。
5. **不採用 per-hand threshold。** A、B 共用同一個 τ_vis。理由：per-hand 門檻只會讓 B 在圖上
   變小，而**實際釋出的 supervision 分布完全沒變**——那是用 rendering 掩蓋生成端分布，
   會讓所有下游品質數字失真。優先順序為：固定 gamma／opacity mapping > adaptive mask >
   （最後才考慮）per-hand threshold。

> 這些性質必須寫進 Dataset Card 與論文的 limitation，不可默默略過。
- **`τ_vis ≠ exist_thr`**：`exist_thr`（`stage2_v2.py`，預設 0.9）是 Molmo 判斷部件是否存在的
  view-level gate，控制哪些 view 進 SAM2，與渲染門檻無關，兩者分開記錄與校準。

Pilot 凍結物：

```json
{"mask_method": "absolute_threshold", "threshold": "<pilot>", "intensity_mapping": "original_score_[0,1]"}
```

### 4.2 顯示規則

非 active = 灰；Hand A active = 紅（強度依 `scoreA`）；Hand B active = 藍（強度依 `scoreB`）；
A/B 同時 active = 紫。final 已 partition，紫色理應罕見；若大量出現 → warning。

必須記錄的 rendering protocol：`mask_method`、`threshold`、`score_to_color_mapping`、
camera poses、point size、background、object normalization、crop rule、rendering software/version。

### 4.3 Soft vs Binary 的使用範圍

- **必須用 continuous score**：A/B Soft Overlap、Soft IoU、SIM、MAE、soft weighted centroid、
  cross-task combined-heatmap similarity。
- **binary mask 只用於**：Stage 2 渲染、Human Review 渲染、union coverage、active-area 統計、
  （未來若啟用）binary Precision / IoU。

---

## 5. Verification Stage 1：RGB-conditioned Visual-Semantic Judge

計畫書指定 `Qwen3-VL-30B-A3B-Instruct`（本地硬體限制見 §11）。取代舊版純文字 LLM judge。

### 5.1 輸入

- object：`object_id`、`object_name`、`views_used`、`components`
- query：`task`、`query`、`goal`、`why_bimanual`、`category`、`mechanism`、`operation`、
  `coordination`、Hand A/B 各自的 `role`/`target`/`contact_region`/`function`、
  `relation`、`symmetric`、`hand_agnostic`
- 影像：`views_used` 所列**全部可讀 RGB**（實測 8 張／object），**固定順序**，
  不重新隨機選 view；圖片不可讀時記錄缺失 view，**不以其他圖片無紀錄替換**。
- 同一 object 的 RGB **載入一次**，供該 object 所有 queries 重複使用（降低 I/O 與 prefill）。

### 5.2 明文不提供（避免被 downstream grounding 汙染）

`scoreA`/`scoreB`、`hitA`/`hitB`、`coverage`、Stage 0 warnings、Molmo existence confidence、
SAM2 confidence、Stage 2 spatial result。

### 5.3 四個評估項目

1. **task_plausibility** `∈ {0,1,2}` — 2 = 合理且影像足以支持；1 = 可能合理但影像不足／
   部件被遮擋／操作方式不確定；0 = 明顯不合理、object-task 錯配、task 無法由該物件支持。
2. **bimanual_validity** `∈ {valid, acceptable, invalid}` —
   `acceptable`（雙手可完成但單手也夠）**不作為嚴格淘汰條件**；其他項全對則可保留，
   若同時有其他 warning → Human Review。
3. **hand_A_visual_semantic_consistency** `∈ {0,1,2}` — 整合判斷 `role + target + function +
   contact_region`：role 是否支持 task、function 是否符合 role、target 是否與 RGB 可見結構一致、
   contact region 所述部件是否存在、位置是否合理、A/B 是否形成合理分工。
   附兩個結構化欄位：
   - `contact_region_evidence ∈ {supported, uncertain, absent}`
   - `role_function_consistency ∈ {consistent, uncertain, conflict}` — **只依該手的 `role` 與
     `function` 文字判斷，不看 RGB、不看 task、target、contact_region**。
     判準（v2.1 強化）：問的是 **primitive 的「直接物理效果」——它做出的運動與施加的力——
     能不能產生該 function**。**function 對整體任務有幫助不算數**，role 必須是產生它的原因。
     `consistent` = 直接物理效果能合理產生該 function；
     `uncertain` = function 太廣泛、太間接或資訊不足；
     `conflict` = 兩者在**運動、施力方向、物體是動還是靜止、或功能目的**上明顯矛盾。
     （拆出此欄位的原因：8B corruption test 中 `role_function_task_conflict` 只有 .12 攔截率，
     judge 讀 `function` 而完全忽略 `role`；把該判斷變成必填欄位，強迫它明文表態。）

   **v2.1（2026-08-02）在 SYSTEM prompt 追加 5 個 worked examples**（hold/stationary → consistent、
   slide/stationary → conflict、press/press-against-table-to-prevent-movement → **consistent**、
   press/pulls-outward → conflict、slide/helps-control → uncertain）。
   例子刻意包含 **press 也能直接提供 stabilization**，以免模型學成
   「function 提到 stabilize 就必須是 hold」這種絕對規則；並明寫「比較直接物理效果，
   不要記住這些字詞」。JSON schema 與 `derive_error_tags()` 規則不變 → v2.0/v2.1 結構可直接比較。
4. **hand_B_visual_semantic_consistency** — 同規則，**A/B 分開輸出**以定位問題來源。

### 5.4 Error tags（固定集合，**2026-07-31 起改為 rule-based**）

**VLM 不再自行產生 error_tags**，schema 已移除四個評估項目內的 `error_tags` 欄位。
tags 由 `judge_stage1.derive_error_tags()` 依結構化結果**程式產生**，寫在紀錄的
**最上層** `error_tags`：

| 條件 | tag |
|---|---|
| `task_plausibility.score == 0` | `implausible_task` |
| 任一手 `contact_region_evidence == "absent"` | `contact_part_absent` |
| `bimanual_validity.label == "invalid"` | `bimanual_conflict` |
| 任一手 `role_function_consistency == "conflict"` | `role_function_conflict` |

固定集合＝上表四個。移除 `role_task_conflict`、`contact_region_implausible`。
**`uncertain` 不產生 error tag**，但屬 Human Review 候選 —— Stage 3 必須自己讀
`role_function_consistency` / `contact_region_evidence`，不可從「tag 為空」推論無疑慮。

改動理由：8B 的自由生成 tag 不具鑑別力（在 bimanual_conflict 樣本上誤觸發
`role_task_conflict` 15 次、在真正的 role conflict 上 0 次）。改為 rule-based 後，
tag 成為結構化欄位的確定性函數 → **corruption test 的 tag hit rate 不再是獨立訊號**，
只能讀作「judge 是否在預期的欄位上出錯」。

### 5.5 輸出

```json
{
  "judge_version": "qwen3vl-visual-semantic-v2.0",
  "sample_id": "...",
  "task_plausibility": {"score": 2, "reason": "..."},
  "bimanual_validity": {"label": "valid", "reason": "..."},
  "hand_A_visual_semantic_consistency": {"score": 2, "contact_region_evidence": "supported", "role_function_consistency": "consistent", "reason": "..."},
  "hand_B_visual_semantic_consistency": {"score": 2, "contact_region_evidence": "supported", "role_function_consistency": "consistent", "reason": "..."},
  "evidence_views": ["00000", "00008", "00017"],
  "self_reported_confidence": 0.91,
  "error_tags": []
}
```

固定：checkpoint、system prompt、JSON schema、image ordering、low-temperature/deterministic、
**max retry = 1**、raw 與 parsed output 都保存。不允許模型新增未定義的核心欄位。
`self_reported_confidence` 只作紀錄，**不作 Auto-Accept hard gate**。

---

## 6. Verification Stage 2：Point-cloud Spatial Judge

計畫書指定 `Qwen2.5-VL-32B-Instruct`（本地硬體限制見 §11）。
**Stage 2 不重新評估 task 是否合理**；若 Stage 1 已判定 task/metadata 不成立，
應先修 metadata，而不是要求 Stage 2 解釋錯誤 metadata。

### 6.1 【L1】完全省略 geometry 審查

**決策：不渲染 geometry-only 視圖，也不做任何 `geometry_usable` 判定——該欄位整個不存在。**
影響的所有位置（避免遺漏）：

| 位置 | 原規格 | 改為 |
|---|---|---|
| Stage 2 渲染 | geometry 6 視圖 + combined 6 視圖 | **只渲染 combined 6 視圖** |
| object-level pass | 每 object 一次 geometry usability 判定 | **取消** |
| `geometry_usable` 欄位 | Auto-Accept 需 `= 2`；Human Review 觸發 `= 1` | **欄位刪除**，不出現在 manifest／gating |
| Stage 0 | — | **不新增**任何 geometry 品質檢查（只保留 §3.1 的 `xyz` 可讀性） |
| Phase 4 步驟 3 | Object-level Geometry Usability | **刪除**，Phase 4 變 4 步 |
| §8.3 人工輸入 | 六張 geometry views + 六張 combined views | **只給六張 combined views** |
| §8.3 人工評分項 5 | 「Point cloud geometry 是否可用？」 | 刪除，人工評分改 8 項 |
| §9.1 Krippendorff α | 含 Geometry usability | 刪除該項 |
| 論文表 2 | 含 Geometry usability 列 | 刪除該列 |
| §7.5 Reject 條件 | 「point cloud geometry 無法使用」 | 保留，但改由人工在 Human Review 時判斷，非自動流程 |

> 效益：每 sample 少渲染 6 張圖（渲染量減半），省下約 63k 次 object-level VLM 呼叫。
> 代價：不再有 geometry 可用性的獨立判定，point cloud 破損只能靠 Stage 2 grounding
> 間接察覺 → **需在論文/Dataset Card 誠實說明此限制**。

### 6.2 輸入

`scores.npz`（`xyz`、`scoreA`、`scoreB`）＋ `meta.json` ＋ point cloud；
metadata 用 `object_name`、`task`、`query`、Hand A/B 的 role/target/contact_region/function、
`coordination`、`relation`、`symmetric`、`hand_agnostic`。

### 6.3 影像

Combined heatmap 六視圖：`front`、`left`、`upper-front`、`back`、`right`、`upper-back`。

- **每張個別傳入 VLM、各自帶明確 view label。**
- **不拼接成一張大 panel**（推翻舊 D13 的 2×3 contact sheet）。
- **不建立 Hand A-only / Hand B-only sheets。**
- 即時渲染、推論後刪除；不永久保存全資料集 render images。
- 所有 samples 固定：camera poses、point size、背景、object normalization、crop rule、
  τ_vis、score-to-color mapping、view naming。

### 6.4 三個評估項目

1. **hand_A_grounding** `∈ {0,1,2}` — 紅色區域是否位於符合 contact region 的實際部位、
   符合 role 與 function、沒有落到錯誤部件、區域大小合理。
   error tags：`wrong_region`、`over_expanded`、`under_localized`
2. **hand_B_grounding** — 同規則。

#### 6.4.1 `over_expanded` 判準（2026-07-28 定案）

**判準是「高分區域有沒有蓋到 metadata 沒有指定的無關表面」，不是「佔了多少比例的點」。**
不得用 active-region ratio 當作 `over_expanded` 的依據。

依 `contact_region` 的**具體程度**分兩類：

| `contact_region` 的描述 | 大面積是否合理 | 判定 |
|---|---|---|
| 泛指整體表面（`the body`、`the outer surface`…），且 `function` 是穩定／支撐 | 合理 | 覆蓋大片 body **不標** `over_expanded` |
| 指定特定側面／特定高度／特定局部（`the mid-height of one side wall`、`the top of the handle`、`the side opposite the handle`…） | 不合理 | 高分區域若擴張到**大量無關表面**（延伸到相對側、爬到其他部件、蓋滿整個 body），**仍標** `over_expanded` |

即：`hold` + `body` 只是**不因為面積大而被自動判為錯**，**不是自動豁免**。
Stage 2 必須實際比對 `contact_region` 與 `function`：
標註要求特定側面或特定高度，而紅／藍區域卻無差別地蓋滿整個物件 → `over_expanded`。

> Prompt 撰寫注意：rubric 只能寫**推理原則**（「比對 contact_region 指定的具體程度」），
> 不得列舉特定物件或特定部件的例子——依 [[prompt-design-principle]]，
> 具體範例會讓 judge 過擬合到那些例子。
3. **dual_coordination** `∈ {0,1,2}` — 兩區域合起來是否能合理共同支持該 task：
   空間配置是否合理、是否落在可共同施力的位置、是否只有一手正確、是否 A/B collapse、
   是否因分區錯誤導致互相衝突、是否與 metadata `relation` 一致。
   error tags：`ab_collapse`、`only_A_valid`、`only_B_valid`、`only_one_correct`、
   `spatially_incompatible`

### 6.5 輸出

```json
{
  "judge_version": "qwen2.5vl-spatial-v1.0",
  "sample_id": "...",
  "geometry_reference": {"object_id": "..."},
  "hand_A_grounding": {"score": 2, "error_tags": [], "reason": "..."},
  "hand_B_grounding": {"score": 2, "error_tags": [], "reason": "..."},
  "dual_coordination": {"score": 2, "error_tags": [], "reason": "..."},
  "evidence_views": ["front", "left", "upper-front"],
  "self_reported_confidence": 0.89
}
```

confidence 同樣只作紀錄，不作 hard gate。

---

## 7. Verification Stage 3：Decision Integration

**不採加權平均**（`w0S0 + w1S1 + w2S2`），採 **rule-based gating + human calibration**。

### 7.1 Auto-Accept（全部滿足）

Stage 0 無 hard failure、A/B roles ∈ core-8、`task_plausibility = 2`、
Hand A/B visual-semantic consistency 皆 = 2、bimanual_validity = `valid`（或僅 `acceptable`
而無其他問題）、Stage 2 Hand A/B grounding 皆 = 2、`dual_coordination = 2`、
無重大 error tags、Stage 1 與 Stage 2 無實質 disagreement、Stage 0 warnings 不構成明顯風險。
（**`geometry_usable = 2` 條件依 L1 移除**。）

若 `acceptable` 是唯一 warning 且其他全對 → 可 Auto-Accept，但保留 flag `bimanual_acceptable`。

### 7.2 Human Review（任一即送人工）

任一核心評分 = 1；Stage 1 與 Stage 2 不一致；bimanual_validity = `acceptable` 且另有 warning；
raw soft overlap 或 center distance 位於極端區間；Stage 2 標記 `over_expanded` / `under_localized`；
A/B 有交換或對稱歧義；task 操作方式少見但不一定錯；RGB 無法確認 contact part 是否存在；
Stage 0 有警告但 Stage 2 無法明確判斷。（**`geometry_usable = 1` 條件依 L1 移除**。）

人工可作出：`accept` / `accept_with_flag` / `correct_metadata` / `regenerate_heatmap` / `reject`。

**Stage 1↔Stage 2 disagreement 操作型定義**（沿用舊 D11，仍適用；只比對應維度，不比總 decision）：
- `hand_X_visual_semantic_consistency = 2` 且 `hand_X_grounding = 0`（或反向），X ∈ {A,B}
- `bimanual_validity = valid` 且 `dual_coordination = 0`
- `bimanual_validity = invalid` 且 `dual_coordination = 2`
- 得分 1 不算 disagreement（直接進 Human Review）；`task_plausibility` 無對應 Stage 2 指標，不比。

### 7.3 Metadata Correction / Regeneration

適用：role ∉ core-8；target/contact_region/function 缺失；Stage 1 hand consistency = 0；
task 合理但 role/function/contact 組合不一致；contact region 所述部件不存在；
object-task pairing 可透過修正補救。

修正後必須依序重跑：**Verification Stage 1 → Generation Stage 2 → Verification Stage 0 →
Verification Stage 2 → Stage 3**。

### 7.4 Heatmap Regeneration

適用：Stage 1 task 與 metadata 合理但 Stage 2 grounding = 0；明顯 `ab_collapse`；
只有一手合理；heatmap 過度擴張或過度稀疏；heatmap 落在錯誤部件。
重新生成後必須再通過 Stage 0 與 Stage 2。

### 7.5 Reject

task 本身明顯不成立；`wrong_object_task` 且無合理修正方式；bimanual_validity = `invalid`
且無法修正；point cloud geometry 無法使用；多次 regeneration 仍失敗；
object-task pair 本質上不適合作為 dual affordance sample。

### 7.6 Final manifest

```json
{
  "schema_version": "quality-manifest-v2.0",
  "sample_id": "...",
  "stage0_programmatic_status": "pass",
  "stage1_visual_semantic_status": "pass",
  "stage2_spatial_status": "pass",
  "bimanual_validity": "valid",
  "final_status": "auto_accept",
  "quality_flags": [],
  "human_reviewed": false,
  "annotation_version": "v1.0",
  "verification_versions": {
    "stage0": "verification-stage0-v1.0",
    "stage1": "qwen3vl-visual-semantic-v1.0",
    "stage2": "qwen2.5vl-spatial-v1.0",
    "rendering": "dual-heatmap-render-v1.0"
  }
}
```

`final_status ∈ {auto_accept, human_accept, accept_with_flag, metadata_corrected,
heatmap_regenerated, rejected}`。

---

## 8. Verification Stage 4：人工評估

### 8.1 Calibration set

150–300 object-task pairs，抽樣須涵蓋：不同 object categories、不同 tasks-per-object、
不同 unordered role pairs、same-role 與 different-role、`valid`/`acceptable`/疑似 `invalid`、
Stage 0 warnings、Stage 1 score 0/1/2、Stage 2 score 0/1/2、extreme raw overlap、
extreme center distance、over-expanded 與 under-localized、symmetric 與 hand-agnostic、
Auto-Accept/Human Review/Correction/Regenerate/Reject 各路徑。
**不得只抽高分資料。**

### 8.2 評審與裁決（1–2 人）

2 人先獨立評分（評分前不得看到對方答案）→ 分歧時兩人討論形成 final decision，
不強制第三位 adjudicator；討論前的兩份原始分數與討論後的 final decision **均須保存**。
Krippendorff's α（ordinal）**只用討論前的獨立原始分數**計算。
若實際只有 1 人：可完成 Human Review 與 evaluation split verification，
但**不得報告 inter-rater agreement**。

### 8.3 人工輸入與評分項（依 L1 調整為 8 項）

輸入：Stage 1 JSON、`views_used` RGB images、**六張 combined heatmap views**、
Stage 0 基本統計與 warning。人工不應只依賴 judge 的文字理由。

1. Task 是否合理？ 2. 使用雙手是否合理？ 3. Hand A metadata 是否合理？
4. Hand B metadata 是否合理？ 5. Hand A heatmap 是否正確？ 6. Hand B heatmap 是否正確？
7. A/B 是否能共同完成 task？ 8. Final decision。
（原第 5 項「point cloud geometry 是否可用」依 L1 刪除。）

Hand metadata 一次整合 `role + target + function + contact_region`。
評分 `2 = Correct / 1 = Acceptable-Uncertain / 0 = Incorrect`；
final decision ∈ `{accept, review, reject}`；
修正標記 ∈ `{metadata_correction, heatmap_regeneration, object_removal}`。

### 8.4 Human Gold Spatial Subset — **第一版暫緩**

不做 200–300 samples × 3 標註者 × 3D point-level painting。因此第一版**可以**主張：
task/metadata 合理性經 RGB-conditioned VLM 與人工驗證、heatmap 空間合理性經 point-cloud VLM
與人工檢查、dual coordination 經自動與人工審核。**不可**宣稱：已用人工 point-level GT
精確驗證完整資料集 heatmap IoU、已建立大規模人工 per-point A/B segmentation benchmark。

### 8.5 對稱案例 permutation-invariant 評分

對 `lift+lift`、`squeeze+squeeze`、`hold+hold`：若做 point-level 比較，需同時比
`A→GT-A, B→GT-B` 與 `A→GT-B, B→GT-A`，取較佳者。第一版用於 A/B swap stability test
與人工 rubric。

### 8.6 Evaluation split

以 **object 為單位**與 training split 分離；每筆皆經人工確認；明顯錯誤者修正/重生成/移除；
保留 `acceptable` 與 ambiguity flags；**不以 Stage 1 或 Stage 2 judge 作為唯一 ground truth**；
保存人工 final decision 與所有修正紀錄。

---

## 9. Judge 可信度驗證

### 9.1 與人工比較（主要指標）

1. **Auto-Accept Precision** = 人工也接受的 Auto-Accept samples / 所有 Auto-Accept samples
   → 完整 training data 自動篩選**最重要**的指標
2. **Bad-Sample Recall** = 被系統攔截的人工錯誤 samples / 全部人工錯誤 samples
   （攔截 = Human Review ∪ Metadata Correction ∪ Heatmap Regeneration ∪ Reject）
3. **Macro-F1**（ordinal 三類 2/1/0 分別算 F1 後平均）
4. **Quadratic Weighted Kappa**
5. Confusion Matrix（僅供分析，不作主表單一分數）

**不列為主要結果**：一般 Accuracy / Precision / Recall、多種重複 Kappa。

「攔截」的 strict vs lenient 定義是 **calibration 階段的槓桿，不硬編**（舊 D14）：
strict = 核心分 = 0 或 invalid；lenient = 核心分 ≤ 1 或 ≠ valid。優先保 Auto-Accept Precision。

### 9.2 Synthetic Corruption Test

**A. Programmatic（6 類，detector = program，須達 100% detection）**
`missing_file_or_key`、`shape_mismatch`、`nan_inf_out_of_range`、`one_hand_all_zero`、
`invalid_enum`、`non_core8_role`。達成即視為功能驗證完成，不再反覆作為模型能力實驗。

**B. Visual-Semantic（detector = Stage 1）** —— 實作為**四類**，見下方「分類 v3」為準：
- `implausible_task`：改成該物件無法合理完成的操作
- `role_function_conflict`：一手 role 改成與**自己的 function** 衝突的 primitive
- `contact_conflict`：contact region 改為 RGB 中不存在或不合理的部件
  （Stage 1 有 RGB → 可檢查**零件是否真的存在**，不只文字內部矛盾）
- `bimanual_conflict`：使兩手分工不再形成合理雙手操作
- ~~`wrong_object_task`~~：把其他物件的 task 配進來 —— **已移除，不測**

已刪除：opposite-map wrong role、對稱案例的 A/B swap、不影響語意的 function swap、
隨機跨物件借用但仍可能合理的 contact、任何無法保證 corruption 後一定錯誤的操作。
**每類都要建立人工確認的錯誤樣本**，避免 corruption 後仍然合理。

**C. Spatial（4 類正式 gate ＋ 1 stress，detector = Stage 2）**
`wrong_region`（含原 wrong-task heatmap）、`over_expanded`、`ab_collapse`、`only_one_correct`；
`random` **只作 stress test，不納入正式 gate**。
已刪除 `missing_one_hand`（Stage 0 全零即可攔截）與 `wrong_task_heatmap`（併入 wrong_region）。
**Spatial 須分 type 個別報告，不合併為單一 headline detection rate。**

Parse fail：自動重試 1 次 → 仍失敗記 `judge_failure`，**不算成功攔截**。

### 9.3 Consistency Tests

- **Repeated inference**：相同輸入重複推論，比對核心分數 / error tags / final decision
- **Prompt paraphrase**：只改寫表達、不改評分標準
- **Image-order**：調整 multi-view 輸入順序
- **A/B swap stability**：對 `hand_agnostic = true` 或 symmetric samples 交換 A/B metadata
  與紅藍 heatmap，檢查是否 permutation-consistent

---

## 10. Dataset-level 品質與多樣性

分四塊，**避免把品質指標誤稱為 diversity，避免跨章節重複報告**。

### A. Scale and Composition
objects 數、object-task pairs 數、object category 分布、tasks-per-object 分布、
**overall primitive-role 分布（A/B 合併統計**，因 A/B 不代表固定左右手**）**、
**unordered role-pair 分布**（`lift+hold == hold+lift`）。

### B. Quality Statistics
exact duplicate count/ratio（mesh hash 或 normalized point-data hash）、
bimanual valid/acceptable/invalid ratio（審核前）→ 釋出時另報 corrected/regenerated/removed
count 與 **residual invalid ratio（原則上應為 0）**、A/B raw soft-overlap 分布、
normalized center-distance 分布、over-expanded heatmap error ratio。
> 後三者是**單一 sample 內部**的 A/B 關係，不能單獨證明跨 task 的空間多樣性。

### C. Diversity（四項主要指標）

1. **Object near-duplicate ratio** — 先排除 exact duplicates，其餘用固定 normalized
   point-cloud embedding 比相似度；threshold 經 pilot 人工檢查後凍結並公開。
   定義 = 至少屬於一組 near-duplicate cluster 的 objects / 所有 unique objects。
   **不以 pair 數作為主要比例**（大 cluster 會產生大量組合而扭曲）。
2. **Within-object semantic near-duplicate task ratio** — **只在同一 object 內比較**
   （`pick up the mug` vs `pick up the chair` 不算 dataset-level duplicate）。
   normalization 固定為 lowercase / 去標點 / 合併空白 / object name → `<OBJECT>` placeholder，
   **不在 normalization 階段做主觀改寫**。
   `R_task-near(o) = |{(qi,qj) ∈ P_o : Sim ≥ τ_task}| / |P_o|`，只算 ≥2 tasks 的 objects，
   dataset-level 取 **object-level macro average**。
   附帶 lexical diagnostic：`U_task(o) = |UniqueNormalizedTasks(o)| / |Tasks(o)|`（非主要指標）。
3. **Normalized role-pair entropy** — core-8 → `K_role = 8·9/2 = 36` 種 unordered pairs。
   `H_role = −Σ p_k log p_k / log 36`。
   ⚠ **分母必須固定 36，不得用「實際出現的 pair 種類數」**
   （否則只出現少數 pairs 但分布平均時會錯誤地接近 1）。
   若未來明文禁止部分組合，分母改為凍結後的合法 pair 總數並在 Dataset Card 公開。
4. **Per-object cross-task combined-heatmap similarity** —
   `H_q(i) = max(s^A_i, s^B_i)`；
   `SoftIoU(X,Y) = Σ min(X,Y) / (Σ max(X,Y) + ε)`；
   `CrossTaskSimilarity(o) = mean over task pairs`。只算 ≥2 tasks 的 objects，取 macro average，
   另報 median/quartiles/distribution。
   ⚠ **低 similarity ≠ 高品質**（不同 task 可能合理共用部件；錯誤/隨機 heatmap 也會壓低），
   必須與 Stage 2 grounding quality、dual coordination quality、wrong-region ratio、
   over-expanded ratio 一起解讀。

次要／附錄：within-object normalized task duplicate ratio、完整 unordered role-pair 分布、
per-object Union Coverage@k（`M_q = M^A_q ∪ M^B_q`，只算有 ≥k tasks 的 objects、每 object 同一個 k、
可多組抽樣取平均、必須公開 canonical mask threshold；**不可單獨解讀**）、
按 tasks-per-object 分組的空間統計。

### D. Appendix
Top 10–20 object names、point-cloud shape 統計、task 平均文字長度、same-role/different-role ratio、
unique role-pair count、Hand A/B active-area 分布、全域 task-template 分布、
完整 object-name frequency、補充 heatmap score 分布。

---

## 11. 【L2】硬體與模型

### 11.1 可用硬體（2026-07-27 實測）

| GPU | 型號 | VRAM | 狀態 |
|---|---|---:|---|
| 0 | RTX PRO 6000 Blackwell Max-Q | 98 GB | **佔用**（stage2，68 GB） |
| 1 | RTX 6000 Ada | 49 GB | **佔用**（47 GB） |
| **2** | **GeForce RTX 3090** | **24 GB** | **空閒 — 本專案唯一可用** |
| 3 | RTX PRO 6000 Blackwell Max-Q | 98 GB | **佔用**（stage2，68 GB） |

- 啟動**必須** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2`，否則會落到跑 stage2 的卡上。
- 3090 = Ampere **sm_86**：**不支援 FP8**（需 Ada/Hopper 以上）→ 官方 FP8 checkpoint 不可用。
  INT4（AWQ / GPTQ，Marlin kernel）可用。

### 11.2 記憶體算術（為何計畫書指定的模型放不下）

| 模型 | bf16 | INT4 (約) | 24 GB 可行性 |
|---|---:|---:|---|
| `Qwen3-VL-30B-A3B-Instruct`（計畫書 Stage 1） | ~61 GB | ~17 GB | 只有 INT4 可行，剩 ~6 GB 給 KV + ViT |
| `Qwen2.5-VL-32B-Instruct`（計畫書 Stage 2） | ~64 GB | ~19 GB | 極度勉強；6 張圖的 KV 恐不足 |
| **`Qwen3-VL-8B-Instruct`（← pilot 採用）** | **~17 GB** | — | **bf16 直上，免量化**，剩 ~5–6 GB 給 KV |
| `Qwen2.5-VL-7B-Instruct`（已在本機） | ~16 GB | — | 07-24 pilot 證實 7B **全盲於 role↔function 矛盾**（0.10 strict） |

HF cache 現況：`Qwen3-VL-30B-A3B-Instruct` / `Qwen3-VL-32B-Instruct` **只有 28K config、
權重未下載**；本機實際只有 `Qwen2.5-VL-7B-Instruct`(16G)、`Qwen3.5-35B-A3B`、Molmo、gemma-4。
→ `Qwen3-VL-8B-Instruct` 需下載（~17 GB）。

### 11.3 【L2 定案】Pilot 用 8B，之後換 30B

- **Pilot：Stage 1 與 Stage 2 共用 `Qwen3-VL-8B-Instruct`（bf16）**。
  選它而非 INT4-30B 的理由：免量化（沒有 AWQ checkpoint 可用性風險）、與最終的 30B-A3B
  **同家族同 chat template**，prompt / JSON schema / guided decoding 可以原封不動搬過去。
- **正式：等 GPU 0/1/3 空出後換 `Qwen3-VL-30B-A3B-Instruct`**（Stage 2 是否改回計畫書的
  `Qwen2.5-VL-32B-Instruct` 屆時再定）。**8B 的所有數字都是 lower bound，不進論文主表**；
  換模型後必須用同一批 corruption samples 重測。
- 兩階段共用同一模型 → Stage 1↔Stage 2 disagreement 變成**相關的**交叉檢查
  （輸入仍不同：RGB vs heatmap render），須在論文中誠實揭露。
- 執行環境：**`mm` env**（torch 2.8.0+cu128 / transformers 4.57.6 / vllm 0.11.0，皆滿足
  Qwen3-VL 需求；3090 是 sm_86，沒有 Blackwell 那套 flashinfer 麻煩）。
- 需 tune：`gpu_memory_utilization`、`max_model_len`、`limit_mm_per_prompt`（image 上限 8）、
  `min_pixels`/`max_pixels`（壓影像 token，8 張 RGB 是最大宗的 prefill 來源）。
- 依 [[check-official-docs-first]]：載入前先看 HF model card + vLLM recipe，不要憑經驗試錯。

---

## 12. 【L3】規模與抽樣

280,064 pairs。單張 3090 跑 8B bf16 + vLLM continuous batching，每 sample 8 張 RGB（Stage 1）
或 6 張 render（Stage 2），樂觀估 2–4 s/sample（Stage 2 另加 CPU 渲染）
→ **單一階段全量約需 7–13 天，兩階段翻倍**；換 30B-A3B 後更久。計畫書 Phase 4
「完整資料自動審核」在此硬體下不可行。實際 throughput 以 pilot 實測為準。

- **Stage 0 維持 100%**（純程式，成本可忽略，這也是 hard failure 的主要防線）。
- **Stage 1 / Stage 2 改為分層抽樣審核**，規模待定；抽樣需分層於 dataset(daily_used/electronics)、
  object category、tasks-per-object、unordered role pair、Stage 0 warning 狀態。
- 論文敘述需相應改為「**auditing** the full dataset programmatically + **statistically
  auditing** semantic/spatial quality on a stratified sample with CIs」，
  **不可宣稱每一筆都經過 VLM 審核**。
- Evaluation split 仍維持 100% 人工確認（規模小，建議 30–50 objects）。

---

## 13. 被新計畫書推翻的舊決定（不得沿用 `0723_prev/`）

| 舊 | 新 |
|---|---|
| D7 canonical mask = per-hand top-k%（k=5/10） | **固定 absolute threshold τ_vis**，明文不採 top-k |
| D8 score-to-color = 待選（gamma 暫定） | active region 內用**原始 continuous score** 強度 |
| D9 / D16 top5 vs top10 VLM 比較實驗 | **作廢**（沒有 k 可選了） |
| D13 geometry 與 combined 各一張 6-view **2×3 panel** | **六張獨立影像**、各自 view label、不拼 panel |
| D13 object-level geometry usability VLM pass | **依 L1 取消** |
| Stage 編號：audit = Stage 1 | audit = **Stage 0**（對齊 generation pipeline） |
| Stage semantic judge = 純文字 LLM（Qwen3-30B-A3B） | **RGB-conditioned VLM**（Qwen3-VL-30B-A3B） |
| D2「一律用 scoreA/scoreB」 | soft overlap 改用 **`scoreA_raw`/`scoreB_raw`**；其餘仍用 final |
| Stage 0 幾何多指標 | 只留 raw soft overlap + normalized center distance |

仍然有效的舊決定：D1(sample_id)、D3(NPZ 欄位分級)、D4(core-8)、D5(enum)、D6(hit/coverage 語意)、
D10(人工 1–2 人)、D11(disagreement 定義)、D14(偵測門檻是校準槓桿)、D15(corruption 目錄 v2)。

---

## 14. 儲存

- **Stage 1 RGB**：直接讀 `views_used` 路徑，不預先複製 G-Objaverse；依 object 按需載入；
  同 object 多 queries 共用；推論後只存 image identifiers 與 evidence view IDs。
- **Stage 2**：load → render 6 combined images → VLM → 存 raw+parsed JSON → **刪除暫存圖**。
- **永久保存**：Stage 0 audit JSON、Stage 1 raw/parsed、Stage 2 raw/parsed、Stage 3 decision JSON、
  model checkpoints 與版本、prompt versions、rendering protocol、canonical threshold protocol、
  human calibration records、Human Review samples、correction/regenerate/reject records、
  human-verified validation/test samples、少量論文定性案例。
- `data_verification/` 約 2 GB 未追蹤 — gitignore vs commit **待決**。

---

## 15. 執行階段

| Phase | 內容 |
|---|---|
| **1 規格凍結** | sample_id、core-8 validation、category/coordination enum、Stage 0/1/2 JSON schema、Stage 3 rubric、`views_used` 載入規格、Stage 2 camera poses 與 rendering protocol、`*_raw` vs final 使用定義、`hitA`/`hitB`/`coverage` 定義、model checkpoint 與 prompt version 記錄方式 |
| **2 Pilot（100–200 samples）** | 比較固定 τ_vis 候選、確認不採 top-k 的效果、測 Qwen3-VL multi-image JSON 穩定性、測 Stage 1 能否從 RGB 判斷部件存在、測 Stage 2 能否從六張獨立圖區分 A/B、測 score saturation、repeated inference / prompt paraphrase / image-order / A/B swap 一致性、建立並人工確認 corruption samples |
| **3 Human Calibration** | 150–300 pairs，2 人獨立→討論裁決；算 Auto-Accept Precision、Bad-Sample Recall、Macro-F1、QWK、Krippendorff α；據此調整 Auto-Accept/Human Review 條件、warning thresholds、rubric |
| **4 自動審核** | Stage 0（100%）→ Stage 1 →（geometry pass 依 L1 刪除）→ Stage 2 → Stage 3。圖片按需載入/即時渲染，推論後刪暫存 |
| **5 人工複查** | Stage 1/2 disagreement、任一核心 score = 1、極端 overlap/center distance、`over_expanded`/`under_localized`、rare/ambiguous tasks、contact visibility 不確定、validation/test candidates、correction 與 regeneration 結果 |
| **6 Freeze 與釋出** | 移除/修正 invalid samples、確認 residual invalid ratio = 0、凍結版本、建立 final quality manifest、完成人工驗證的 evaluation split、公開 verification protocol / prompt / model / rendering versions / duplicate 與 diversity protocols、保存代表性案例 |

Phase 2 結束後凍結：`τ_vis`、`rendering_version`、`stage1_prompt_version`、
`stage2_prompt_version`、`decision_rubric`、`warning_thresholds`、`near_duplicate_thresholds`。

---

## 16. 實作現況與待決事項

### 已完成

```
data_verification/pipeline/
├── README.md            # 本文件
├── common.py            # 常數/enum、npz+meta 契約、slugify（與 stage2_v2.py:95 逐字相同）、
│                        # expected_samples()、soft overlap / center distance / active_frac
├── audit.py             # Verification Stage 0（CLI）
├── check_partition.py   # 一次性診斷：refine/partition/prune 各階段的 A/B 寬度（§4.1.2 來源）
├── render.py            # 六視角 combined heatmap + 分層抽樣 + HTML gallery
├── corrupt.py           # Stage 1 corruption test（build / score）
├── judge_stage1.py      # Verification Stage 1（Qwen3-VL-8B on GPU2）
└── judge_stage2.py      # Verification Stage 2（probe / judge）
```

### Stage 2 實作現況（2026-07-28）

`judge_stage2.py` 兩個模式，皆用 Qwen3-VL-8B、GPU2、與 Stage 1 相同的記憶體設定。
六張獨立視角**即時渲染、推論後刪除**（計畫書 §12.2），τ_vis = 0.20 凍結協定。

**1. `probe` — A/B 分辨前置測試（計畫書 Phase 2 必測項）**

渲染的上下方向 = 原始 y 軸（已用已知樣本驗證：A「rim 上緣」y=+0.253 / B「seam 下方側壁」
y=−0.057，渲染確為紅上藍下）。取 |Δ質心高度| / 物件高度 ≥ 0.25 的樣本，
強制選擇「紅在藍上／藍在紅上／同高」，與質心真值比對：

| n | 正確 | 說「同高」 | **紅藍顛倒** |
|---:|---:|---:|---:|
| 29 | 24 (0.83) | 5 | **0** |

**0 次顛倒**——模型不會把紅認成藍，錯誤全是保守回答。前置條件通過。
（那 5 筆與分離度無關，也與 active region 大小無關，n 太小不做歸因。）

**2. `judge` — grounding 判定（24 筆 smoke test，237.8 s，1 筆 judge_failure）**

hand A 2/1/0 = 17/3/3、hand B = 18/3/2、dual = 18/2/3，Auto-Accept 17/23。
error tags 僅出現 `wrong_region` 5、`only_one_correct` 5。

> ⚠ **已知失效：8B 對 `over_expanded` 完全失明。**
>
> | 該手 active fraction | 手數 | score 2/1/0 | 標 `over_expanded` |
> |---|---:|---|---:|
> | ≥ 0.90 | 11 | 8/2/1 | **0** |
> | ≥ 0.60 | 3 | 3/0/0 | **0** |
> | ≥ 0.20 | 13 | 10/2/1 | **0** |
> | < 0.20 | 19 | 14/2/3 | **0** |
>
> 46 隻手中 `over_expanded` **一次都沒出現**。23 筆裡有 11 筆某一手覆蓋 ≥90% 的點雲，
> 其中 7 筆 dual_coordination 仍是 2。
>
> 更糟的是**它把 metadata 的文字當成觀察結果反述回來**：藍色實際覆蓋 99.3% 的整個物件時，
> 它仍宣稱「藍點集中在框架頂緣」「集中在燈桿中段」——正好是 metadata 寫的位置。
> 這兩筆的 `contact_region` 都指定了具體位置（"the upper rim of the frame"、
> "the mid-height of the lamp stem"），依 §6.4.1 應判 `over_expanded`。
>
> 與 Stage 1 的 `role_function_task_conflict` 是**同一個失效模式：以文字為錨、不讀圖**。
> Stage 2 的 8B 結果同樣只能作開發除錯。
>
> judge_failure 1/24 是退化重複（同一句反覆輸出直到 max_tokens），非 JSON 格式問題。

### Stage 2 spatial corruption test（`corrupt_stage2.py`，2026-07-28）

四類 gate，各 20 筆，**每筆與自己的 clean original 成對**。corruption 改的是 **heatmap 陣列**
（非 metadata），每筆實體化成獨立 `scores.npz` 後走完全相同的渲染與 prompt 路徑。

**eligibility filter（程式化修改不天然有效）**

| 類別 | 過濾條件 |
|---|---|
| `wrong_region` | SoftIoU(原,移後) ≤ 0.05、質心位移 ≥ 0.30×bbox 對角線、非全零、排除 symmetric/hand_agnostic，**且 SoftIoU(移後, 夥伴手) ≤ 0.10** |
| `over_expanded` | 只用 `contact_region` 指定局部位置者（排除泛指整體表面），且原 active < 0.50 |
| `ab_collapse` | 排除 symmetric/hand_agnostic、同 role、同 target、原 A/B SoftIoU > 0.30 |
| `only_one_correct` | 一次只破壞一手；**計分時**再排除 clean 判定未同時接受雙手者 |

> **建置時抓到的缺陷**：離某手區域最遠的點，通常正是**夥伴手**所在處，所以天真的「移到最遠處」
> 會落在夥伴的合理接觸區、變成半個 ab_collapse。實測 A/B SoftIoU 從 0.006 升到 **0.341**。
> 加入 `MAX_PARTNER_IOU` 後降到最大 0.095。

**建置後幾何驗證**（median）

| 類別 | 該手 active 前→後 | A/B SoftIoU 前→後 | SoftIoU(該手 前,後) |
|---|---|---|---|
| wrong_region | 0.035 → 0.035 | 0.002 → 0.016 | 0.0002（max 0.025） |
| over_expanded | 0.288 → 0.850 | 0.004 → 0.305 | 0.394 |
| ab_collapse | 0.764 → 0.132 | 0.017 → **1.000** | — |
| only_one_correct | 0.044 → 0.044 | 0.007 → 0.011 | 0.0001（max 0.049） |

**結果（8B，160 judgements，3 judge_failure，人工確認前）**

| 類別 | n | strict | lenient | **AA escape** | dropA | dropB | dropDual |
|---|---:|---:|---:|---:|---:|---:|---:|
| wrong_region | 19 | 0.37 | 0.58 | 0.42 | +0.44 | +0.11 | +0.50 |
| **over_expanded** | 20 | **0.00** | **0.10** | **0.90** | +0.05 | +0.10 | +0.10 |
| ab_collapse | 20 | 0.35 | 0.50 | 0.50 | +0.26 | +0.47 | +0.47 |
| only_one_correct | 12 | 0.25 | 0.50 | 0.50 | +0.75 | +0.42 | +0.75 |
| OVERALL | 71 | 0.24 | 0.41 | 0.59 | | | |

clean originals（n=78）：false-positive strict 0.12 / lenient 0.23、auto-accepted 0.77。
（`only_one_correct` 有 8 筆因 clean 判定未同時接受雙手而依規則剔除。）

error-tag（diagnostic）：wrong_region 0.32、only_one_correct 0.25、
**over_expanded 0.00、ab_collapse 0.00**。

> **結論：8B 在 Stage 2 完全不可用於品質結論，比 Stage 1 更差。**
> - `over_expanded` **0.00 strict、0.90 Auto-Accept escape**——把區域灌到覆蓋 85% 的物件，
>   九成仍被放行。這把 §6.4 smoke test 的觀察量化了。
> - `ab_collapse` 把 B 設成與 A **完全相同**（渲染後整片紫色、無紅無藍，已目視確認渲染正確），
>   仍有 50% 被 Auto-Accept，且 `ab_collapse` tag **一次都沒發過**。
> - OVERALL Auto-Accept escape **0.59**：近六成的空間錯誤會被靜靜放行。
> - clean false-positive（0.12 strict）也比 Stage 1（0.03）高，Stage 2 判定整體更嘈雜。

### Stage 1 執行環境（3090 專屬設定，踩過的坑）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 VLLM_PLUGINS="" \
  python judge_stage1.py --dataset daily_used --limit 25
```

- **`VLLM_PLUGINS=""` 是必要的**：vLLM 啟動時會自動載入已安裝的 entry-point plugins，
  `mm` env 裡的 **paddlex 註冊了一個，import paddle 會在它自己的 `monkey_patch_variable`
  裡 SEGFAULT**——engine 連一行 log 都印不出來就死。腳本已用 `os.environ.setdefault` 內建此設定。
- **記憶體**：8.8B 參數 bf16 ≈ 17.6 GB，24 GB 卡只剩 ~6 GB 給 vision tower／sampler／KV。
  vLLM 啟動 profiling 會跑 dummy sampler（`max_num_seqs × vocab 152k` 的 logits sort），
  預設值會 OOM。已設 `max_num_seqs=8`、`enforce_eager=True`、`gpu_memory_utilization=0.92`。
- `max_tokens` 預設 1024：640 會把四段 reason 從 JSON 中間截斷，guided decoding 救不回來，
  會變成 `judge_failure`。

**實測（25 objects / 91 samples，2026-07-28）**：485.8 s wall（含 ~150 s 載入）
→ 穩態約 **3.7 s/sample**、**0 judge_failure**。
全量 280k samples 單卡約需 **12 天** → 佐證 §12 的分層抽樣結論。

### Stage 1 corruption test — 分類 v3（2026-07-31，對齊 judge schema v2.0）

四類，每類 25 筆；**每筆與自己的 clean original 成對評分**。
流程：`corrupt.py build` → **人工確認**（排除修改後可能仍合理、或同時含多種錯誤者）
→ `judge_stage1.py --manifest` → `corrupt.py score`。

每一類都對應**唯一一個結構化欄位**（target axis）。這是 v3 的核心改動：
偵測率不再只問「有沒有被標壞」，而是問「**有沒有標在對的軸上**」。

| 類別 | 內容 | target axis（strict 判準） |
|---|---|---|
| `implausible_task` | 保留物件，把 task 改寫成**該物件無法完成**的操作。模板點名一個具體機構（hinged door / threaded cap / sliding drawer / zipper / hinge），只用在該機構明確不存在的物件上 | `task_plausibility.score == 0` |
| `role_function_conflict` | 只改一手的 primitive role，使 role 與**自己的 function 文字**明顯矛盾，**其餘欄位完全不動** | 該手 `role_function_consistency == "conflict"` |
| `contact_conflict` | contact region 移到該物件明確沒有的大型部件 | 該手 `contact_region_evidence == "absent"` |
| `bimanual_conflict` | 真正無法共同支持 task 的雙手配置 | `bimanual_validity.label == "invalid"` |

> **改名**：`role_function_task_conflict` → **`role_function_conflict`**。此類只改 role 並檢查
> role↔function 的文字一致性，不再直接評估 task。舊名連同 `role_task_conflict`、
> `contact_region_implausible` 一併從程式與 expected-tag 對照中移除。
> 注意改名會改變該類的 per-type RNG stream（`Random("<type>|<seed>")`），抽到的樣本與 v2 不同。
> `wrong_object_task`（移植其他物件的 task）仍**不在本測試中**。

manifest 每筆新增 `target_axis` 與 `hand`（被修改的那一手；task/雙手類為 `null`）。
scoring 讀 target axis 時**只看被改的那一手** —— 若允許「任一手」命中，
另一手被誤傷反而會被算成偵測成功。

**主要指標**（六項，`error-tag hit rate` 已降為 diagnostic）：
1. **strict detection rate** — 任一軸被判壞（score 0 或 invalid）
2. **lenient detection rate** — 任一軸不完美（score ≤1 或 label ≠ valid）
3. **target-field detection rate** — target axis 上的 strict／lenient。
   **`uncertain` 只計入 lenient，不計入 strict。**
4. **Auto-Accept escape rate**
5. **clean false-positive rate**
6. **core score / label change**（clean − corrupt）
7. **cross-axis spillover rate** — corruption 是否誤傷其他軸。
   採**成對**定義：某個 off-target 軸在 clean 上是 `ok`、在 corrupt 上變成
   `bad`（0 / absent / conflict / invalid）才計為 spillover，否則樣本本身既有的弱點
   會被誤算到 corruption 頭上。理想結果是 target axis 命中而 spillover = 0，
   代表 judge 能**定位**問題而不只是感覺到不對勁。

`corrupt.py score` 只接受 `judge_version == "qwen3vl-visual-semantic-v2.0"` 的紀錄，
其餘直接丟棄並印出計數 —— v1 紀錄沒有 `role_function_consistency`，
混進來會讓每一個 target axis 都靜靜地讀成未命中。
輸出目錄改為 `quality_evaluation/corruption_v2/`，**舊的 `corruption/` 原封保留**。

#### 結果 v3（8B，judge v2.0，100 corruptions + 100 clean，0 judge_failure，**人工確認前**）

`corruption_v2/`，2026-07-31，200 judgements / 1667 s（8.34 s/sample）。

| 類別 | n | strict | lenient | **tgt-strict** | tgt-lenient | **AA escape** | core Δ | **spillover** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| implausible_task | 25 | 0.96 | 0.96 | 0.88 | 0.88 | 0.04 | +1.81 | **0.96** |
| **role_function_conflict** | 25 | **0.16** | 0.48 | **0.12** | **0.16** | **0.52** | **+0.16** | 0.08 |
| contact_conflict | 25 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | +1.09 | **0.88** |
| bimanual_conflict | 25 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | +1.29 | **0.92** |
| OVERALL | 100 | 0.78 | 0.86 | 0.75 | 0.76 | 0.14 | | **0.71** |

clean originals：false-positive strict 0.06 / lenient 0.19，auto-accepted 0.81。

**結論一：新增 `role_function_consistency` 欄位並沒有解決 role↔function 盲點。**
target-strict 0.12、AA escape 0.52 —— 與 v2 舊 schema 的 0.12 / 0.64 幾乎相同。
把判斷變成必填欄位、強迫模型明文表態，**不足以讓 8B 真的去讀 `role` 欄位**。
（v3 的 `role_function_conflict` 與 v2 的 `role_function_task_conflict` 因改名而抽到不同樣本，
兩者是獨立抽樣下的同一結論，不是同一批樣本的重測。）

**結論二：8B 偵測得到問題，但無法定位。整體 spillover = 0.71。**
其他三類 target axis 幾乎滿分（0.88–1.00），代價是**同時把其餘三個軸一起打壞**：
contact 被改壞時有 19/25 連 `role_function` 也變 conflict、11/25 連 `bimanual` 也變 invalid；
task 被改壞時 24/25 連 bimanual、24/25 連 role_function 一起崩。
這是全有全無的判斷模式 —— 一旦覺得樣本不對，四個軸同步歸零。
因此 **error tag 目前無法用來指出「錯在哪個欄位」**，只能指出「這筆有問題」。

**結論三：`role_function_conflict` 的 spillover 低（0.08）不是優點**，
而是「幾乎沒反應」的副產物 —— 沒有偵測到，自然也不會誤傷其他軸。
spillover 只有在 target 命中率也高時才有意義。

⇒ 正式實驗必須換 30B-A3B，用**同一份 `corruption_v2/manifest.jsonl`** 重測；
若 30B 的 spillover 仍在 0.7 量級，Stage 3 的欄位級 gating 就不能建立在 tag 定位上。

#### 30B-A3B vs 8B（同一份 manifest，v2.0 prompt，2026-08-03）

`Qwen3-VL-30B-A3B-Instruct`（58 GB，13 shards，本次才下載）在 **GPU0（RTX PRO 6000, 97.9 GB）**。
引擎設定與 8B 完全相同（`max_num_seqs=8`、`enforce_eager`、`max_model_len=8192`、
`gpu_memory_utilization=0.92`），唯一差別是下面這個解碼護欄。

**踩到的坑：guided decoding 的空白迴圈，以及兩個「靜默失效」的關閉方式。**
30B smoke test 8 筆就有 1 筆 judge_failure，raw 內容是
`{"task_plausibility":` 之後無限吐 `\t\n` —— JSON grammar 允許無界空白，
greedy 解碼會卡在裡面燒完 max_tokens，retry（temp 0.2）也救不回來。
vLLM 0.11 關掉它**只有一種寫法有效**：

```python
LLM(..., structured_outputs_config=StructuredOutputsConfig(
        backend="xgrammar", disable_any_whitespace=True))
```

兩個看起來理所當然的寫法都**不會報錯、也不會生效**：
(1) `SamplingParams.guided_decoding=GuidedDecodingParams(disable_any_whitespace=True)`
——V1 引擎的 structured output 是引擎層級設定，per-request 參數被忽略；
(2) 已 deprecated 的 `guided_decoding_disable_any_whitespace=` kwarg。
兩者事後引擎 log 都仍印 `StructuredOutputsConfig(backend='auto', disable_any_whitespace=False)`。
**backend 也必須明寫 `xgrammar`** —— 在 `backend='auto'` 下這個旗標會被 pydantic 直接拒絕。
開啟後 30B smoke 8/8 通過，且速度快 2.4×（22.5 → 9.4 s/sample，省下空白 token）。
CLI 旗標為 `--disable_ws`，**預設關閉**，以免動到既有 8B 結果。

**因此必須跑控制組**：`--disable_ws` 不是中性的（它同時改變 8B 的數字），
所以另外用 8B 在同樣旗標下重跑一次，才能把「模型差異」與「解碼護欄差異」分開。

| target-strict | 8B v2.0 | 8B v2.0 **控制組**<br>(+ws guard) | **30B** v2.0<br>(+ws guard) |
|---|---:|---:|---:|
| implausible_task | 0.88 | 0.92 | **0.96** |
| **role_function_conflict** | 0.12 | **0.04** | **0.04** |
| contact_conflict | 1.00 | 0.96 | 0.80 |
| bimanual_conflict | 1.00 | 1.00 | **0.24**（lenient 0.72）|
| OVERALL target-strict | 0.75 | 0.73 | 0.51 |
| OVERALL target-lenient | 0.76 | 0.76 | 0.65 |
| role_function **AA escape** | 0.52 | 0.68 | **0.72** |
| clean FP strict / lenient | 0.06 / 0.19 | 0.05 / 0.07 | 0.05 / 0.14 |
| clean auto-accepted | 0.81 | 0.93 | 0.86 |
| **OVERALL spillover** | 0.71 | 0.72 | **0.53** |
| judge_failure / 200 | 0 | 1 | 1 |

執行成本：30B 2383 s（11.92 s/sample）＋載入 129 s，**峰值 VRAM 88.1 GB**
（= `gpu_memory_utilization=0.92` 的保留量；權重本身約 58 GB，KV cache 佔其餘）。
8B 控制組 887 s（4.43 s/sample）於 GPU3。**30B 約為 8B 的 2.7 倍推論時間、4 倍顯存。**

**結論一：30B 沒有改善 role↔function 判斷。** 在同一解碼設定下兩者都是 **0.04**
（25 筆只抓到 1 筆），AA escape 甚至更高（0.72）。逐例看機制完全相同 ——
role 被改成 `slide`、function 仍寫 "stabilize the knife against rotation"，
30B 的理由寫成「the role of sliding to maintain contact and provide counter-force」，
**把 role 揉進 function 的語意裡合理化掉，而不是拿兩者互相檢驗**。
這不是模型容量問題，換更大的同家族模型無效。

**結論二：30B 的定位能力確實較好（spillover 0.72 → 0.53）**，
`role_function_conflict` 更是零 spillover。代價是它整體更保守：
bimanual_conflict 只有 0.24 判 `invalid`（另外 0.48 判 `acceptable`，故 lenient 0.72），
contact 也從 0.96 掉到 0.80，clean lenient FP 從 0.07 升到 0.14。
**30B 換來的是「比較不亂扣分、錯了比較知道錯在哪」，不是「抓得比較準」。**

**結論三：解碼護欄本身會動到結果**（8B role_function 0.12 → 0.04、
clean lenient FP 0.19 → 0.07），所以跨版本比較必須固定 `--disable_ws`。
既有 8B v2.0 / v2.1 兩份結果都是**未開**護欄產生的。

⇒ role↔function 這個軸不能靠換模型或改 prompt 解決，需要換**機制**
（例如把 role↔function 拆成獨立的一次純文字呼叫，不與視覺判斷同批），這是下一步。

#### v2.1 prompt 強化的 A/B 對照（同一份 `corruption_v2/manifest.jsonl`，8B）

v2.1 只改 SYSTEM prompt（直接物理效果的判準 + 5 個 worked examples），schema 與
`derive_error_tags()` 完全不變 → 兩次結果結構可直接比較。
判斷檔分別存為 `judged_v2.0.jsonl` / `judged_v2.1.jsonl`，
以 `corrupt.py score --judged <file> --judge_version <ver>` 分別計分。
v2.1 有 **1 筆 judge_failure**（bimanual_conflict，n 因此為 24）。

| 指標 | v2.0 | v2.1 | Δ |
|---|---:|---:|---:|
| **role_function target-strict** | 0.12 | **0.20** | +0.08 |
| **role_function target-lenient** | 0.16 | **0.28** | +0.12 |
| role_function AA escape | 0.52 | **0.60** | +0.08（**變差**）|
| role_function spillover | 0.08 | 0.20 | +0.12 |
| clean FP strict | 0.06 | **0.04** | −0.02 |
| clean FP lenient | 0.19 | **0.11** | −0.08 |
| clean auto-accepted | 0.81 | 0.89 | +0.08 |
| OVERALL target-strict | 0.75 | 0.74 | −0.01 |
| OVERALL spillover | 0.71 | 0.76 | +0.05 |

**逐筆配對**（25 筆 role_function_conflict，看被改那一手的判定變化）：
`consistent→consistent` 17、`consistent→conflict` 2、`conflict→conflict` 2、
`consistent→uncertain` 2、`uncertain→conflict` 1、`conflict→consistent` 1。
**淨增 +3 筆命中，n=25，落在雜訊範圍內。**

**結論：prompt 強化沒有解決盲點。** 25 筆中仍有 17 筆連判定都沒動，target-strict 0.20 距離
其他三類的 0.88–0.96 仍差一個量級。真正改善的是 clean false-positive（lenient 0.19→0.11），
即 v2.1 的判斷比較不會亂扣分，但**沒有讓 8B 開始讀 `role` 欄位**。

**結構性發現：`role_function_consistency` 不在 Auto-Accept 判準內。**
README §7.1 的 Auto-Accept = 三個 score 皆 2 且 label ≠ invalid，**完全不看這個欄位**，
所以此軸再準也無法降低 AA escape。實測 v2.1 判為 `conflict` 的 5 筆恰好全部沒有 auto-accept，
但那是因為 8B 判 conflict 時會**連帶**把該手 score 打成 0（與 spillover 0.76 同一現象），
**不是 gate 擋下來的**。若換上定位更精準的模型（判 conflict 但 score 維持 2），
這個洞就會實際漏樣本 → **Stage 3 的 Auto-Accept 條件必須把 `role_function_consistency`
納入，這是待辦事項。**

#### 【已淘汰】v2 結果（8B，舊 schema，100 corruptions + 100 clean，0 judge_failure，人工確認前）

> 以下數字產自 judge v1.0（model 自行生成 error_tags、無 `role_function_consistency`），
> **不可與 v3 結果混用或比較**，保留僅為記錄該次失效模式。

| 類別 | n | strict | lenient | **AA escape** | score drop |
|---|---:|---:|---:|---:|---:|
| implausible_task | 25 | 0.96 | 0.96 | 0.04 | +1.81 |
| **role_function_task_conflict** | 25 | **0.12** | **0.36** | **0.64** | **+0.15** |
| contact_conflict | 25 | 0.96 | 1.00 | 0.00 | +1.17 |
| bimanual_conflict | 25 | 1.00 | 1.00 | 0.00 | +1.51 |
| OVERALL | 100 | 0.76 | 0.83 | 0.17 | |

**clean originals：false-positive strict 0.03 / lenient 0.12，auto-accepted 0.88**
→ detection 是真的在分辨，不是一律給低分。

**結論：8B 只能作開發與除錯，不得用來產生正式品質結論。**
`role_function_task_conflict` 的 **Auto-Accept escape rate = 0.64**——近三分之二的
role↔function 矛盾會被靜靜地放行，而其他三類是 0.00–0.04。與 07-24 的 7B pilot 同一失效模式。
逐例檢查出機制：**judge 直接讀 `function` 的語意、完全略過 `role` 欄位**——
role 改成 `slide`、function 仍寫「stabilize…」時，它整段理由只在談「施加穩定力是合理的」，
等於把 role 當不存在。這一類 corruption **只動 role、其餘欄位一字未改**，
因此它就是這個失效模式的乾淨對照。

error-tag（diagnostic）：implausible_task 0.88、contact 0.76、bimanual 0.32、
role_function **0.00**。tag 鑑別度不足（8B 曾在 bimanual 樣本上大量誤發 role 相關 tag），
**目前只有 score 可信**。

正式實驗必須換 30B-A3B，並用**同一份 manifest** 重測才能比較。

---

判定分布（8B pilot，**下限值、不進論文**）：task_plausibility 2/1/0 = 84/3/4；
bimanual valid/acceptable/invalid = 84/2/5；hand A 2/1/0 = 78/6/7；hand B = 82/5/4；
不會被 Auto-Accept 的樣本 13/91（14%）。error tags 有實際觸發
（`contact_region_implausible` 9、`role_task_conflict` 7、`implausible_task` 4、`bimanual_conflict` 4）
——8B 在此小樣本上**沒有**重現 07-24 pilot「對 role↔function 矛盾全盲」的行為，
但這要等 corruption test 才算數。

**輸入（2026-08-03 起）**：直接讀 `data_generation/outputs/stage1/{ds}/stage1_all.json`，
**不再走 `data_verification/data/` 快照**——快照當時 md5 相同，但沒有任何機制保證它一直相同，
指向生成端就不可能對到過期資料。

**輸出（2026-08-03 起）**：`data_verification/outputs/stage0/{dataset}/`
（舊的 `quality_evaluation/stage0/` 是 pilot 位置，不再寫入）

| 檔案 | 內容 |
|---|---|
| `audit_results.jsonl` | 每 sample 一筆完整記錄 |
| `audit_summary.json` | 聚合；warning 與 diagnostic 分開列，另有 unique-affected 與 flag-occurrence 兩套計數 |
| `hard_failures.jsonl` | 全部 hard failure sample 清單 |
| `metadata_corrections.jsonl` | 全部需修 metadata 的 sample 清單 |
| `orphans.json` | orphan query dirs + orphan object dirs |

```bash
python audit.py --dataset daily_used   --workers 24       # 全量
python audit.py --dataset electronics  --workers 24
python audit.py --example --limit 300                     # 舊 fixture（只測管路）
```

**效能實測（2026-08-03，24 workers）**：daily_used 251,390 samples / 167 s ≈ 1,506/s；
electronics 28,674 samples / 14 s ≈ 2,009/s → 全量 280,064 samples **約 3 分鐘**。
（舊值 12 workers ≈ 735/s ≈ 6 分鐘。）Stage 0 純 CPU/I-O，不佔 GPU。

**正式全量結果（2026-08-03）**：

| | daily_used | electronics |
|---|---:|---:|
| objects / expected samples | 57,441 / 251,390 | 5,528 / 28,674 |
| present / not_generated | 250,030 / 1,360（306 objects 未生成） | 28,656 / 18（4 objects） |
| orphan query dirs / object dirs | 0 / 0 | 0 / 0 |
| schema_pass | 247,328（98.38%） | 28,403（99.05%） |
| hard failure | 804（全為 `all_zero_A/B`；19 筆 A、B 皆零） | 57（3 筆皆零） |
| metadata_correction | 1,912 | 197 |
| warning（不含 diagnostic） | 32,105 | 4,187 |

hard failure 100% 是 `all_zero_A/B`，沒有任何 `missing_qdir` / `skipped_by_generator` /
檔案損毀——**已生成的 object 目錄，query 目錄一個都沒少**。
（抽查 `aae6968d.../q2_compress_the_cushion`：`hitB=35` 但 `scoreB.max()=0.0`，
確認是 partition/prune 真的把 B 打成全零，**不是 audit 誤判**。）
metadata_correction 主要是 `coordination_invalid` 與 `role_not_core8_A`。

> **設計要點（別回頭改壞）**：audit **以 stage1 為權威樣本清單**，不是走訪 stage2 目錄。
> 因為 `stage2_v2.py:264` 會靜默跳過 roles/molmo_queries 不等於 2 的 query，
> 走訪目錄會完全看不到它們而報出假的滿分。
> 另外 `expected_samples()` 會回報 **orphan query dirs**、`orphan_objects()` 會回報
> **orphan object dirs**（磁碟上有、stage1 完全沒列的整個 object 目錄）——
> 這是 stage1↔stage2 版本漂移的偵測器；舊 fixture 就是靠它抓出 662/779 objects 漂移，
> 否則會被誤記成 49% 的 generator skip。

### 待決

- [x] Stage 1 / Stage 2 checkpoint → **pilot = `Qwen3-VL-8B-Instruct`，之後換 30B-A3B**（§11.3）
- [x] `Qwen3-VL-8B-Instruct` 已下載（17 GB，2026-07-27）
- [ ] **A/B 共用單一 τ_vis 的矛盾**（§4.1.1）——Phase 2 必須裁決
- [ ] **Stage 1/2 抽樣規模與分層方式**（§12）
- [ ] τ_vis 值（Phase 2）
- [ ] warning 門檻正式值（現為 PROVISIONAL）
- [ ] corruption strict/lenient 攔截門檻（Phase 3，舊 D14）
- [ ] near-duplicate threshold（object embedding）與 τ_task（task semantic）
- [ ] human-verified test split 規模（建議 30–50 objects）
- [ ] `data_verification/` gitignore vs commit
- [ ] Stage 2 生成完成後重新對帳 object/pair coverage
