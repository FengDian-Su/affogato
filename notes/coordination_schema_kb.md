# Bimanual Coordination — Physical Common-Sense Manual (KB v0)

## What this is

A small **manual of physical common sense** for how humans coordinate two hands to manipulate an
object. It is **not** a lookup table. When the generator decomposes a high-level task into two
hand-agnostic role affordances, it reads this whole file and **reasons WITH these principles** — it
should generalize to objects not listed here, the same way a person uses known physics to handle a
new object. Roles describe what a *hand* physically does; a robot later maps each role to one of its
two end-effectors (hand-agnostic, no left/right).

How to use it (instruction to the LLM):
- Treat the schemas as physical intuitions, not rules to match 1-1.
- A task is genuinely bimanual only if it follows one of these physical logics (or a sensible
  combination/extension of them). If no principle plausibly applies to the object, it is fine to
  conclude the task is **not** bimanual.
- Per-object feasibility (is this specific object actually movable / does it have two separable
  graspable regions) is checked downstream by the verification gate — here, just reason about the
  physics.

## General principles (the physics chapter)

1. **Reaction needs an anchor.** Any action that applies a force or torque the environment cannot
   absorb (twisting, pulling, prying, peeling) needs a *second* contact to hold the object steady —
   otherwise the object just moves with the action. If gravity/friction/a mount already absorb the
   reaction, the second hand is unnecessary (then it is *not* bimanual).
2. **Two contacts must be physically independent.** The two regions should be parts that can move or
   be held separately — not two labels on the same rigid piece.
3. **Size/weight can force two hands.** An object too large or heavy to hold stably with one grasp
   needs two supporting contacts at distinct regions.
4. **One hand sets the frame, the other acts** (for asymmetric tasks): the holding/supporting hand
   fixes the object's pose so the acting hand can work precisely against it.

## Schemas

```yaml
- schema: rotate_to_open
  human_intuition: "要把東西轉開,另一隻手得壓住本體不讓它跟著轉"
  mechanism: 一個部件繞軸相對本體旋轉
  roles:
    - {role: hold,   function: 抵住反作用力矩, where: 軸附近最大可抓的剛體區}
    - {role: rotate, function: 施加力矩,       where: 蓋 / 旋鈕 / 閥}

- schema: hold_and_pull_open
  human_intuition: "拉開蓋子/抽屜/拉環時,另一隻手要把本體按住,不然整個被拉走"
  mechanism: 一個部件沿某方向相對本體平移/掀開
  roles:
    - {role: hold, function: 固定本體, where: 本體上穩固可抓處}
    - {role: pull, function: 施加拉力把部件帶開, where: 蓋 / 拉環 / 把手}

- schema: slide_open
  human_intuition: "推開滑蓋時,一手穩住機身,另一手把蓋平移"
  mechanism: 一個蓋/件沿軌道相對本體滑動
  roles:
    - {role: hold,  function: 固定本體, where: 機身}
    - {role: slide, function: 平移可動件, where: 滑蓋 / 門 / 卡榫}

- schema: pour
  human_intuition: "倒東西時,一手托住重量,另一手控制傾倒的角度與方向"
  mechanism: 容器需被支撐並受控傾斜
  roles:
    - {role: support, function: 承重並穩定, where: 把手 / 本體下部}
    - {role: tilt,    function: 控制傾倒方向, where: 本體上部 / 出口側}

- schema: co_carry
  human_intuition: "東西太大或太重,單手抓不穩,兩手得在兩端對撐著抬"
  mechanism: 剛體需多點支撐才能平衡舉起/搬運
  roles:
    - {role: support, function: 分擔載重並穩定, where: 一側 / 底部}
    - {role: support, function: 分擔載重並穩定, where: 對側 / 底部}

- schema: pull_apart
  human_intuition: "兩手各抓一邊往相反方向拉,才能撕開或分離"
  mechanism: 兩個可分離子件需反向力才會分開
  roles:
    - {role: pull, function: 施力, where: 一側邊 / 翼 / 半部}
    - {role: pull, function: 反向施力, where: 對側邊 / 翼 / 半部}

- schema: hold_and_press
  human_intuition: "用力按壓某處時,另一手要在背面/旁邊抵住,不然東西被壓跑或翻倒"
  mechanism: 按壓力需要對側支撐來抵消
  roles:
    - {role: hold,  function: 抵住按壓的反作用力, where: 背側 / 底座}
    - {role: press, function: 施加按壓力, where: 按鈕 / 表面 / 機構}

- schema: align_and_insert
  human_intuition: "把一個東西插進孔位時,一手扶住/對準接收端,另一手把件推進去"
  mechanism: 可插入件需對準並推入一個孔/槽
  roles:
    - {role: hold,   function: 固定並對準接收端, where: 有孔 / 槽的本體}
    - {role: insert, function: 把件移入孔位, where: 可插入的部件}

- schema: stretch_or_bend
  human_intuition: "兩手抓住兩端施反向力,把柔性/可彎物體拉直、撐開或折彎"
  mechanism: 可變形物體需兩端反向力才會變形
  roles:
    - {role: hold, function: 固定一端, where: 一端}
    - {role: pull, function: 對另一端施反向力使其變形, where: 另一端}
```

## Notes

- Vocabulary is aligned to the team doc's role set (hold/support/lift/push/pull/press/rotate/slide/
  tilt/insert/separate/align) and to the bimanual-manipulation taxonomy of Krebs & Asfour (RA-L 2022:
  coordinated, symmetric/asymmetric, stabilizing vs acting roles).
- The list is intentionally small and abstract. Add a schema only when it captures a *distinct
  physical logic*, not a new object.
- Human-demonstration video (later) is for **witness** ("humans actually coordinate this way") and
  **discovery** (finding a physical logic not listed here) — not for adding object-specific entries.
