# Robot-executable role vocabulary — grounding & recommendation

Source: 6-angle literature sweep + synthesis (workflow `wf_62fab023-45d`, 2026-06-05).
85 candidate roles distilled across 49 sources (VLA models, primitive libraries, manipulation
benchmarks, bimanual papers, affordance/grasp taxonomies, skill ontologies).

## Design lens (two tests that cut the set cleanly)

1. **Executable-primitive test** — is the verb a *named controller* in a real robot/VLA source
   (RAPS, MAPLE, HACMan++, BEHAVIOR-1K, Meta-World, RLBench, CALVIN, ManiSkill, Open-X, etc.)?
2. **Role-not-task test** — is it what *one hand* does, vs a composite task?
   `open` = hold + rotate, `fold` = hold + grasp/pull, `carry` = hold + hold. Tasks are NOT roles.

A role = `{ verb, target part, physical function }`, hand-agnostic, executed simultaneously with the
other hand's role. Inter-hand *alignment / opposition* lives in the separate free-form `relation` field.

## Verdict on the current 12

| current verb | verdict | why |
|---|---|---|
| hold | **keep** (narrow) | backbone stabilizer; define as *static prehensile maintenance* (grip + immobilize, zero net motion) — distinct from new `grasp`(acquire) & `support`(non-prehensile). Krebs, Stabilize-to-Act/BUDS, ALOHA, 2HANDS, PerAct2 |
| support | **keep** (sharp def) | weakest as a named skill; keep ONLY as *non-prehensile bear-weight from below* (palm under a tray), else merge into hold. UMD/IIT-AFF part-affordance label, ALOHA "support from dropping" |
| push | keep | canonical non-prehensile lateral translate; every benchmark + RAPS/MAPLE/HACMan |
| pull | keep (def) | *grasp + retract* (−direction); RAPS, Meta-World handle/lever-pull, BridgeData open-drawer. Use `pull` not the task-word `open` |
| press | keep (distinct) | *localized normal force into a yielding part* (button/switch) — distinct from push (lateral). Meta-World button-press, RLBench press_switch, BEHAVIOR TOGGLE |
| lift | keep | core whole-object upward role; RAPS, RT-1 (pick=lift), CALVIN, PerAct2 "both arms lift" |
| rotate | **keep + parameterize** | make it ONE *axis-parameterized* rotation that absorbs `tilt` + twist (RAPS factors rotation by axis param θ). Axis arg = own-axis twist (knob/cap, intra) vs horizontal-axis tilt/pivot/reorient (inter). Highest-value cleanup |
| slide | keep (def) | *translate along a linear/prismatic track* vs free-space push/pull. RAPS "shift", CALVIN move_slider, RLBench/Meta-World slider |
| **tilt** | **MERGE → rotate** | no VLA/primitive source names `tilt` as a skill; it's rotate about a horizontal/edge axis. Near-synonym weakens grounding |
| **align** | **DROP → relation field** | not an executable per-hand skill — it's a GOAL/relation between the two targets (you align BY moving/rotating). Belongs in the coordination `relation` field (per locked design) |
| insert | keep | best-grounded; clean asymmetric pair (hold receptacle + insert peg). Meta-World peg-insert, ManiSkill assembly, RLBench insert_* |
| **separate** | **DROP → two `pull` + opposing relation** | not a single-hand primitive — a symmetric two-hand *outcome*. Express as pull+pull with an opposing/diverging `relation`; for un-gripping use extended `release` |

## Two real ADDITIONS (gaps in the current 12)

- **grasp** — *acquire* a force-closure grip (close-gripper). The universal entry primitive across ALL
  libraries (RAPS, MAPLE, HACMan++, BEHAVIOR-1K GRASP, Open-X canonical cluster, GR00T, π0). It's the
  acquiring half before any actuation and the prehensile half of a co-carry.
- **place** — deposit at a target pose + release. The single most common *terminal* verb in the whole
  corpus (RT-1/RT-2/OpenVLA/Octo, CALVIN place, RLBench put_*, LIBERO Put, ManiSkill, BEHAVIOR PLACE_*).
  Transport had no terminating role-verb without it.

> Caveat for OUR formulation: `grasp`/`place` are *temporal phases* of a transport (grasp→lift→place),
> whereas our roles are TWO SIMULTANEOUS coordinated contacts. In a simultaneous snapshot a co-carry is
> `hold`+`hold` / `support`+`lift`, not grasp/place. So grasp/place are most useful if we later model
> task *phases*; for the simultaneous-2-role schema they are optional.

## Recommended sets

**Grounded core for our simultaneous 2-role schema (9):**
`hold, support, push, pull, press, lift, rotate, slide, insert`
= current 12 − align − separate − tilt(merged into rotate).
Stabilizer roles: `hold` (prehensile immobilize), `support` (non-prehensile bear-weight).
Actuator roles: `push, pull, press, rotate, slide, insert, lift`.

**Full robot-primitive core (11):** add `grasp` + `place` (transport-phase completeness).

**Extended (add only if the task distribution demands):**
`release` (handover/disengage, inverse of grasp), `pour` (container dispensing), `lower` (controlled
set-down, RAPS "drop"), `scoop` (granular transfer), `clamp` (enclose-from-opposite-sides co-carry,
the BiNoMaP "wrapping" role the core can't express).

**Explicitly NOT roles** (they are validation *tasks* whose per-hand decomposition the core reproduces):
`open` (= hold + rotate/pull), `fold` (= hold + grasp/pull), `carry` (= hold + hold), `pick` (= grasp + lift).

## Family mapping

- **intra_object** = STABILIZER (`hold` / `support`) + ACTUATOR (`push/pull/press/rotate/slide/insert`).
- **inter_object** = transport chain `grasp/support → lift → place` and `rotate`(reorient axis) for
  flip/pivot; symmetric co-carry = `hold+hold` / `lift+lift` (two identical roles, disambiguated
  downstream by spatial clustering per locked decision 2 in [[bimanual-task-proposal-design]]).
