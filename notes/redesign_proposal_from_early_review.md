# Redesign proposal — from the FULL 109-object review of 12B+thinking (vllm)

**Status: PROPOSAL for user review. NOT applied.** Prompts backed up at
`pipeline/stage1_task_role_assemble.py.bak_pre_levers`. Full review JSON at
`review/full_review_109_12b_thinking.json`; early 19-obj review at `review/early_review_19obj_levers.txt`.

## Result (109 objects, 373 queries, adversarial multi-judge: judge→verify→synthesis, 219 agents)
**good 208 / weak 48 / wrong 117 = 55.8% good** — vs the ~52% prior baseline = **+3.8 pts**.
Codes: **W3=32** (part-hallucination, biggest), **W8=21** (pour/tilt coordination), **W1=20** (task-not-real),
**W7=18** (region unreachable/mismatch), K1=19, K4=15, **W4=14** (verb-direction), K3=7, K2=7, W2=7, W5=3, W6=2.

**Key insight:** 12B+thinking only added +3.8 because the losses are **UPSTREAM (W3 component
hallucination)** and at **GROUNDING (W7 reachability)** — places extra reasoning budget can't help.
The 4 levers below target ~147 of the 165 non-good queries → the clear path to **>80%**.

## The 4 systemic levers — all principled REASONING-GUIDES (never hard task-name rules)

### Lever 3 (HIGHEST value, UPSTREAM) — stage0 component extraction: evidence-gated operability
`prompt/get_component_prompt.json` "parse". 32 W3 are ALL here: a fabricated part (lid on a sealed/grate
top, latch with no hasp, port-cap, a folding rule mis-parsed as a spring tape with a "housing") poisons
every task built on it — a **multiplier** (2–4 downstream wrongs per bad part).
- **Principle:** to list a lid/door/drawer/cap/latch/handle you must point to the visible feature that
  proves it operates — a separation seam / hinge line / thread / graspable protrusion / free edge. A flush
  continuous surface, a printed mark, or a label is NOT operable (describe as fixed surface or omit).
- **Object-identity sanity:** first state what the geometry actually IS from the views, independent of the
  category name; if the name implies a mechanism the geometry doesn't show, trust the geometry.
- Persist a seal/closure state per opening (taped/sealed/integral vs free) so downstream "open/tip-out" on
  a taped/apertureless box is flagged non-self-contained.

### Lever 1 (op-plan, STEP 1/2) — payload-vs-frame + joint-state (the tip-out fix you AGREED to)
W8 (21) + W4 (14). Ask for every task: **what is the moving element (payload) vs the stable frame (held
fixed)?** One hand owns the frame, the other the motion (unless a pure symmetric carry of one rigid payload).
- pour/empty → contents are the payload, body is the frame → one hand anchors base, other tilts → stabilize
  +actuate EMERGES (no "tip-out⇒hold+tilt" lookup).
- verb must produce force in the goal direction: an inward push on a vertical face can't lift; opposing-face
  lifting needs grip/clamp (fixes W4 push-as-lift). Edit the ACTIVE `OP_PLAN_PROMPT`, not dead `ROLE_DECOMP_PROMPT`.

### Lever 2 (ground, contact_region) — resting-pose reachability + specificity
W7 (18) + K1 (19). **The object sits on a surface in its rest pose; a hand can only contact a face EXPOSED
in that pose — the underside/base it rests on is occluded, never place a contact there** (kills the "bottom
surface" stabilize cluster). + name the contact by a feature that distinguishes it from neighbours (which
edge/wall/corner); if the region would equally identify three spots it is too generic.

### Lever 4 (brainstorm + rank) — causal task-reality + self-containment
W1 (20) + W2 (7). Replace the keyword checklist with a **causal narration**: (1) does a real person do this
bare-handed routinely to THIS object? (2) trace to the end state — does it consume any object not in the
scene (recipient vessel to pour into, tube, clothespin, loose garment)? if yes → not self-contained → drop.
(3) is the goal a genuine state change, or is "after" == "before" (spinning a symmetric wheel)? drop ill-defined.

**Lower-tier (note, don't over-invest):** K4=15 (derive why_bimanual FROM the chosen coordination scheme so
an asymmetric split can't be mislabeled "symmetric"); W5=3 (folds into Lever 1's load-bearing reasoning).

## Locked-decision filter (so the redesign doesn't override you)
- KEEP the universal "pick up the X" on every object + accept its W2 (your 2026-06-06 lock). Lever 4 drops
  only *brainstormed* non-bimanual tasks (latch on a heavy stable chest), NOT the fixed pick-up.
- Clean the hedge-boilerplate in `fixed_inter_task` ("support base + steady top for small OR grip two sides
  for large") — you already banned boilerplate; replace with one honest line, keep the pick-up.

## Why NOT auto-applied + restarted (judgement call)
Lever 3 is upstream (stage0) and a multiplier; the others touch sophisticated, opinion-laden prompts with a
locked decision (universal pick-up) and dead-code traps. Auto-rewriting + restarting risked over-reaching
your locks — what you've told me to avoid. So: 12B+thinking baseline done (55.8%), 26B-A4B no-thinking A/B
running, and this proposal awaits your approve/adjust before the next run. Recommended order to apply:
Lever 3 first (biggest multiplier, upstream), then 1/2/4.
