# LLM-as-a-judge for task and role quality: what we learned

Qwen3.8-27B rating the task and role-assignment of released DUALingo pairs on the human review's
0/1/2 scale, built 2026-10-06/07 over 22 prompt versions. The code is
`data_verification/quality_review/qwen_judge.py`; every run, including each experiment, is in
`data_verification/outputs/quality_review/release1000/qwen38_27b/<version>/`, and each run folder's
`config.json` holds the exact prompts it used. This note records the path, what turned out to
matter, the designs that failed, and how we would run it again. The heatmap criterion, which the model has to
judge by looking, has its own note: `llm_judge_heatmap.md`.

## Goal and constraints

- The paper reports an "LLM-as-a-judge" score next to the human scores. The judge's scores should
  come close to the raters', without the judge looking tuned to agree with us.
- **The judge's raw greedy output is the label.** No calibration with token probabilities afterwards;
  every standard lives in the prompt (user's call, after a post-hoc rule was tried and removed).
- **Prompts stay short and general.** They rely on the model's everyday knowledge and describe the data;
  they do not list if-else rules or specific examples.
- The judge sees what a rater sees: the same eight 512 px RGB views (no hands), the object name, the
  task, and for role also both hands' roles.
- Reference: two raters (michaellee, FengDian-Su) on all 1,000 pilot pairs, object-disjoint dev 700 /
  test 300. Three more raters are planned.

## Final result (v22, one run on all 1,000 pairs, 0 failures)

| | Qwen | michaellee / FengDian-Su | Exact agreement Qwen vs ML / FS | Exact agreement ML vs FS | Gwet AC1 Qwen vs ML / FS | AC1 ML vs FS |
|---|---:|---:|---:|---:|---:|---:|
| Task, all | 93.2 | 95.8 / 95.8 | 88.3% / 86.3% | 88.9% | .874 / .853 | .882 |
| Task, dev / test | 93.8 / 91.8 | 96.1 / 96.4, 94.8 / 94.2 | | | | |
| Role, all | 96.4 | 98.9 / 96.0 | 94.4% / 91.3% | 92.7% | .942 / .909 | .924 |
| Role, dev / test | 97.5 / 93.7 | 99.2 / 96.6, 98.2 / 94.7 | | | | |

Score = mean label / 2 × 100. The test split was scored for the first time at v15. Later versions
were still chosen on dev, but test had been seen, so it is no longer a pristine held-out set.

The paper's numbers come from the v23 run. It used the same task and role prompts, ran together with
the heatmap criterion, and kept 991 / 1,000 task and 994 / 1,000 role labels identical. Greedy vLLM
output is not bit-exact across batch compositions. That run scores task 93.3 (AC1 .875 / .853) and
role 96.5 (AC1 .942 / .911).

## The path

Phase 1. Against one rater (michaellee), dev, scores task / role:

| Version | Change | Task | Role |
|---|---|---:|---:|
| v1 | rater rubric verbatim, 256 px contact sheet | 76.4 | 89.5 |
| v2 | + rater-practice guidance | 84.2 | 91.6 |
| v3 | short general question, 8 original views, a `parts_visible` field before the score | 88.3 | 79.7 |
| v4 | middle grade means "unclear", not "awkward"; roles need not be the best way | 91.7 | 92.1 |

Phase 2. A second rater finished, and both raters became the reference:

| Version | Change | Task | Role | Outcome |
|---|---|---:|---:|---|
| v5 | object name can be wrong, go by the views; 3D models fuse moving parts; 0 only if clearly wrong, unsure → 1 | 93.1 | 91.2 | kept for task |
| v5 + `decide()` | count a 0 only when p(0) ≥ 0.5 | 94.1 | 97.1 | removed: post-hoc |
| v6a, v6b | role: "picture it", or reason before score | | 93.7 / 92.1 | false alarms unchanged (57 / 53 vs 57), fewer catches |
| v7 | the stage-1 generator's verb glosses, hand patterns and task kinds pasted in | 89.6 | 88.6 | stricter |
| v5-noname | object name removed | 88.4 | 90.2 | Qwen misidentifies objects without it |

Phase 3. Wording search:

| Version | Change | Task | Role | Outcome |
|---|---|---:|---:|---|
| v8 | very short prompt, bare grades, no parts field | 85.9 | 80.1 | the model stops using 1 |
| v9a | "nothing clearly wrong" anchors, no parts field | 84.6 | 83.7 | |
| v9b | "nothing clearly wrong" anchors, parts field kept | 93.4 | 81.9 | "something seems off" pulled 115 roles to 1 |
| v5-think | thinking mode on | 91.4 | 93.0 | 6× slower, 25 runs past 8,192 tokens |
| v10 | "teach it like a small model": numbered steps, each its own JSON field; role starts with `plan` | 83.3 | 94.0 | role up; task's "does a real one have the part?" field → "no" → 0 |
| v11a, v11b | task in the step style (object / use fields) | 85.6 | | worse |

Phase 4. Role ablations, dev:

| Version | Change | Role | False alarms on both-Good pairs |
|---|---|---:|---:|
| v12 | the `plan` step only, no verb glosses or listed faults | 93.9 | 44 |
| v14a | + a "how to read a role" paragraph | 93.4 | 40 |
| v14b | "picture the given roles" instead of a plan | 91.7 | 59 |
| v15b | + affordance framing only | 93.4 | 48 |
| v15c | + "closest of eight verbs is acceptable" only | 94.4 | 39 |
| **v15a** | **both** | **96.9** | **18** |
| v16 (task) | the same affordance framing on task | task 93.4 | no change from v5 |

Phase 5. Frozen v15 on all 1,000 pairs:

| Split | Task | Role |
|---|---:|---:|
| dev | 92.9 | 96.9 |
| test | 91.5 | 90.2 |

Role did not transfer to test, for two reasons:
- **The test mix differs:** 39% pose tasks (dev 16%) and 30% co-actuate (dev about 0%).
- **The prompt partly overfit dev:** even within inter- and intra-object tasks, test flags ran about 4×
  dev's.

Phase 6. Task leniency, all 1,000 pairs:

| Version | Change | Task | Outcome |
|---|---|---:|---|
| v17b | a part that is small, hidden or not modelled separately is not missing; judge as if a real one's part were there | 93.5 | |
| v18c | + "the hands may be a person or robot of any size; pick up / move are asked of every object" | 93.3 | universal-task flags 6 → 0; kept |
| v19 | parts field: visible / not shown / absent | 94.4 | "not shown" still → 1 |
| v20 | Good grade: "even if the parts it needs are not shown" | 96.2 | rejected |

v20 was rejected because it reached the raters' score through leniency: it caught half as many of
the pairs the raters flagged (michaellee's 63: 18 → 9). The user's call was that the judge should
stay grounded on the object.

Phase 7. Role on dev, all 1,000 pairs:

| Version | Change | Role | Outcome |
|---|---|---:|---|
| v21b | + what lift / rotate / hold cover; "if the views are hard to see, rely on the name and everyday knowledge" | 96.5 | test 94.0 |
| v22 | each hand as fields ("touches X at …; action: rotate; purpose: …") instead of "rotate the handle …" | 96.5 | false alarms 33 → 27; handle-tilt flags 7 → 1 |

Final: v22, with the token cap raised from 300 to 1,024 (one answer had overflowed).

## What we expected the model to understand, and what it did instead

We started from the assumption that a 27B vision-language model reads an annotation the way a rater
does. Most of the work came from finding out where it does not. Each entry below gives what we
assumed, what the model actually did, and what we found and changed.

### Reading the object

1. **The object name and the views.**
   - *We assumed* the model would trust what it sees over a name when the two disagree.
   - *It actually* trusted the name. A cabinet with a drawer, named "Window", got "a window has no
     drawers". A lidded jar named "Pillow" got "a pillow cannot be opened".
   - *So we* wrote "the name can be wrong, go by the views" (v5).
   - *We then assumed* removing the name would force it onto the images. *It actually* misidentified
     ordinary assets from the views alone (a beaker became a waste drum, a remote a sign), and task
     fell to 88.4. The name helps more than it misleads, so we kept it with that caveat.
2. **3D models versus real objects.**
   - *We assumed* the model knows a 3D asset is a simplified object, and would judge "close the lid" on
     a teapot as a person does: teapots have lids.
   - *It actually* judged the literal mesh: "the lid is modelled as one solid piece, so it cannot be
     closed".
   - *So we* worked through four steps:
     - saying that moving parts are often fused helped only partly (v5);
     - "a part that is not modelled separately is not missing" moved some cases (v17);
     - making "not shown" its own answer option, next to "visible" and "absent", showed the model
       understood the difference, but it still scored "not shown" as "unclear" (v19);
     - writing acceptance into the Good grade fixed the score but made the judge lenient everywhere
       (v20), so we stopped there.

     The remaining gap is partly real: one rater also flags some of these cases.
3. **Who does the task.**
   - *We assumed* "pick up the toilet" or "move the conveyor belt" would be read as the dataset means
     it, as a task asked of every object.
   - *It actually* pictured one ordinary person: "a toilet is a fixed fixture", "far too heavy for
     two hands".
   - *So we* stated the embodiment (v18): the hands may belong to a person or a robot of any size, and
     pick up and move are asked of every object. Universal-task flags fell from 6 to 0.
4. **What the views contain.**
   - *We assumed* the model knew the views show only the object.
   - *It actually* gave 1 with "the views are 2D and do not show the hands" or "the views are too
     dark to verify".
   - *So we* said the views show only the object, with the hands described in words, and that it
     should rely on the name and everyday knowledge when the views are hard to see (v14, v21).

### Reading a role

5. **"Rotate the handle".**
   - *We assumed* "Hand A: rotate the handle, to tilt the teapot" reads as turning the teapot by its
     handle.
   - *It actually* read it as the handle itself rotating: "the handle is fixed and cannot be rotated".
     This persisted even when the prompt explicitly said a motion can move the whole object through
     the part held (v14a).
   - *So we* found two causes:
     - the model had no idea what a role is, which the affordance framing fixed (v15a);
     - our own input line read as an imperative with the handle as its object. Listing the same
       content as fields ("touches handle at …; action: rotate; purpose: …") cut these from 7 of 63
       to 1 (v22).
6. **The closed verb set.**
   - *We assumed* the model would see that eight verbs cannot name every motion, and would read
     "pull" to open a hinged lid, or "lift" to cradle a ball, as reasonable.
   - *It actually* checked whether the verb names the motion exactly: "pull is unworkable for a
     hinged lid", "lift is dynamic, cradling is static". Pasting the generator's full verb
     definitions made it stricter still (v7).
   - *So we* changed the question the score answers: is this a reasonable closest choice among the
     eight? Combined with the affordance framing, role rose from 93.9 to 96.9 (v15a). We later said
     what the most used verbs cover: lift also holds the object up or steady, rotate turns or tilts
     the whole object, hold keeps it still (v21).
7. **The holding hand.**
   - *We assumed* "Hand B: hold the body, to stabilize it against the rotation" is obviously the
     counter-force that lets Hand A turn the object.
   - *It actually* called it a contradiction: "the hands work against each other". Our own 0 grade
     had listed "the hands work against each other" as a fault, which invited the reading.
   - *So we* removed listed faults from the grades, and described hold as keeping the object still
     while the other hand works.

### Using the instructions

8. **A rubric.**
   - *We assumed* the human rubric, given verbatim, would make the model rate like a rater.
   - *It actually* became a fault-finder (task 76.4).
   - *So we* switched to one short question per criterion that leans on everyday knowledge (v3).
9. **The middle grade.**
   - *We assumed* the model would keep "OK" for minor issues, as the raters do.
   - *It actually* put any imperfection there, such as a workable but not ideal grip, or "something
     seems off".
   - *So we* defined the middle grade as "unclear" and said roles need not be the best way (v4).
10. **Explanations.**
    - *We assumed* that once we explained a convention ("read the verb with its purpose", "hold
      provides the counter-force"), the model would apply it.
    - *It actually* read the explanation but scored from its first impression. A "how to read a role"
      paragraph fixed 23 cases and broke 19 (v14a). In one teapot case its own plan said "grip the
      handle to rotate the body", and it still scored 0.
    - *So we* found that the score follows the structured field the model has just filled. Decisions
      had to move into fields, field options, the wording of the question and the input format, not
      paragraphs of guidance.
11. **"Clearly".**
    - *We assumed* "0 only if clearly wrong; if not sure, give 1" would keep doubtful cases off 0.
    - *It actually* chose 0 when it was nearly undecided, with p(0) about 0.4 against p(2) about 0.35.
      Its sense of "clearly" does not track its own uncertainty.
    - *So we* tried thresholding the probabilities afterwards. It worked, but we removed it, because
      the label must be what the model outputs. That pushed every fix back into the prompt.
12. **More thinking.**
    - *We assumed* letting the model think first would give better judgments.
    - *It actually* got worse on task, overthought 1.8% of items past 8,192 tokens (25 of 1,400), and ran 6× slower.
    - *So we* replaced free reasoning with one short field per axis: parts visible for task, a
      normal-way plan for role.
13. **Imagining the normal way.**
    - *We assumed* a "how would a person do this?" step would simply help understanding.
    - *It actually* helped role a lot, since the model stopped auditing the mesh. But it then treated
      its own plan as the only right answer ("should hold the neck, not the body"). The same kind of
      step made task stricter.
    - *So we* kept it for role only, and added "even if not exactly how you would do it" with the
      closest-choice framing.
14. **A shorter prompt.**
    - *We assumed* less structure would let common sense come through.
    - *It actually* turned binary with bare grades, and without the parts field its false alarms on
      task tripled (v8, v9a).
    - *So we* kept the look-first field and the graded wording.

## What mattered

1. **Human-vs-human agreement is the ceiling; measure against it first.**
   - The two raters agree on 89–93% of pairs but almost never on which few pairs are flawed. On dev
     task, each flagged about 39 pairs, and only 4 overlapped. That puts ordinal α near 0.
   - With a single reference rater, a prompt overfits that person: v4 had α .23 with one rater and
     .05 with the other.
   - Report exact agreement and Gwet's AC1 next to the human-vs-human value. α alone looks like
     failure on data that is almost all Good.
2. **Describe what the annotation is, not rules for judging it.** The largest single gain (role
   93.9 → 96.9) came from two sentences describing the data:
   - **Affordance:** a role marks where an embodied agent makes contact and what it does there. The
     touched region need not move or deform, and the object stays whole.
   - **Closed vocabulary:** the action is the closest of eight verbs, and the judge should rate it as
     a closest choice, not as an exact name for the motion.

   Either sentence alone did nothing; together they did. Explanations of how to read the fields
   (v14a), or long glosses from the generator (v7), did not help.
3. **A short structured field before the score is the strongest lever for a 27B non-thinking model.**
   - For task, `parts_visible` forces the model to look at the views: dropping it raised false
     alarms from 33 to 83.
   - For role, `plan` ("how would a person normally do this?") moves the model from auditing the 3D
     mesh to common sense: fused-part objections fell from 34 to 10.
   - Free reasoning (thinking mode, reason-before-score) did not help. The same field can also hurt
     on the other axis: a plan-like step on task made it worse.
4. **The model decides on the field it just filled, not on the guidance paragraph.** Once
   `parts_visible = no`, the score goes to 1 or 0 whatever the grades say. Two consequences:
   - Fixes have to be in the field's options or in the grade wording.
   - Pushing a field toward a fixed answer (v20) turns the judge lenient across the board.
5. **The input format is part of the prompt.** "Hand A: rotate the handle" reads in English as the
   handle being turned. Listing the same content as fields reduced those misreadings from 7 to 1 with
   no prompt change. Present inputs the way the human UI does, as separate fields rather than a
   sentence.
6. **Grade wording sets the distribution.**
   - Bare anchors ("2 = yes, 0 = clearly not") made the model binary.
   - A middle grade defined as "awkward" or "something seems off" drew hundreds of 1s.
   - What worked: "1 = unclear", "0 = clearly …", "if not sure, give 1", and "need not be the best way".
7. **The object name helps more than it misleads**, even though names are sometimes wrong. Without
   it, Qwen misidentifies many 3D assets: a beaker became a waste drum, a remote became a sign. Keep
   the name, and say it may be wrong.
8. **The model is near-indifferent on many pairs.**
   - One added sentence flips about 25 of 1,000 labels, and identical prompts move scores by ±0.3
     from run to run.
   - Treat differences under about 0.5 as noise.

## Designs that failed (do not repeat)

- **Pasting the human rubric or the generator's own prompt into the judge** (v1, v7). Long,
  rule-shaped text makes the model hunt for faults. More text means a stricter, more literal judge.
- **Post-hoc probability rules** (`decide()`, threshold calibration). They improve the numbers but
  cannot be defended as "an LLM judged it".
- **Removing the look-first field to shorten the prompt** (v8, v9a).
- **Thinking mode** for a judge on short items: no gain, about 6× slower, truncations at 8,192 tokens.
- **Task-side "checker" framing and identity fields** ("You check tasks…", `object` / `use` fields):
  stricter.
- **Leniency written as an instruction about the answer** ("Good even if the part is not shown"). It
  matches the score but loses detection, and looks like steering in an appendix.
- **Specific examples in the prompt:** avoided on purpose. Examples seed the judgments they name.

## If we did it again: how to write the judge prompt

1. **Start by describing what the annotation is.** Do not paste the rater rubric or the generator's
   prompt. Two kinds of description moved the scores the most, and we would write them on day one:
   - **The representation:** a role is an affordance annotation, telling an embodied agent where to
     make contact and what to do there. The touched region need not move or deform, and the object
     stays whole.
   - **The conventions the generator was bound by.** Ask the judge to rate within them, not against
     an ideal:
     - actions come from a closed set of eight verbs, so each is the closest choice;
     - pick up and move are asked of every object, and the agent may be any size;
     - 3D models fuse moving parts;
     - the views show only the object, with no hands;
     - the object name may be wrong.
2. **Ask one plain question per criterion, framed as feasibility rather than quality.**
   - Task: "does it make sense and is it meaningful for this object?"
   - Role: "could the task be done with these roles? It need not be the best way; judge the roles
     for the task as given."
3. **Write grades so the model's doubt lands in the middle.**
   - Use "2 = yes", "1 = unclear", "0 = clearly not", plus "if not sure, give 1".
   - Avoid a middle grade that names a quality ("awkward", "something seems off"): it fills up.
   - Avoid bare yes/no anchors: they make the judge binary.
   - Never write leniency as an instruction about the answer ("Good even if …"). It buys score with
     lost detection and reads as steering.
4. **Before the score, give one short structured field that makes the model take the right first step
   for that axis.**
   - Task: say whether the needed parts are visible, which grounds the judgment in the views.
   - Role: write how a person would normally do the task, which turns mesh-auditing into common
     sense.

   A field that helps one axis can hurt the other, so choose per axis. The score follows the field the
   model just filled ("parts: no" → 1 or 0), so the field's wording and options are where outcomes
   are decided. Fields worked; free reasoning (thinking, reason-before-score) did not.
5. **Present inputs the way a person sees them.** Each hand's role should be separate fields
   ("touches X at …; action: rotate; purpose: …"), not a sentence whose grammar changes the meaning
   ("rotate the handle" = the handle turns).
6. **Keep it short, and teach it like a small model.** Use short sentences, plain definitions of the
   data, and a field to fill in. Leave out if-else rules, lists of faults and examples. Every extra
   sentence shifts a model that is near-indifferent on many items, and long rule text turns it into a
   fault-finder.
7. **Ground on all the evidence a rater has.**
   - Give the original-resolution views and keep the object name.
   - Tell the model to rely on the name and everyday knowledge when the views are hard to see.
8. **When it disagrees with the raters, read its reasons and ask which part of the prompt let it read
   the data that way.** The answer is the data description, the field, the grades, or the input
   format. Fix that part in general terms; do not patch the example.

The final prompts (`qwen_judge.py`, v22) follow this structure. The role prompt is laid out as:
description of the annotation → the eight-verb convention and what the main verbs cover → what
the inputs are → steps (plan, parts, score, reason) → grades → "judge the roles for the task as given".

## Engineering notes

- **Environment:** vLLM 0.22 nightly in the `gemma4` env, run with
  `LD_LIBRARY_PATH=$CONDA/envs/gemma4/lib` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Engine settings:** TRITON_ATTN + TORCH_SDPA ViT on Blackwell, xgrammar JSON schema, greedy
  decoding, thinking off.
- **Throughput:** about 0.8 s/request per GPU with 8 images, so 2,000 requests take about 15 min
  on two GPUs.
- **GPU2 is an RTX 3090 (24 GB) and cannot hold the 27B model.** Use GPU0 or GPU3, which are 96 GB
  each, and check `nvidia-smi` first: other users share GPU3.
- **Token cap:** 300 tokens was too tight once, when the model put its whole answer inside the
  `plan` string. 1,024 is the setting now. A failed parse is reported as a failure, not patched.
- **Variant runners:** the experiments' runners were throwaway scripts that patched `qwen_judge`'s
  prompts. Their exact prompts survive in each run's `config.json`. Trust `config.json` over folder
  names: `qwen38-27b-task-role-v12/` holds a v13 run left from an interrupted session, and
  `qwen38-27b-role-v12/` is the plan-only ablation.
