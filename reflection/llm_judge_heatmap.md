# LLM-as-a-judge for paired heatmaps: what we learned

Qwen3.8-27B rating the paired contact heatmaps of released DUALingo pairs (Hand A orange, Hand B teal)
on the review's 0/1/2 scale. Built 2026-10-08 over 14 prompt versions on the dev split (700 pairs,
two raters). It is the `heatmap` criterion of `data_verification/quality_review/qwen_judge.py` (v23).
Experiment runs are in `data_verification/outputs/quality_review/release1000/qwen38_27b/qwen38-27b-heat-*/`
and the final run is in `qwen38-27b-v23/`. Each run's `config.json` holds its exact prompt.

The task / role note (`llm_judge_task_role.md`) still applies. This note covers what was different
for a criterion the model has to judge by looking.

## Setup

- **Same rules as task / role:**
  - the label is the model's raw greedy output;
  - the prompt is short and general;
  - no quotes from the rater rubric, and no sentences that explain our own pipeline's output.
- **What the judge sees:**
  - the three review sheets (object, Hand A, Hand B; eight views each) at full resolution, 512 px per
    view;
  - the task;
  - both roles without their "where" text.
- **Reference:** two raters on 1,000 pairs. Heatmap is the criterion the raters flag most and agree on
  most: on dev, michaellee flags 13% and FengDian-Su 25% of pairs, and their AC1 is .72.

## Final result (v23 = the v13 prompt, one run on all 1,000 pairs, 0 failures)

| | Qwen | michaellee / FengDian-Su | Exact agreement Qwen vs ML / FS | Exact agreement ML vs FS | Gwet AC1 Qwen vs ML / FS | AC1 ML vs FS |
|---|---:|---:|---:|---:|---:|---:|
| Heatmap, all | 90.3 | 89.6 / 82.0 | 81.2% / 72.4% | 76.2% | .784 / .665 | .712 |
| Heatmap, dev | 91.4 | 90.7 / 83.6 | 84.0% / 73.7% | 76.7% | .818 / .686 | .722 |
| Heatmap, test | 87.7 | 87.2 / 78.3 | 74.7% / 69.3% | 75.0% | .699 / .615 | .689 |

- **Test was scored once**, after the prompt was frozen; it is the only clean held-out result in
  either note. On test the judge's agreement falls below the raters' own (AC1 .66 vs .69), where on
  dev it was above (.75 vs .72), and its score is above both raters' (87.7 vs 87.2 / 78.3).
- **Pairs flagged** (all 1,000):
  - Good pairs flagged: 57 of 697 (dev 34 / 498, test 23 / 199).
  - Both-flagged caught: 55 of 106 (dev 35 / 63, test 20 / 43).
  - Of the 30 pairs both raters graded 0, it flags 22, and grades 17 of them 0.
- **Lenient on mild flaws, compared with a second rater:**
  - Of the heatmaps one rater grades 1, it passes 71% (178 / 251) as 2. The other rater passes 56%
    (140 / 251).
  - The two raters differ a lot here: michaellee passes 67% of FengDian-Su's 1s, and FengDian-Su
    passes 32% of michaellee's.
  - The heatmap score is therefore an upper estimate. This is the vision ceiling described below.
- **Reproducibility:** 691 of 700 dev labels are identical to the heat-v13-2x experiment run, which
  used the same prompt. Greedy decoding in vLLM is not bit-exact across batch compositions, so about
  1% of labels flip between runs.

## The path (dev, 700 pairs; raters score 90.7 / 83.6)

"Good flagged" counts pairs both raters called Good that the model flagged, out of 498. "Both-flagged
caught" counts pairs both raters flagged that the model also flagged, out of 63; about 12 of those 63
are task- or object-level problems no heatmap check can see.

| Version | Change | Score | Good flagged | Both-flagged caught |
|---|---|---:|---:|---:|
| heat-v1 | what a heatmap is + "say where the region is" + "could the hand act within it?" | 92.3 | 44 | 24 |
| v2a | the "where" field also asks how the region looks (one area … spread over most of the object) | 91.4 | 35 | 30 |
| v2b | + the role's "where" text left out | 94.7 | 15 | 25 |
| v3b | + per-hand field: "would touching anywhere in this region work?" | 94.0 | 13 | 27 |
| v4b / v5b | + "the brightest part is the hand's first choice" | 96.7 / 96.6 | 8 / 5 | 13 / 15 |
| v6a / v6b | part + fit (on / spills / off / none) fields; with / without "where" text | 86.5 / 91.5 | 82 / 41 | 38 / 33 |
| v7b | stricter Good definition, spill = "a good share elsewhere" | 89.5 | 53 | 34 |
| **v3b-2x** | v3b with sheets at 512 px per view | 92.7 | 17 | 30 |
| v8b / v9b-2x | small-model step format (look / works; covers / needs / works) | 96.3 / 93.1 | 9 / 25 | 17 / 29 |
| v11b-2x | v3b without the pipeline-specific sentences, in plain own words | 92.4 | 25 | 31 |
| **v12p1-2x** | grades on the raters' own scale: clean / flawed but usable / unusable | 86.8 | 64 | 39 |
| v12p2-2x | + "the points are see-through" | 84.6 | 82 | 36 |
| **v13-2x** | + the size sentence covers every hand that holds, supports or moves the whole object | 91.3 | 35 | 35 |
| v14a / b-2x | + "judge by what the task needs, not the part named" (b: no part name at all) | 93.9 / 94.1 | 19 / 20 | 29 / 29 |

Every wording change landed on one trade-off curve between Good pairs flagged and flawed pairs caught.
Two changes moved the curve itself: full-resolution sheets, and grading on the raters' scale. v13 is
the point the user chose: slightly stricter than v3b, without v12p1's false alarms.

## What we expected the model to see, and what it did instead

1. **Describing the heatmap.**
   - *We assumed* "say where the orange region is" would make the model look.
   - *It actually* wrote the part named in the role ("the recessed front face of the drawer") and then
     found that it matched. A missing region on a Rubik's cube came back as "the back face".
   - *So we* asked how the region looks as well (one clear area, a few areas, scattered specks, a faint
     trace, spread over most of the object), and left the role's "where" text out (false alarms 44 → 15).
2. **Brightness.**
   - *We assumed* "the brightest part is the hand's first choice" would mirror how a person reads a
     heatmap.
   - *It actually* checked only the bright core, which usually sits on the right part, and passed the
     spill that raters penalize. Catches fell by half.
3. **Usable versus clean.**
   - *We assumed* "could the hand do its job within this region?" was what Good means.
   - *It actually* passed flaws it could see: extra specks on the feet of a jar, spill down the sides of
     a lid, three dots on a mug handle. The raters' Good means usable and clean: the rubric's 2 is "no
     observable defect" and its 1 is "a visible defect". Grading on those three levels caught these
     flaws (v12p1).
4. **Region size.**
   - *We assumed* "a holding hand's region can be large" covered size.
   - *It actually* flagged every whole-object action once a large region could be a flaw: lifting,
     rotating or pushing the body, or turning a bowl by its rim.
   - *So we* made the size sentence cover every hand whose possible contacts span the object: holding,
     supporting or moving the whole object (v13: Good flagged 64 → 35, catches 39 → 35).
5. **The part named in the role.**
   - *We assumed* part names help the model find the region.
   - *It actually* judged regions against the word ("covers the whole bowl, not just the rim") instead
     of what the task needs. This is the same trap as "rotate the handle" for roles.
   - *So we* tried telling it to judge by the task instead (v14). That turned it lenient again, because
     the same reading also excused real spill.
6. **Seeing.**
   - *We assumed* resolution and explanation would fix misread regions.
   - *It actually* gained only a little from full resolution, though that was the one input change that
     moved the curve. Explaining the see-through point cloud made it more suspicious everywhere.
     Descriptions of the same image also changed with unrelated sentences: "concentrated on the lid"
     under one prompt, "covers the top edge and cap" under another.
   - About 10 of the 33 misses we inspected are regions it misread. That is the model's limit, not the
     prompt's.
7. **Small-model step format.**
   - *We assumed* explicit steps would help here as they did for roles.
   - *It actually* followed the scoring key exactly (699 of 700) while its look answers stayed wrong.
     Steps fix reasoning, not seeing.
   - A short "where" field invited the role's part name again, and dropping the "spread over most of
     the object" option removed the main overflow signal.

## What mattered

1. **Looking at the misses and the false alarms as images.**
   - Grouping them and mapping each group to a prompt component found both big levers: the grade scale,
     and the gap in the size sentence.
   - Tuning wording against scores alone only ever moved along the curve.
2. **Grading on the raters' own scale:** clean / flawed but usable / unusable, not "usable".
3. **A look-first field whose options describe appearance**, worded so the role text cannot answer it.
4. **Full-resolution images.** Raters can zoom; the judge needs the pixels.
5. **Describing what a heatmap is, not how ours come out.**
   - Possible contacts per hand, brighter = more likely, and size following the hand's possible
     contacts.
   - Sentences that excused our own output (straight-cut halves, whole-body support) cost nothing to
     remove.
6. **The raters' own disagreement bounds pair-level agreement.**
   - On gray-zone flaws the raters disagree: an extra patch was a 1 on one jar and a 2 on a teapot.
   - No wording can match both.

## Designs that failed

- **"Brightest part first"**: lenient, missed the spill raters penalize.
- **Explaining the see-through rendering**: more false alarms, no fewer misreadings.
- **Showing the role's contact text**: the model matched regions against the exact spot (114 Good
  pairs flagged).
- **A few-words "where" field and no "spread over most of the object" option**: the role part name
  again.
- **"Judge by the task, not the part named"**: also excused real spill, back to v3b's point.

## If we did it again

1. Before writing the prompt, look at a sample of flagged and Good heatmaps and write down what raters
   penalize. Here that was any visible flaw (spill, an extra patch, bits, a tiny region), not only
   unusable regions.
2. Grade on the raters' scale from the first version.
3. Describe the heatmap generically:
   - possible contacts per hand;
   - brightness as likelihood;
   - region size following whether the hand's possible contacts span the object or one part.
4. Use a look-first field with appearance options, and no "where" text from the role.
5. Use full-resolution images.
6. Expect a vision ceiling. If catching flawed heatmaps matters more than matching the raters' score,
   the next lever is the model's vision (a stronger VLM), not the prompt.

## Engineering notes

- **Full-resolution sheets:** `python -m data_verification.quality_review.render_hires` writes
  `release1000/renders_2x/`, pixel-identical to the experiment renders, 2048 × 1060 per sheet.
- **Context and speed:** about 6.3K image tokens per request, so the judge's default `--max-model-len`
  is 16384.
  - A heatmap request takes about 2.3 s on one GPU.
  - The final run, with all three criteria, took 35 minutes for 3,000 requests on two GPUs. That is
    1.3 s per request per GPU on average, or about 3.9 GPU-seconds per pair.
- **Paper numbers:** `qwen_judge evaluate` writes `evaluation.json` into the run folder and prints
  each criterion's table row. The experts are the raters who rated every pair. Their score is the
  mean of each expert's score, ± the SD across experts, with AC1 judge–expert and expert–expert.
  Rerun it when the other raters finish.
- **Experiment runner:** heatmap variants ran through a scratchpad runner that patched `qwen_judge`.
  It took the version from the `HEAT_VERSION` environment variable, because vLLM's spawned workers
  re-run the main file and would misread argv.
- **The integrated judge reproduces the experiment prompts exactly:** heatmap 100/100, task and role
  200/200 on a check of 100 pairs.
