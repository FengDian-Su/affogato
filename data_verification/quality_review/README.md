# Release quality review (human, multi-rater)

Local FastAPI + SQLite web app for the paper's annotation-quality assessment. Raters score
**released** DUALingo pairs on three criteria, each Bad / OK / Good (0 / 1 / 2):

| Step | Criterion | Rater sees |
|---|---|---|
| 1 | Task validity: is this a sensible two-handed task for this object? | RGB views, object name, task |
| 2 | Role assignment: do the two hand roles divide the task correctly? | + each hand's action, target part, contact region, function |
| 3 | Paired heatmaps: do the orange (A) and teal (B) regions mark usable contacts? | + Hand A / Hand B heatmaps from the RGB cameras |

Each step's evidence is revealed only after the previous step is scored, so the task is judged
before the roles are visible and the roles before the heatmaps. Bad and OK need at least one
reason tag. The rubric lives in `rubric.py` and is frozen into the database at `init`. The
heatmap verdicts and tags are verbatim from the pilot heatmap UI (`../human_review/`), so the
new heatmap labels stay comparable with the pilot gold.

**Design: every rater scores every pair independently** (the plan is five raters). There is no
consensus or adjudication step: each criterion's result is the average of the raters' own scores.
Raters can edit any of their ratings from History. Nobody sees anyone else's labels, and model
judgments are not shown anywhere.

## Pilot heatmap ratings carried over

`import_v1.py` carries the pilot's heatmap ratings into the review database, linked by object id +
task. A rating is carried only where the released heatmap still looks like the one that was rated:
for both hands, the coloured region of the released render must overlap the rated render with
IoU >= 0.7 (median 0.996). That holds for **966 pairs**; the other **34** changed visibly in the
September regeneration (4 of them are A/B swaps of same-role pairs, e.g. lift + lift) and carry
nothing over, so everyone rates them in full. Their IDs are kept as a record in the database
metadata (`import_v1`).

Only the pilot's two blind raters have heatmap ratings of their own. EN's pilot records are
adjudications, made after seeing both raters' labels, so they are not independent ratings and do
not count toward the average.

| Rater | Carried pair (966) | Other pair (34) |
|---|---|---|
| michaellee, FengDian-Su | all three steps; heatmap pre-filled with their own pilot rating, changeable | all three steps |
| everyone else, EN included | all three steps | all three steps |

A pre-filled pilot rating is only sent back if the rater changes it; otherwise the pilot rating
stands as theirs. Pilot raters must sign in with exactly their pilot username (case-sensitive).

## Sample frame

The 1,000 pilot human-gold pairs (`../outputs/human_review/pilot1000_export/stage2_human_gold.jsonl`,
all daily_used, object-disjoint 700 dev / 300 test). Their task and role text is re-read from
the current stage-2 release (identical to the pilot for all 1,000). The heatmaps are
re-rendered from the release `scores.npz` because the release was regenerated with
conditional-mean mask consolidation (2026-09-14..17) after the pilot renders were made. The
renderer is the validated camera-aligned one (`../pipeline/render_aligned_heatmaps.py`); RGB
sheets are pixel-identical to the pilot's. The manifest records each pair's `meta.json` /
`scores.npz` SHA-256, so every rating points at exact release bytes.

## Run

All commands from the repo root, in the `gemma4` env (FastAPI, uvicorn, numpy):

```bash
PY=/home/michaellee/miniconda3/envs/gemma4/bin/python

# 1. Manifest + renders (done; ~2 min on 32 workers, 541 MB of lossless WebP).
$PY -m data_verification.quality_review.prepare

# 2. Database (done). Add --force to recreate it before any ratings exist.
$PY -m data_verification.quality_review.app init

# 2b. Carry over the pilot heatmap ratings (done). Refuses once anyone has rated.
$PY -m data_verification.quality_review.import_v1 --min-iou 0.7

# 3. Serve (how it runs now: two detached tmux sessions).
OUT=data_verification/outputs/quality_review/release1000
tmux new-session -d -s quality_review "$PY -m data_verification.quality_review.app serve \
    --host 0.0.0.0 --port 8770 --allow nycu --allow 192.168.50.0/24 \
    --team-code-file $OUT/team_code.txt 2>&1 | tee -a $OUT/serve.log"
tmux new-session -d -s quality_review_tunnel \
    "~/.local/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8770 2>&1 | tee $OUT/tunnel.log"
grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' $OUT/tunnel.log | head -1   # the public URL
```

This machine sits behind a NAT router (192.168.50.1) that only forwards SSH, so outsiders cannot
reach `IP:port` directly. A Cloudflare quick tunnel gives a public HTTPS URL instead; it changes
whenever the tunnel session restarts, so re-share it then. Two guards keep it private:

- `--allow nycu` admits only NYCU's own address space (every prefix AS9916 announces, IPv4 and
  IPv6; `--allow` can be repeated with more CIDRs). Behind the tunnel the real address comes from
  `CF-Connecting-IP`, verified end to end. Off campus, raters connect through the NYCU VPN.
- `--team-code-file` makes sign-in require a shared code (`team_code.txt`, mode 600, so other users
  of this machine cannot read it). Share it privately with the raters.

Restarting only the `quality_review` session (after a code or rubric change) keeps the tunnel and
its URL. Back up the live database first with SQLite's backup API (`review.backup_*.sqlite3`).

To add reason tags while people are rating, edit `rubric.py` and run
`$PY -m data_verification.quality_review.app update-rubric`, then restart the `quality_review`
session. The command only accepts added tags; any other wording change is refused, so every
existing rating stays valid, and each update is logged in the database (`rubric_history`).
v1.1 added the task reason "Not a meaningful task".

Raters sign in with any name; names are only labels. Each rater walks the same queue order at
their own pace, and the header shows their own progress out of 1,000. Keyboard: `G`/`O`/`B`
(or `2`/`1`/`0`) score the open step, `Enter` moves on or submits, click a view to zoom
(`←`/`→` move through views), `?` opens the guide.

## Results for the paper

```bash
$PY -m data_verification.quality_review.app export   # -> outputs/quality_review/release1000/export/
```

| File | Content |
|---|---|
| `per_pair.jsonl` | Per pair and criterion: the mean over raters (0-100) and every rater's label |
| `per_object.jsonl` | Per object and criterion: the mean of its pairs' scores |
| `annotations.audit.jsonl` | Every rating, including pilot imports and revisions |
| `summary.json` | Per criterion: the averaged score, its spread across raters, each rater's own score, Good %, distribution, Krippendorff's α; rater table, reason counts, provenance |
| `paper_table.tex` | Criterion · Score (± std across raters) · Good (%) · α, ready for `\input` |
| `rubric_table.tex` | The rubric as an appendix table (also in `figures/`) |

Each criterion has its own score: every rater's mean rating / 2 × 100 over the pairs they rated,
averaged over the raters (each rater counts once); Good % is averaged the same way. A score is not
an accuracy: all-OK and half-Bad/half-Good both give 50, so report it next to Good %. α is ordinal
Krippendorff's alpha across raters, checked against the published reference example in
`test_app.py`. Until every rater has finished, raters cover different pairs, so read the final
numbers at the end.

Appendix screenshots (rated as an anonymous `rater1` on a demo copy, 3200 px wide) are in
`../outputs/quality_review/release1000/figures/`.

## LLM judge (Qwen3.8-27B)

`qwen_judge.py` rates every pair on the same three criteria, from the evidence a rater has at each
step:
1. **Task:** the eight RGB views, the object name and the task.
2. **Role:** the same, plus both hands' roles.
3. **Heatmaps:** the three review sheets at full resolution, 512 px per view (`render_hires.py`).

The label is the model's raw greedy output under a fixed prompt. The score-token probabilities are
only logged. The prompt history is in the comments above `PROMPTS`. What we learned is written up in
`reflection/llm_judge_task_role.md` and `reflection/llm_judge_heatmap.md`.

```bash
$PY -m data_verification.quality_review.render_hires          # full-resolution sheets (done; 645 MB)
# One GPU per shard. vLLM in the gemma4 env needs the env's lib dir on LD_LIBRARY_PATH.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/michaellee/miniconda3/envs/gemma4/lib \
    $PY -m data_verification.quality_review.qwen_judge run --split all --shard 0/2
$PY -m data_verification.quality_review.qwen_judge evaluate   # vs the raters -> <run>/evaluation.json
```

Result for v23 on all 1,000 pairs, measured on 2026-10-08 against the two raters who had finished:

| Criterion | Judge | Raters |
|---|---:|---:|
| Task | 93.3 | 95.8 |
| Role | 96.5 | 97.5 |
| Heatmaps | 90.3 | 85.8 |

On every criterion, the judge's Gwet's AC1 with each rater is within .02 of the raters' AC1 with
each other.

## Tests

```bash
$PY -m unittest data_verification.quality_review.test_app -v
```
