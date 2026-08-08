#!/usr/bin/env python3
"""Minimal heatmap-localization verifier for the Claude pilot benchmark.

The render scheme changed on 2026-08-07: the object is now a black point cloud on a mid-grey
background and colour BRIGHTNESS carries prediction strength (was: light-grey object on white,
darker = stronger). Every prompt here states that legend, so the prompts and the renders move
together - a judge run against the old renders with these prompts is invalid, and vice versa.

Measured pathologies this axis has to be written against (corruption tests, 2026-07-28):
  - the model narrates the metadata as if it had observed it. The description is a CLAIM to check,
    not a caption, and the prompts say so.
  - exact-segmentation wording over-penalizes broad maps even when they contain a physically useful
    task-conditioned contact. Potential contact regions on the same reachable structure are valid.
  - collapsed A/B maps escape detection. Their own sub-score forces the comparison.
"""

import json

from simple_verifier_common import WorkItem, verifier_main


AXIS = "stage2"
VERSION = "qwen3vl-stage2-contact-v18.7-concrete-visual-tests"

# Human-confirmed visual calibration examples. The runner may attach their existing composite
# renders to the prompt; these labels are never converted into pixel statistics or thresholds.
VISUAL_ANCHORS = [
    {
        "sample_id": "2c6cca23925c449cab735b60d6e34cf3/q1_rotate_the_adjustment_knob",
        "score": 2,
        "note": "GOOD: a compact task-valid candidate is clearly visible and formed as a usable contact",
    },
    {
        "sample_id": "c69723cebddc4657a850f3aa93b781db/q0_pick_up_the_barrel",
        "score": 2,
        "note": "GOOD: a broad task-valid candidate is clearly visible and formed as a usable contact",
    },
    {
        "sample_id": "f8bb24227a5449969441221d048cdc69/q2_open_the_middle_door",
        "score": 2,
        "note": "GOOD: several separate but formed task-valid candidates are a usable set of alternatives",
    },
    {
        "sample_id": "7b38d99ed20549c7975fcb3948387238/q2_tilt_the_mug_for_pouring",
        "score": 1,
        "note": "OK: the correct contact is indicated, but the candidate itself is only weak glints rather than a formed usable patch",
    },
    {
        "sample_id": "0dc3897ba5484a96a4255394283ecfa3/q4_lock_the_slide_open",
        "score": 1,
        "note": "OK: the contacts support the task, but a main probability field is visibly partial rather than a fully recovered contact shape",
    },
    {
        "sample_id": "402c51dac965426dbb7693c7b7b9db08/q5_open_the_chest_for_access",
        "score": 1,
        "note": "OK: the contacts support the task, but a main probability field has incomplete spatial formation",
    },
    {
        "sample_id": "7c16c7ed67cd4315a23d6e8f2d2060cc/q1_rotate_the_control_knobs",
        "score": 0,
        "note": "BAD: one hand has essentially no visible candidate",
    },
    {
        "sample_id": "c5da4115ea894a689c645fc414481aea/q0_pick_up_the_teapot",
        "score": 0,
        "note": "BAD: one hand has no task-valid candidate and visibly targets a functionally unrelated part",
    },
]

# `mcq` pass-1 answer space; ids match the gold `fault` vocabulary. Fixed order all run.
FINDINGS = [
    "nothing_predicted",
    "wrong_region",
    "fragmented",
]

# `multi` sub-scores: one verdict per hand, over that hand's own map. No coordination sub-score -
# whether the two roles make sense together is the stage1 axis, already verified at full scale, and
# asking again here only pulls the judge away from where the colour sits.
MULTI_AXES = ["hand_A", "hand_B"]

# Emitted BEFORE the sub-scores. These were free-text ("name the parts that carry orange") and the
# judge simply transcribed the expected-contact line into them - "handle / the finger gap of the
# handle", slash format and all - then scored 2 on a picture with no orange in it. A closed boolean
# cannot carry copied wording, so presence is asked as true/false and the prompt states the
# invariant that false forces 0.
_FORMATION = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
_TASK_FIT = {"type": "string", "enum": ["consistent", "ambiguous", "inconsistent"]}
MULTI_OBSERVE = {
    "orange_present": {"type": "boolean"},
    "teal_present": {"type": "boolean"},
    "orange_formation": _FORMATION,
    "teal_formation": _FORMATION,
    "orange_task_fit": _TASK_FIT,
    "teal_task_fit": _TASK_FIT,
}

_LEGEND = """The object is a black point cloud on a flat grey background, shown from eight
viewpoints in each picture. Hand A's predicted contact map is painted orange and hand B's teal.
Colour brightness is contact possibility: brighter colour means a stronger, more likely contact;
dim colour is a weaker candidate; black means no predicted contact for that hand. Judge the bright
high-possibility regions first. Do not give a faint halo the same weight as the main bright region.

Judge only where the colour sits. Whether the task itself makes sense is not your question."""

# Stage 2 evaluates the public artifact: task plus two heatmaps. Intermediate proposed contact
# locations are intentionally absent. The maps are a two-way partition, so segmentation exclusivity
# is not a meaningful quality target; recoverable contact candidates and task consistency are.
_CONTACT = """Evaluate the two heatmaps from only the stated task and visible object geometry. Hand A
and hand B have no hidden prescribed locations or fixed semantic roles. Treat their colours as an
unordered pair when deciding how the two hands could perform the task.

Use this hierarchy.

1. HEATMAP QUALITY. Judge this before task consistency. For each colour independently, read
brightness as probability and inspect the bright regions first. Identify its main proposed contact
candidate or candidate set, then ask whether the probability field actually recovers the spatial
shape of those contacts. Merely revealing where a contact might be is not enough for GOOD.

Assign a formation grade from the coloured probability support alone, before considering the task.
Apply these four visual tests to each main high-probability candidate:

TRACE TEST: In every view where that surface is visible, can you trace the same region's boundary or
endpoints from the colour itself? Projection may change its 2D outline, but it must end on the same
physical seams, rims, junctions, part boundaries, or repeatable surface locations.

INTERIOR TEST: Inside that traced extent, does bright support form a filled patch or band? Ordinary
point-cloud pinholes are allowed. A field fails when its apparent interior is mainly isolated dots,
thin remnants, disconnected crescents, or a fading wash that must be mentally filled.

LANDMARK TEST: Across views, do the region's endpoints and transitions repeatedly meet the same
visible physical landmarks? If the colour stops at unrelated places or only the RGB object shape
suggests where it should continue, the field is incomplete.

RESIDUAL TEST: Outside the main region, are there additional bright patches large enough to create a
genuinely different contact hypothesis? Negligible specks do not matter. Several patches are not a
defect when each independently passes the trace, interior, and landmark tests as a distinct formed
candidate.

Map those observations to the grade:

5 CLEANLY FORMED: trace, interior, and landmark tests clearly pass; support is concentrated in the
formed region or regions; no meaningful unexplained residual patch remains.

4 FORMED: the same physical region can still be traced and has a filled bright core, but one visible
nonfatal defect remains: mild erosion, a soft/ragged section of boundary, a partial interior gap, or
a secondary candidate patch. No mental completion is needed to identify the main footprint.

3 ZONE-LEVEL / INCOMPLETE: one can name the coloured part or general surface zone, but at least one
of the trace, interior, or landmark tests materially fails. Typical evidence is an open-ended partial
coating, a diffuse wash, a boundary that drifts between physical landmarks, or a region whose missing
shape must be supplied from RGB geometry. It may be bright, contiguous, and task-consistent; those
facts do not make it formed.

2 POORLY FORMED: only disconnected fragments or remnants indicate possible regions; no main region
passes the trace and interior tests, but a rough location is still inferable.

1 UNRECOVERABLE: no usable region is inferable, including absent colour or only isolated specks.

Area is never a test. A broad wall or nearly the whole object passes when its coloured interior is
filled and its extent repeats across views; its boundary may be the object silhouette. A tiny knob,
complete rim, or thin handle passes by the same tests. Do not require a hand-sized patch.

Do not revise this grade after reading the task. The RGB geometry may establish which coloured
fragments belong to the same physical surface, but it must not supply missing probability support.
Coherence is local to each candidate, not global to every point of the same colour. A compact
control, thin rim, long handle, or broad wall can all receive grade 5. Separate formed candidates
are legitimate alternatives and need not join into one contour. Viewpoint and occlusion may split
one physical region between panels.

2. TASK CONSISTENCY. Only after fixing both formation grades, interpret the two colours jointly. Ask whether there is at least one
physically plausible assignment of their candidates to the acting and supporting contacts needed by
the stated task. Either colour may take either role. A candidate is task-consistent when it could
participate in such an assignment; it need not match one predetermined contact location. Prefer the
best plausible interpretation supported by the images, while rejecting contacts that clearly cannot
contribute to the task.

The two hand fields partition essentially the entire object. Therefore colour elsewhere is expected:
do not judge overflow, exclusivity, total coloured area, coverage percentage, candidate count, or
global connectedness. Extra coverage cannot cancel a formed candidate. Individual black points,
point-cloud sampling texture, mild probability gradients, small holes, and approximate boundaries
are not defects by themselves. Require a material loss of the candidate's recoverable shape before
calling it incomplete, but do not excuse that loss merely because its task location is correct.

For each colour, label task fit as consistent, ambiguous, or inconsistent under the best plausible
two-hand assignment. Then map the two independent judgments:
- 2 GOOD: formation 4-5 AND task fit consistent.
- 1 OK: formation 2-3 with a recoverable task contribution, OR formation 4-5 with ambiguous task fit.
- 0 BAD: formation 1, no visible colour, or task fit inconsistent.

Both hands must score 2 for the sample to be GOOD. Correct task location can never upgrade formation
1-3 to GOOD. Presence records whether any colour is visible."""


PROMPTS = {
    "multi": """You are shown the same object twice, as two pictures. The FIRST picture is hand A's predicted contact map in orange, the SECOND is hand B's in teal, each from the same eight viewpoints. Score the two hands independently.

""" + _LEGEND + """

""" + _CONTACT + """

The eight views together cover the object, so a region turned away in one view faces you in another. Check every view before concluding that a hand's colour is not somewhere.

Score each hand only by the mapping above. Do not collapse formation and task fit into one intuitive
impression.

The task provides intent, not an exact mask or fixed A/B assignment. Judge what is visible.

Fix both formation grades before making either task-fit judgment.

Report in this order:
orange_present: Is there any orange at all in the FIRST picture, in any of its eight views? Decide this by looking, before you read the claim again.
teal_present: The same for teal in the SECOND picture.
orange_formation: integer 1-5 using the formation scale above.
teal_formation: integer 1-5 using the same scale.
orange_task_fit: consistent, ambiguous, or inconsistent under the best plausible two-hand assignment.
teal_task_fit: the same for teal.
hand_A: 2 for formation 4-5+consistent; 1 for a recoverable nonfatal case; otherwise 0.
hand_B: the same. A false presence flag forces formation 1 and score 0.
reason: ONE short sentence that names any formation grade below 4 before discussing task fit.""",

    "direct": _LEGEND + """

The first picture is hand A's map, the second is hand B's.

2  Both heatmaps are formed and together support a plausible execution of the task.
1  The task remains recoverable, but at least one usable candidate is weak or unformed.
0  A hand has no usable candidate, or its candidates clearly cannot contribute to the task.

""" + _CONTACT + """

Return JSON only: {"score": 0|1|2, "reason": "one concise visual observation"}.""",

    "bad": _LEGEND + """

The first picture is hand A's map, the second is hand B's.

Answer one question: does either colour lack a usable candidate, or clearly fail to contribute to
any plausible two-hand execution of the task?

""" + _CONTACT + """

Return JSON only: {"clear_bad": true|false, "reason": "one concise visual observation"}.""",

    "mcq": _LEGEND + """

The first picture is hand A's map, the second is hand B's. Choose exactly one answer.

Most maps are fine. The breakages are listed first and the last two answers are the non-broken ones.
Choose a breakage only when you can say in your reason which hand it is and where its colour actually
sits. If a map merely looks imperfect, that belongs in acceptable_imprecise.

""" + _CONTACT + """

- nothing_predicted: one hand has almost no colour anywhere on the object - a few stray points or
  none at all.

- wrong_region: a colour has no visible candidate that could plausibly contribute to the task.

- fragmented: the colour is only isolated noise and forms no recognizable contact shape. A localized
  but visibly broken heatmap belongs in acceptable_imprecise, not here.

- acceptable_imprecise: the task remains plausibly recoverable, but at least one candidate is weak
  or lacks a stable formed contact region.

- fully_sound: both colours have formed candidates that together support a plausible execution of
  the task.

Return JSON only:
{"finding": "<one id above>", "reason": "one concise visual observation"}""",

    "good": _LEGEND + """

The first picture is hand A's map, the second is hand B's. Neither hand has been found to clearly
lack a usable contact.

Answer one question: does each colour have a formed candidate, and do the two colours together
support a plausible execution of the task?

""" + _CONTACT + """

Answer false when a correctly localized map is visibly fragmented or incoherent, not merely because
its contour is imperfect.

Return JSON only: {"fully_good": true|false, "reason": "one concise visual observation"}.""",
}


def build_item(row: dict) -> WorkItem:
    with open(row["meta_path"], encoding="utf-8") as f:
        meta = json.load(f)
    text = "\n".join([
        f"Task: {meta.get('task')}",
        "Judge the orange and teal heatmaps from this task only; no target locations or hand-role "
        "assignment are provided.",
        "Return the required JSON.",
    ])
    return WorkItem(
        sample_id=row["sample_id"], object_id=row["object_id"],
        image_path=(row["heat_png_A"], row["heat_png_B"]), user_text=text,
    )


if __name__ == "__main__":
    verifier_main(AXIS, VERSION, PROMPTS, build_item, FINDINGS, MULTI_AXES, MULTI_OBSERVE)
