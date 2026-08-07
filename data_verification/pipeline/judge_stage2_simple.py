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
VERSION = "qwen3vl-stage2-contact-v13.6-anchored"

# Human-confirmed visual calibration examples. The runner may attach their existing composite
# renders to the prompt; these labels are never converted into pixel statistics or thresholds.
VISUAL_ANCHORS = [
    {
        "sample_id": "58fff2344f5045968974ea2d485625e8/q4_expand_the_opening_for_foot_entry",
        "score": 2,
        "note": "stable upper/lower shoe contact regions; internal point noise does not erase them",
    },
    {
        "sample_id": "6bf8e590522e49659d40c4d2a52b41ca/q3_adjust_the_lectern_angle",
        "score": 2,
        "note": "stable top and column regions across views",
    },
    {
        "sample_id": "7bbd31f280034a519db0f07c08d7bfa7/q2_tilt_the_jug_for_pouring",
        "score": 2,
        "note": "recognizable upper grasp and lower support regions",
    },
    {
        "sample_id": "b3f0fba05f264d7f893ab81a088b9328/q3_open_the_lid",
        "score": 2,
        "note": "a small rim band and broad wall support are both formed contacts",
    },
    {
        "sample_id": "5951a136f06d4212b9b833b58fed219b/q2_open_the_drawer",
        "score": 1,
        "note": "task-related colour, but at least one hand lacks a stable formed region",
    },
    {
        "sample_id": "c3a50e99ae574d97a711bfb56a434179/q4_close_the_book",
        "score": 1,
        "note": "task-related colour, but the probability structure is visibly unformed",
    },
    {
        "sample_id": "de516e7330e1491fac4144e3d2f458f1/q3_stretch_the_mask_opening",
        "score": 1,
        "note": "task-related colour, but at least one field lacks a stable cross-view shape",
    },
    {
        "sample_id": "fa8b4476c1b14c9d9aceef6979d21428/q2_open_the_wicker_basket_lid",
        "score": 1,
        "note": "correct regions, but at least one probability field is not well formed",
    },
    {
        "sample_id": "7c16c7ed67cd4315a23d6e8f2d2060cc/q1_rotate_the_control_knobs",
        "score": 0,
        "note": "one hand has no visible prediction",
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
_ALIGNMENT = {"type": "string", "enum": ["clear_hit", "partial_hit", "severe_miss"]}
_SHAPE = {"type": "string", "enum": ["complete", "incomplete", "absent"]}
MULTI_OBSERVE = {
    "orange_present": {"type": "boolean"},
    "teal_present": {"type": "boolean"},
    "orange_shape": _SHAPE,
    "teal_shape": _SHAPE,
    "orange_alignment": _ALIGNMENT,
    "teal_alignment": _ALIGNMENT,
}

_LEGEND = """The object is a black point cloud on a flat grey background, shown from eight
viewpoints in each picture. Hand A's predicted contact map is painted orange and hand B's teal.
Colour brightness is contact possibility: brighter colour means a stronger, more likely contact;
dim colour is a weaker candidate; black means no predicted contact for that hand. Judge the bright
high-possibility regions first. Do not give a faint halo the same weight as the main bright region.

Judge only where the colour sits. Whether the task itself makes sense is not your question."""

# One principle, from which the grades follow: read the map as a physical contact. It replaces a
# stack of separate clauses (breadth is fine / speckle is not / tidiness is not correctness / a small
# patch still counts) that had accumulated one failure at a time, and it fixes what that stack got
# wrong. The old breadth clause ended "the question is only whether the region the words name is
# among what that hand's colour covers", which told the judge to ignore everything outside the named
# region. This rubric instead asks whether the prediction contains a usable task-conditioned
# contact, while still rejecting maps whose only stable surface is functionally unrelated.
#
# The two maps are disjoint by construction and together cover essentially the whole object (median
# 100% of the cloud over the pilot set): this is a partition into an A-side and a B-side, not a pair
# of localized spots, so breadth genuinely cannot be a defect. Coherence can.
_CONTACT = """Evaluate each hand's heatmap as a task-conditioned spatial probability field, not as a
pixel-perfect segmentation mask. Keep visual formation and semantic validity independent.

1. PRESENCE. First decide whether that hand has any visible coloured prediction. No visible field is
absent and scores 0.

2. FIELD FORMATION. Before using the task wording, judge only the probability structure in the
images. Start with the bright high-probability core and use dimmer support only to understand its
shape. A complete field has dominant bright structure that forms a stable, recognizable 3D contact
region across views, with support extending it smoothly. Use incomplete when bright structure breaks
into unrelated islands, jumps between structures, changes location across views, or never settles
into a recognizable shape. Sparse points, black points, holes, approximate boundaries, and region
area are not quality measures by themselves; judge whether the probability mass as a whole is formed.

Do not let a correct target excuse an unformed field, and do not use the contact description to
imagine a shape that is not visually present.

3. CONTACT VALIDITY. Now judge whether the bright core and its surrounding field lie on the instructed
contact, the same local mechanism, or another region a hand could plausibly use for the task. The
wording is a semantic anchor, not an exact mask. Extra coverage is expected because the two maps
partition the object; it is not an error when the field remains task-relevant. Absolute area is never
a quality measure: a small precise contact and a broad graspable surface can both be valid.

Use clear_hit when the target/task relation is clear and partial_hit when it is plausible but
ambiguous. Use severe_miss only when the intended region is essentially unmarked and the dominant
field is both spatially far and functionally unrelated. A broad field, an unformed field, or a field
on the same usable mechanism is never severe_miss by itself."""


PROMPTS = {
    "multi": """You are shown the same object twice, as two pictures. The FIRST picture is hand A's predicted contact map in orange, the SECOND is hand B's in teal, each from the same eight viewpoints. Score the two hands independently.

""" + _LEGEND + """

""" + _CONTACT + """

The eight views together cover the object, so a region turned away in one view faces you in another. Check every view before concluding that a hand's colour is not somewhere.

Score each hand:
2 = clear_hit AND a well-formed coherent probability field (complete).
1 = partial_hit, or a valid contact whose probability field is visibly unformed (incomplete).
0 = severe_miss, or no visible contact shape. Use 0 only for near-absence or clear major deviation.

When assigning 1 for incompleteness, name the visible fragmentation or incoherence in the reason.

The words provide intent, not an exact mask. Never copy them as if they described what is visible;
judge whether the visible contact can serve the task.

Make the localization and completeness observations for each hand independently before scoring.

Report in this order:
orange_present: Is there any orange at all in the FIRST picture, in any of its eight views? Decide this by looking, before you read the claim again.
teal_present: The same for teal in the SECOND picture.
orange_shape: complete when the heatmap forms a smooth coherent contact region; incomplete when it
is visibly unformed; absent when no contact shape is visible. Decide from image structure alone.
teal_shape: the same for teal.
orange_alignment: clear_hit, partial_hit, or severe_miss against hand A's instructed contact.
teal_alignment: clear_hit, partial_hit, or severe_miss against hand B's instructed contact.
hand_A: 0 for severe_miss/absent; 2 only for clear_hit+complete; otherwise 1.
hand_B: the same. A false presence flag also forces 0.
reason: ONE short sentence. Do not walk through the views.""",

    "direct": _LEGEND + """

The first picture is hand A's map, the second is hand B's.

2  Each hand has a clear localization hit and a smooth, coherent contact region.
1  Localization is usable, but at least one heatmap is visibly fragmented or incoherent.
0  A hand is nearly absent or its main bright region severely misses the instructed contact.

""" + _CONTACT + """

Return JSON only: {"score": 0|1|2, "reason": "one concise visual observation"}.""",

    "bad": _LEGEND + """

The first picture is hand A's map, the second is hand B's.

Answer one question: is either hand nearly absent, or is its main bright region clearly far from the
instructed contact while that contact is essentially unmarked?

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

- wrong_region: a hand's main bright region clearly and severely misses the instructed contact while
  that contact is essentially unmarked.

- fragmented: the colour is only isolated noise and forms no recognizable contact shape. A localized
  but visibly broken heatmap belongs in acceptable_imprecise, not here.

- acceptable_imprecise: both hands are localized plausibly, but at least one heatmap is visibly
  fragmented, discontinuous, or lacks a stable coherent region.

- fully_sound: each hand clearly hits its instructed contact and forms a smooth, coherent contact
  region.

Return JSON only:
{"finding": "<one id above>", "reason": "one concise visual observation"}""",

    "good": _LEGEND + """

The first picture is hand A's map, the second is hand B's. Neither hand has been found to clearly
lack a usable contact.

Answer one question: does each hand clearly hit its instructed contact with a smooth, coherent
contact region?

""" + _CONTACT + """

Answer false when a correctly localized map is visibly fragmented or incoherent, not merely because
its contour is imperfect.

Return JSON only: {"fully_good": true|false, "reason": "one concise visual observation"}.""",
}


def build_item(row: dict) -> WorkItem:
    with open(row["meta_path"], encoding="utf-8") as f:
        meta = json.load(f)
    roles = meta.get("roles")
    if not isinstance(roles, list) or len(roles) != 2:
        raise ValueError("meta.roles must contain exactly two hands")
    a, b = roles
    # no `relation` line: stage2 meta never carries the field, so it was a constant "not specified"
    # on every sample - an empty slot the judge can only speculate into.
    # "expected contact" read as ground truth and the slash form read as a half-finished answer:
    # the judge copied "handle / the finger gap of the handle" straight into its observation field.
    # Naming these as claims, in prose, removes the template there was to complete.
    def claim(label: str, colour: str, role: dict) -> str:
        where = (role.get("contact_region") or "").strip()
        target = (role.get("target") or "").strip()
        if target and target.lower() not in where.lower() and target.lower() != "body":
            where += f", on the {target}"
        return f"Claim under test, hand {label} ({colour}): the {colour} marks {where}."

    text = "\n".join([
        f"Task: {meta.get('task')}",
        claim("A", "orange", a),
        claim("B", "teal", b),
        "Return the required JSON.",
    ])
    return WorkItem(
        sample_id=row["sample_id"], object_id=row["object_id"],
        image_path=(row["heat_png_A"], row["heat_png_B"]), user_text=text,
    )


if __name__ == "__main__":
    verifier_main(AXIS, VERSION, PROMPTS, build_item, FINDINGS, MULTI_AXES, MULTI_OBSERVE)
