#!/usr/bin/env python3
"""Minimal heatmap-localization verifier for the Claude pilot benchmark."""

import json

from simple_verifier_common import WorkItem, verifier_main


AXIS = "stage2"
VERSION = "qwen3vl-stage2-simple-v5.0"

# `mcq` pass-1 answer space; ids match the gold `fault` vocabulary. Fixed order all run.
FINDINGS = [
    "wrong_part",
    "essentially_absent",
    "ab_collapsed",
    "covers_unmentioned_surfaces",
]

_LEGEND = """Two contact maps are painted on one object, shown from eight viewpoints. Orange marks
where hand A is expected to touch, teal marks hand B, and the deeper the colour the stronger the
prediction. Light grey is unmarked surface.

Judge only where the colour sits. Whether the task itself makes sense is not your question."""

PROMPTS = {
    "direct": _LEGEND + """

2  Each colour sits on the region its description names, and the two are arranged the way the
   described contacts require.
1  Roughly right, but one colour is faint, smeared across a boundary, or only partly on its region.
0  A colour sits on a different part than the one described, is essentially absent, or both colours
   land in the same place when the two contacts have to be distinct.

Judge extent against the description, not against area. A description naming a broad surface should
be marked broadly; a description naming one specific feature should be marked there and not across
unrelated surfaces. A large marked area is an error only when it covers surfaces the description
never mentions.

Return JSON only: {"score": 0|1|2, "reason": "one concise visual observation"}.""",

    "bad": _LEGEND + """

Answer one question: is a colour in the wrong place?

That means it sits on a different part than the one described, is essentially absent, spreads across
surfaces the description never mentions, or both colours land in the same place when the two
contacts have to be distinct.

Breadth alone is not an error: a description naming a broad surface should be marked broadly.
Imprecision is not an error either — only misplacement is.

Return JSON only: {"clear_bad": true|false, "reason": "one concise visual observation"}.""",

    "mcq": _LEGEND + """

Choose exactly one answer.

Most maps are fine. The breakages are listed first, and the last two answers are the non-broken
ones. Choose a breakage
only when you can point to it in your reason - which part carries the wrong colour, which surface is
marked but unmentioned. If a map merely looks imperfect, that belongs in acceptable_imprecise.

Breadth by itself is never a breakage: a description naming a broad surface should be marked broadly,
and the two hands may legitimately differ in breadth.


- wrong_part: a colour sits on a different part than the one its description names.

- essentially_absent: one of the two colours is missing or reduced to a few stray points.

- ab_collapsed: both colours land on the same place, although the two contacts have to be distinct.

- covers_unmentioned_surfaces: a colour spreads onto surfaces its description never mentions, and you
  can name which ones.

- acceptable_imprecise: the colours are on the right regions but the map is not crisp - one is faint,
  drifts a little past a boundary, or is only partly on its region. Nothing is misplaced.

- fully_sound: each colour sits on the region its description names, its edges follow that feature,
  and the two are separated or overlapping as the described contacts require.

Return JSON only:
{"finding": "<one id above>", "reason": "one concise visual observation"}""",

    "good": _LEGEND + """

This map has already been checked for misplaced colour.

Answer one question: is it fully sound, rather than merely approximate?

Fully sound means each colour is concentrated where its description points, its edges follow the
feature rather than drifting past it, and the two regions are separated or overlapping as the
described contacts require.

Answer false when a colour is faint, drifts across a boundary, or leaves you unsure which feature it
is meant to mark.

Return JSON only: {"fully_good": true|false, "reason": "one concise visual observation"}.""",
}


def build_item(row: dict) -> WorkItem:
    with open(row["meta_path"], encoding="utf-8") as f:
        meta = json.load(f)
    roles = meta.get("roles")
    if not isinstance(roles, list) or len(roles) != 2:
        raise ValueError("meta.roles must contain exactly two hands")
    a, b = roles
    relation = meta.get("relation") or "not specified"
    text = "\n".join([
        f"Task: {meta.get('task')}",
        f"Orange / Hand A expected contact: {a.get('target')} / {a.get('contact_region')}",
        f"Teal / Hand B expected contact: {b.get('target')} / {b.get('contact_region')}",
        f"Expected relation: {relation}",
        "Return the required JSON.",
    ])
    return WorkItem(
        sample_id=row["sample_id"], object_id=row["object_id"],
        image_path=row["heat_png"], user_text=text,
    )


if __name__ == "__main__":
    verifier_main(AXIS, VERSION, PROMPTS, build_item, FINDINGS)

