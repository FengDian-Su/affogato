#!/usr/bin/env python3
"""Minimal RGB-conditioned task/role verifier for the Claude pilot benchmark."""

import json

from simple_verifier_common import WorkItem, verifier_main


AXIS = "stage1"
VERSION = "qwen3vl-stage1-simple-v12.0"

# The breakage half of the `mcq` answer space (the other two answers, acceptable_imprecise and
# fully_sound, are added by mcq_answers()). Ids match the gold `fault` vocabulary so tag agreement is
# directly comparable. Order is fixed for the whole run so any position bias stays measurable.
FINDINGS = [
    "task_not_possible",
    "part_absent",
    "forces_cancel",
    "anchor_moves",
    "action_cannot_produce_effect",
    "hand_obstructs",
]

# `multi` sub-scores, mirroring judge_stage1.py's four items. Order fixes the tie-break for the
# error type, so the earlier an axis sits the more it "owns" a shared minimum.
MULTI_AXES = ["task", "bimanual", "hand_A", "hand_B"]

PROMPTS = {
    'gate': """Answer one question about this proposed two-handed interaction: is it fully sound, or
merely workable?

The hand actions come from a coarse fixed set: hold, lift, push, pull, press, slide, rotate,
squeeze. Read each as the nearest available description of that hand's contribution; it need not
name the whole task. A part that is not on this object is an error, but a part that is merely
hidden, occluded or too small to resolve is uncertainty - check every view before calling a part
absent. Judge the object in front of you, not the object category in general.

Answer true only when you can confirm all of it: every named part is located in the views, each
contact is somewhere a hand can reach, each action word is what that hand really does rather than
the closest available substitute, and the two hands' contributions combine into the task.

Answer false when anything is wrong OR unconfirmable: a part you cannot find, a contact described
too vaguely to check, an action that is only a stand-in for a motion this vocabulary cannot express,
a hand that contributes nothing, or two hands that work against each other.

Return JSON only: {"fully_good": true|false, "reason": "one concise sentence"}""",

    'tag': """This proposal has already been judged not fully sound. Say what kind of problem it is.
Choose exactly one.

The hand actions come from a coarse fixed set: hold, lift, push, pull, press, slide, rotate,
squeeze. Read each as the nearest available description of that hand's contribution; it need not
name the whole task. A part that is not on this object is an error, but a part that is merely
hidden, occluded or too small to resolve is uncertainty - check every view before calling a part
absent. Judge the object in front of you, not the object category in general.

- task_not_possible: the object's visible structure makes the task impossible whatever the hands do.
- part_absent: a part the proposal needs is not on this object.
- forces_cancel: the two contacts create forces that oppose each other and prevent the movement.
- anchor_moves: the supporting hand travels with what it should hold still, or restrains a part that
  has to move.
- action_cannot_produce_effect: one hand, at its stated contact, cannot make any positive
  contribution to the task.
- hand_obstructs: one hand occupies or blocks the contact or path the other needs.
- acceptable_imprecise: nothing is actually broken - it is only vague, unconfirmable from these
  views, or an approximate word for a motion the vocabulary cannot express.

Return JSON only: {"finding": "<one id above>", "reason": "one concise sentence"}""",

    'axis_task': """Answer one question only: can this object actually perform the stated task?

The hand actions come from a coarse fixed set: hold, lift, push, pull, press, slide, rotate,
squeeze. Read each as the nearest available description of that hand's contribution; it need not
name the whole task. A part that is not on this object is an error, but a part that is merely
hidden, occluded or too small to resolve is uncertainty - check every view before calling a part
absent. Judge the object in front of you, not the object category in general.

Ignore how the two hands were assigned; that is judged separately.

2 = plausible and the images support it; 1 = possibly plausible but the images are insufficient or
the manner of operation is uncertain; 0 = the object cannot support it, or the task belongs to a
different object.

Return JSON only: {"score": 0|1|2, "reason": "one concise sentence"}""",

    'axis_bimanual': """Answer one question only: do the two hands, as described, form one coherent
operation?

The hand actions come from a coarse fixed set: hold, lift, push, pull, press, slide, rotate,
squeeze. Read each as the nearest available description of that hand's contribution; it need not
name the whole task. A part that is not on this object is an error, but a part that is merely
hidden, occluded or too small to resolve is uncertainty - check every view before calling a part
absent. Judge the object in front of you, not the object category in general.

Assume the task itself is fine and both parts exist; those are judged separately. Two hands are the
ordinary way people handle objects and two hands using the same action is normal, so do not mark it
down for redundancy - only for interference.

2 = the two contributions combine into the task; 1 = they work together but the division of labour
is weak or hard to picture; 0 = one hand blocks, locks or cancels what the other must do, or the two
roles cannot form a coherent operation.

Return JSON only: {"score": 0|1|2, "reason": "one concise sentence"}""",

    'axis_hand_A': """Answer one question only: is Hand A's description right? Ignore Hand B entirely.

The hand actions come from a coarse fixed set: hold, lift, push, pull, press, slide, rotate,
squeeze. Read each as the nearest available description of that hand's contribution; it need not
name the whole task. A part that is not on this object is an error, but a part that is merely
hidden, occluded or too small to resolve is uncertainty - check every view before calling a part
absent. Judge the object in front of you, not the object category in general.

Judge Hand A as one verdict over its action, target, contact region and function. The function must
be the consequence of that action applied at that contact, not a restatement of the task.

2 = it agrees with the visual evidence; 1 = broadly reasonable but the part cannot be confirmed or
the description is too vague to check; 0 = the action and function conflict, the part does not
exist, or the contact is clearly implausible for the stated function.

Return JSON only: {"score": 0|1|2, "reason": "one concise sentence"}""",

    'axis_hand_B': """Answer one question only: is Hand B's description right? Ignore Hand A entirely.

The hand actions come from a coarse fixed set: hold, lift, push, pull, press, slide, rotate,
squeeze. Read each as the nearest available description of that hand's contribution; it need not
name the whole task. A part that is not on this object is an error, but a part that is merely
hidden, occluded or too small to resolve is uncertainty - check every view before calling a part
absent. Judge the object in front of you, not the object category in general.

Judge Hand B as one verdict over its action, target, contact region and function. The function must
be the consequence of that action applied at that contact, not a restatement of the task.

2 = it agrees with the visual evidence; 1 = broadly reasonable but the part cannot be confirmed or
the description is too vague to check; 0 = the action and function conflict, the part does not
exist, or the contact is clearly implausible for the stated function.

Return JSON only: {"score": 0|1|2, "reason": "one concise sentence"}""",

    'multi': """You are shown one object from eight rendered views and a proposal for using two hands to perform a task. Judge the proposal, not whether the object itself is practical to handle. Score each item independently.

Treat the object as movable and operable. Apparent weight, size, mounting, installation, solidity and unseen construction are outside the judgment. Every visible part may be touched and acted on.

The views establish visible geometry, not hidden mechanism. Failure to see a joint, seam, gap, release or other mechanism does not establish that it is absent. A feature is absent only when its relevant region is shown clearly enough and the visible geometry affirmatively contradicts it.

Only the two ends carry weight. 2 is for a description that is clean: everything it names is there,
everything it says is borne out, nothing is left loose. 0 is for one that is visibly broken. 1 is the
whole middle - it holds together, but something about it is loose, and it is where anything you
cannot settle belongs.

2 is earned, not assumed. Give it when you have checked each thing the description claims and found
each one to hold - not when nothing happened to catch your eye. If you would have to say "probably"
about any part of it, that part is loose, and the answer is 1.

Each hand action is the nearest primitive from this fixed vocabulary: hold, lift, push, pull, press, slide, rotate, squeeze. It need only contribute to the task, not describe the complete task motion. The stated function must nevertheless be a plausible consequence of that action at that target and contact; merely repeating the task is not a function.

Judge whether both hands can act simultaneously at reachable, distinct contact regions. Matching actions are valid, and holding, supporting or steadying is a real contribution. Touching the same object is not a conflict. A role conflict exists only when the descriptions require one hand to prevent the exact motion the other must produce; do not infer one from presumed attachment or construction.

Use this scale:
2 = you checked each claim and each one holds: everything named is present, the contacts are where
    they are said to be, and the stated roles do what they claim. Nothing is left uncertain.
1 = it holds together, but something is loose: a contact placed only vaguely, a part you can see but
    cannot delineate, a division of work that is marginal, or wording too general to check.
0 = the views affirmatively contradict the description, or the stated roles are mutually impossible.

Apply it separately:
task: Does the task act on something this object has? A different object or a feature clearly shown absent is contradictory.
bimanual: Do the two hand contributions combine into one coherent operation?
hand_A / hand_B: Does that hand's action, target, contact and function form one coherent contribution?""",

    'direct': """You are shown one object from eight views and a proposed two-handed interaction.

The only available hand actions are hold, lift, push, pull, press, slide, rotate, and squeeze. Judge the proposal with these three checks:

1. GROUNDING - Actively locate visual support for every named part and contact in the eight views. Do not assume a part exists merely because that kind of object might have one. If no view supports its presence, treat it as absent; if it is merely hard to distinguish, treat that as uncertainty.

2. ACTION FIT - Decide whether the primitives directly and precisely capture the task-relevant motion of the hands. A primitive is approximate when the proposal works only by treating it as the nearest available stand-in for an essential motion the vocabulary cannot express. A sensible approximation may score 1, but a vocabulary limitation alone is not grounds for 0.

3. COORDINATION - Judge the forces and constraints created by both contacts together. Matching primitives are not conflicting merely because they match; they may reinforce one another. A supporting or stabilizing hand contributes only if it leaves the task-required motion free. A real, reachable contact still fails if it restrains, locks, blocks, or opposes motion required by the task. Call a conflict only when the contacts and roles establish such a mechanism.

Score:
2 = All parts and contacts are visually grounded, the primitives directly describe the useful hand motions, and the hands clearly coordinate toward the task.
1 = The interaction is viable and non-blocking, but a primitive is an acceptable stand-in for unavailable motion, or grounding, placement, or coordination is materially vague.
0 = A required part is absent, the object cannot support the task, a contact is unreachable, a hand does not plausibly contribute even under the coarse vocabulary, or either hand specifically blocks or opposes required motion.

Return JSON only: {"score": 0|1|2, "reason": "one concise sentence"}.""",

    'mcq': """Look at all eight views and choose exactly one answer.

The available hand actions - hold, lift, push, pull, press, slide, rotate, squeeze - are a coarse
vocabulary. Read each as the nearest available description of that hand's contribution. An action
need not complete or exactly name the whole task, and two hands using the same action is the
ordinary way people carry, lift and steady things.

Two rules govern every answer:

A part that is not on this object is an error. A part that plausibly exists but is hidden, occluded
or too small to resolve in these views is uncertainty, not an error - check every view before
concluding a part is absent.

A hand's stated function must be the consequence of its action applied at its stated contact, not a
restatement of the task. "Helps accomplish the task" is not a function.

Choose a breakage only when you can state its mechanism in your reason - which force opposes which,
which part is missing, what blocks what.

fully_sound is not "no breakage found". Before choosing it, confirm positively, hand by hand: you
located that hand's named part in the views, its contact is somewhere a hand can reach, its action
word is what that hand really does rather than the closest available substitute, and its function
follows from that action at that contact. If you cannot confirm all of this for BOTH hands, the
answer is acceptable_imprecise.

Judge the object in front of you, not the object category in general.


- task_not_possible: the object's visible structure makes the task impossible whatever the hands do.
  Not for a task the primitives merely fail to name literally.

- part_absent: after actively checking all eight views, a part the proposal needs is clearly not
  there. Do not infer a part from the object category; a part that is merely hard to delineate is
  acceptable_imprecise.

- forces_cancel: the two contacts create forces or torques that oppose each other on the required
  movement and so prevent it. Matching actions, symmetric contacts, or both hands pushing together
  do not by themselves establish cancellation.

- anchor_moves: the supporting hand cannot do its job - it travels with what it should hold still,
  or it restrains a part that has to move. Touching a moving part is not enough; the lost anchor or
  the locked motion must be clear.

- action_cannot_produce_effect: one hand, at its stated contact, cannot make any positive
  contribution to the task, even read as the nearest available word and allowing force through
  connected parts. Not for an action that simply does not complete the task alone.

- hand_obstructs: one hand occupies or blocks the contact or path the other needs, so the two cannot
  happen together. Proximity in the render is not enough.

- acceptable_imprecise: nothing is broken, but you cannot confirm it is right. The views are
  insufficient, the relevant part is occluded or too small to resolve, the manner of operation is
  uncertain, the contact is described too vaguely to check, or the action is only the nearest
  available word for a motion this vocabulary cannot express. This is the answer whenever the
  evidence does not settle the question.

- fully_sound: every named part is visible, both contacts are reachable, each action is the direct
  ordinary word for what that hand does, and the two hands work together toward the task.

Return JSON only:
{"finding": "<one id above>", "reason": "one concise sentence stating the mechanism, or why it is sound"}""",
}


def build_item(row: dict) -> WorkItem:
    meta_path = row["meta_path"]
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    roles = meta.get("roles")
    if not isinstance(roles, list) or len(roles) != 2:
        raise ValueError("meta.roles must contain exactly two hands")

    def hand(label: str, value: dict) -> str:
        return "\n".join([
            f"Hand {label}:",
            f"  action: {value.get('role')}",
            f"  target/contact: {value.get('target')} / {value.get('contact_region')}",
            f"  function: {value.get('function')}",
        ])

    relation = meta.get("relation") or "not specified"
    text = "\n\n".join([
        f"Object: {meta.get('object_name')}\nTask: {meta.get('task')}",
        hand("A", roles[0]),
        hand("B", roles[1]),
        f"How the hands work together: {relation}",
        "Return the required JSON.",
    ])
    return WorkItem(
        sample_id=row["sample_id"], object_id=row["object_id"],
        image_path=row["rgb_png"], user_text=text,
    )


if __name__ == "__main__":
    verifier_main(AXIS, VERSION, PROMPTS, build_item, FINDINGS, MULTI_AXES)

