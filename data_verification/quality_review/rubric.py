"""Release-quality rubric: the single source of truth for the review UI, the export, and the
paper appendix.

Three axes, each scored Bad / OK / Good (0 / 1 / 2) and revealed in order, so every judgement
sees only the evidence it needs: the task is judged from the RGB views alone, the roles are
added next, and the heatmaps last. The heatmap verdicts and tags are kept verbatim from the
pilot heatmap UI (`stage2-human-review-v1.0`) so the new labels stay comparable with the
pilot human gold and with the verifier tooling that reuses that wording.
"""

from __future__ import annotations

RUBRIC_VERSION = "release-quality-rubric-v1.1"  # v1.1 adds task reason not_meaningful

SCALE = {
    0: {"name": "Bad", "key": "B"},
    1: {"name": "OK", "key": "O"},
    2: {"name": "Good", "key": "G"},
}

PRINCIPLES = [
    "Good means usable as-is, with no observable defect. OK marks a genuine, visible "
    "imperfection or uncertainty, not a default. Bad needs positive visible evidence of an error.",
    "Each step reveals new evidence. Score what that step adds; do not mark a later step down "
    "only because an earlier step was wrong.",
    "Judge from what is visible. Names and descriptions can be imperfect; when text and views "
    "disagree, the views decide.",
]

AXES = [
    {
        "id": "task",
        "title": "Task validity",
        "short": "Task",
        "question": "Is this a sensible two-handed task for this object?",
        "evidence": "RGB views, object name, task",
        "verdicts": {
            2: "The object visibly affords this task, and two hands are natural or clearly useful.",
            1: "Plausible but vague or unusual, or the second hand adds little.",
            0: "Impossible or incoherent for this object, or it acts on a part the object lacks.",
        },
        "tags": [
            ("implausible_task", "Object cannot afford this task"),
            ("not_meaningful", "Not a meaningful task"),
            ("part_absent", "Acts on a part the object lacks"),
            ("forced_bimanual", "One hand would do"),
            ("too_vague_to_verify", "Too vague or ambiguous to judge"),
            ("other", "Other (add a note)"),
        ],
    },
    {
        "id": "role",
        "title": "Role assignment",
        "short": "Roles",
        "question": "Do the two hand roles divide this task correctly?",
        "evidence": "+ each hand's action, target part, contact region and function",
        "verdicts": {
            2: "Each action, at its named region, achieves its function; together the roles "
               "complete the task.",
            1: "Workable, but a region is vague or awkward, or a function only loosely serves "
               "the task.",
            0: "A named part is absent, an action cannot achieve its function, or the roles "
               "conflict or miss the task.",
        },
        "tags": [
            ("contact_part_absent", "Named part or region not on the object"),
            ("role_function_conflict", "Action cannot achieve its function there"),
            ("role_task_conflict", "Roles do not combine into the task"),
            ("too_vague_to_verify", "Region too vague to locate"),
            ("other", "Other (add a note)"),
        ],
    },
    {
        "id": "heatmap",
        "title": "Paired heatmaps",
        "short": "Heatmaps",
        "question": "Do the orange (A) and teal (B) regions mark usable contacts for the task?",
        "evidence": "+ Hand A and Hand B heatmaps, rendered from the same cameras as the RGB views",
        "guidance": "Brighter means more likely contact. Role text is a fallible reference, not an "
                    "exact mask; judge the visible regions against the task. Region area alone "
                    "is not evidence of quality.",
        "verdicts": {
            2: "Task-usable, formed, and no observable defect.",
            1: "Recoverable, but has a visible defect or uncertainty.",
            0: "Clear miss, contradiction, absence, or unusable field.",
        },
        "tags": [
            ("missing_or_unusable", "Missing, too faint, or render unusable"),
            ("fragmented_or_incomplete", "Fragmented, incomplete, or not formed"),
            ("diffuse_overflow_or_ambiguous", "Diffuse, overflow, or ambiguous candidates"),
            ("wrong_or_implausible_region", "Wrong or implausible task region"),
            ("hand_or_joint_task_inconsistency", "One-hand or joint-task inconsistency"),
            ("insufficient_evidence", "Insufficient visual evidence"),
            ("other", "Other (add a note)"),
        ],
    },
]

AXIS_IDS = tuple(axis["id"] for axis in AXES)
AXIS_TAGS = {axis["id"]: {tag for tag, _ in axis["tags"]} for axis in AXES}


def rubric_payload() -> dict:
    """JSON-serialisable rubric for the frontend and for provenance records."""
    return {
        "version": RUBRIC_VERSION,
        "scale": {str(value): spec for value, spec in SCALE.items()},
        "principles": PRINCIPLES,
        "axes": [
            {
                **{key: value for key, value in axis.items() if key not in ("verdicts", "tags")},
                "verdicts": {str(value): text for value, text in axis["verdicts"].items()},
                "tags": [{"id": tag, "label": label} for tag, label in axis["tags"]],
            }
            for axis in AXES
        ],
    }
