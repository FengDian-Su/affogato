#!/usr/bin/env python
"""
Stage 1 — task proposal + coordination-aware role decomposition + assemble.

Input: stage0_filter_components.py output (kept objects with components).
Per object (multi-pass proposer + one strict curator; no intermediate taxonomy):
  Round 1: broad BRAINSTORM of normal designed operations (a valid task = a normal user operation the
     object is DESIGNED to support without damage, tools, external placement, or completing an
     external action) in THREE families: object_level / component_level / functional_pose
     (functional_pose = both hands put the object into a designed USE POSE - tilt to pour, hold level
     to carry - without completing the external action or adding contents)
  -> Rounds 2..N: RETRY rounds do a component sweep + functional-pose sweep (avoid-list = all
     proposed + universal); a dry retry stops that object early
  -> ONE strict CURATE composes the final set (validation / semantic dedup / family quotas)
  -> per task: role decomposition into 2 hand-agnostic robot-executable roles (+contact_region)
               + pattern + role-conditioned Molmo instructions
  -> emit a query-ready record per task.

Output: one entry per object: {object_id, object_name, views_used, components, queries:[...]}.
Each query = {task, query, goal, roles, relation, pattern, molmo_queries, answer, ...}.

Reuses bimanual_annotation/gemma.py. Run with the **gemma4** env.

  CUDA_VISIBLE_DEVICES=3 \
  /home/michaellee/miniconda3/envs/gemma4/bin/python \
      data_generation/pipeline/stage1_v2.py \
      --in outputs/stage0_filtered.json --out outputs/stage1_dataset.json
"""
import os
import sys
import json
import math
import argparse
import subprocess
import shutil
import time

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate corrupt/truncated render PNGs (libpng CRC errors)

HERE = os.path.dirname(os.path.abspath(__file__))            # data_generation/pipeline/


def _find_repo_root(d):
    """Walk up until we find the repo (the dir containing bimanual_annotation/)."""
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "bimanual_annotation")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.dirname(HERE))


REPO_ROOT = _find_repo_root(HERE)
DATA_GEN = os.path.join(REPO_ROOT, "data_generation")
BIMANUAL_DIR = os.path.join(REPO_ROOT, "bimanual_annotation")
# per-process temp dir for cleaned views (concurrent shards must NOT share one); under /tmp so it
# never clutters the repo
VIEW_DIR = os.path.join("/tmp", f"tmp_views_stage1_{os.getpid()}")

ROLE_VERBS = ["hold", "lift", "push", "pull", "press", "slide", "rotate", "squeeze"]

ROLE_GLOSS = (
    "hold = hand contacts the object to stabilize, support, provide reaction force, or maintain pose without driving any state change itself; "
    "lift = hand grasps or cups the object/part and applies force against gravity to raise it clear of its support surface; "
    "push = hand applies a linear force directed away from itself, causing the object or part to translate, close, or shift position; contact is a force-application point, not a sustained pressed-against surface; "
    "pull = hand applies a traction force toward itself or away from the object's base, causing the part to be drawn out, opened, or separated; "
    "press = fingers or palm apply short-range, localised, inward normal force to trigger, activate, or displace a small-travel part; "
    "slide = hand presses the object or part against a plane, track, or surface and moves it along that surface in the tangential direction while maintaining normal contact pressure throughout; "
    "the defining feature is the combination of sustained surface contact and constrained tangential translation; "
    "rotate = hand applies torque to produce angular displacement about an axis; covers turn, twist, flip, and tilt as long as the primary change is rotational; "
    "squeeze = fingers, palm, or both hands apply inward compressive force from two or more sides, causing deformation, mechanism closure, content expulsion, or clamp closure; "
)


# ---------------------------------------------------------------- helpers
def parse_json(text):
    try:
        s = text.find("{"); e = text.rfind("}") + 1
        return json.loads(text[s:e])
    except Exception as ex:
        return {"error": f"parse failed: {ex}", "raw": text}


# Re-save each view as a clean PNG in an isolated subprocess (corrupt render -> hard SIGBUS crash,
# uncatchable in-process). Timeout guards hung NFS reads. Returns the clean paths that survived.
_PREP_WORKER = (
    "import sys, os\n"
    "from PIL import Image, ImageFile\n"
    "ImageFile.LOAD_TRUNCATED_IMAGES = True\n"
    "out_dir = sys.argv[1]; views = sys.argv[2:]\n"
    "for i, p in enumerate(views):\n"
    "    try:\n"
    "        op = os.path.join(out_dir, f'v{i:02d}.png')\n"
    "        Image.open(p).convert('RGB').save(op)\n"
    "        print(f'{i}\\t{op}', flush=True)\n"
    "    except Exception:\n"
    "        pass\n"
)


def safe_prepare_views(views, out_dir, timeout=120):
    """Re-save views as clean PNGs in a subprocess; return [(orig_index, clean_path), ...] survivors."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass
    out = []
    try:
        r = subprocess.run([sys.executable, "-c", _PREP_WORKER, out_dir, *views],
                           capture_output=True, text=True, timeout=timeout)
        for ln in r.stdout.splitlines():
            parts = ln.strip().split("\t")
            if len(parts) == 2 and parts[1].endswith(".png") and os.path.exists(parts[1]):
                out.append((int(parts[0]), parts[1]))
    except subprocess.TimeoutExpired:
        return []
    return out


def view_label(view_path):
    """Human-readable viewpoint label for a gObjaverse view, from its camera json."""
    d = os.path.dirname(view_path)
    idx = os.path.basename(d)
    try:
        o = json.load(open(os.path.join(d, f"{idx}.json"))).get("origin")
        r = math.sqrt(sum(t * t for t in o)) or 1.0
        elev = math.degrees(math.asin(o[2] / r))
        az = math.degrees(math.atan2(o[1], o[0])) % 360
    except Exception:
        return "A view of the object."
    if elev > 60:
        return ("TOP-DOWN view: camera looking straight DOWN on the TOP of the object "
                "(use it to see lids, openings, controls, or anything on the top face).")
    if elev < -60:
        return "UNDERSIDE view: camera looking straight UP at the BOTTOM / base of the object."
    return (f"Oblique view from ~{round(elev)} deg above, rotated to azimuth ~{round(az)} deg around "
            f"the object (shows its top edge plus the side facing the camera).")


def components_to_text(components):
    return "\n".join(f"- {c['name']}: {c.get('interaction', '')}" for c in components)


def candidates_to_text(cs):
    return "\n".join(f"- {c.get('task')} ({c.get('goal')})" for c in cs) or "- (none)"


def to_query(task):
    """Fixed dataset-query wrapper (affogato-parallel, task-centric, hand-free):
    'Point to the regions of the object you would use to <task>.'"""
    t = str(task).strip().rstrip(".")
    if t:
        t = t[0].lower() + t[1:]
    return f"Point to the regions of the object you would use to {t}."


def universal_tasks(object_name, size_class="two-handed"):
    """The always-added WHOLE-OBJECT TRANSPORT tasks, grounded in HOI action-frequency stats (pick-up/take
    is #1-2 in every egocentric dataset; move/relocate is RT-1's single largest skill and the whole of
    Housekeep). Every object gets MOVE; non-furniture also gets PICK UP (furniture can't be hand-lifted).
    The two are kept physically DISTINCT: pick up = acquire + LIFT clear of the support; move = change the
    support LOCATION by the SIZE-appropriate mode (slide / two-hand carry / push-roll) WITHOUT necessarily
    lifting clear. No size hedge-boilerplate - the WHY is one honest, size-specific reason."""
    o = str(object_name).strip()
    sc = str(size_class).lower()
    if "furniture" in sc:
        return [{
            "task": f"move the {o}", "category": "inter",
            "goal": f"{o} in place -> {o} pushed / rolled to a new position (stays on the floor)",
            "why_bimanual": (f"the {o} is too large to lift, so both hands push and steer its body together "
                             f"to slide or roll it to a new spot without it veering"),
        }]
    pick = {
        "task": f"pick up the {o}", "category": "inter",
        "goal": f"{o} resting -> {o} lifted clear of the surface and held",
        "why_bimanual": (f"raising it clear and level beats gravity only with a weight-bearing hold; one "
                         f"hand cannot both bear the load and keep it from tipping, so two hands must lift "
                         f"and steady it together, or it tips or slips"),
    }
    if "hand-sized" in sc:
        # a hand-sized object is moved/slid one-handed, so MOVE is not a genuine two-hand task here;
        # its real two-handed operations (open / pour / twist / press) come from the mechanism rounds.
        return [pick]
    move = {   # two-handed: a genuine two-hand carry (both hands needed to keep it level)
        "task": f"move the {o}", "category": "inter",
        "goal": f"{o} at one spot -> {o} carried to a new spot and set down",
        "why_bimanual": (f"both hands carry the {o} together at two opposite sides so it stays level "
                         f"and balanced while it is translated"),
    }
    return [pick, move]


def ensure_body_component(object_name, comps):
    """Guarantee a groundable RIGID-BODY part exists. Many objects' component lists contain only
    movable sub-parts (lid, flap, knob), which forces whole-object lifts to (wrongly) target a lid.
    Prepend an explicit body/shell so the decomposer can ground 'pick up / carry / flip' on it."""
    body_words = ("body", "shell", "base", "chassis", "frame", "main")
    has_body = any(any(w in str(c.get("name", "")).lower() for w in body_words) for c in comps)
    if has_body:
        return comps
    o = str(object_name).strip().lower()
    body = {"name": "body",
            "interaction": (f"the main rigid body/shell of the {o}; grip its outer walls here to "
                            f"lift, carry, flip, or steady the whole object")}
    return [body] + list(comps)




_PLACEHOLDER_BITS = ("<", "...", "where hand a grips", "different place from a",
                     "a part copied", "one verb from", "specific spot on")


def _is_placeholder(s):
    s = str(s or "").strip().lower()
    return (not s) or s == "..." or any(b in s for b in _PLACEHOLDER_BITS)


def valid_decomp(roles):
    """Reject degenerate decompositions where the model echoed the JSON-template placeholders
    (e.g. contact_region == 'where hand A grips') or left a field empty / identical."""
    if not isinstance(roles, list) or len(roles) != 2:
        return False
    for r in roles:
        if not isinstance(r, dict):
            return False
        if (_is_placeholder(r.get("role")) or _is_placeholder(r.get("target"))
                or _is_placeholder(r.get("contact_region"))):
            return False
    a = str(roles[0].get("contact_region", "")).strip().lower()
    b = str(roles[1].get("contact_region", "")).strip().lower()
    return not (a and a == b)


def normalize_pair(dec):
    roles = dec.get("roles", [])
    if len(roles) != 2:
        return {"pattern": None, "relation": dec.get("relation"), "symmetric": None, "hand_agnostic": True}
    ra, rb = roles[0].get("role"), roles[1].get("role")
    return {"pattern": f"{ra} + {rb}", "relation": dec.get("relation"),
            "symmetric": ra == rb, "hand_agnostic": True}


def molmo_instruction(role):
    """'Point to <region> where a robot hand should <verb> it to <function>.'"""
    region = str(role.get("contact_region") or role.get("target") or "").strip().rstrip(".")
    region_phrase = region if region.lower().startswith("the ") else f"the {region}"
    verb = str(role.get("role", "")).strip()
    fn = str(role.get("function") or "").strip().rstrip(".")
    q = f"Point to {region_phrase}"
    if verb:
        q += f" where a robot hand should {verb} it"
        if fn:
            q += f" to {fn}"
    return q + "."


# ---------------------------------------------------------------- prompts
# Multi-pass proposer + one strict curator. NO intermediate taxonomy (seed/mechanism audits were
# tried and each taxonomy word became a junk-legalization channel: containment->fill,
# orientation->flip, attachment->detach-fused-parts). Coverage comes from SEARCH PRESSURE instead:
# one broad brainstorm round + retry rounds that hunt overlooked designed operations. Core validity
# principle everywhere: a task is a normal user operation the object is DESIGNED to support without
# damage, tools, added objects, or external placement.
TASK_BRAINSTORM_PROMPT = {
    "system": (
        "You are an annotation assistant for a 3D bimanual affordance dataset. "
        "Propose daily bimanual manipulation tasks for one object. "
        "A valid task is a normal user operation that the object is designed to support "
        "without damage, tools, external placement, or completing an external action. "
        "Do not generate filler tasks; returning empty lists is correct when no good task exists."
    ),

    "user": (
        "Object: {object_name}\n"
        "The images show this object from multiple views.\n"
        "Groundable components:\n"
        "{components}\n\n"

        "Current contents, if explicitly visible or listed:\n"
        "{current_contents}\n\n"

        "Already proposed tasks to avoid repeating:\n"
        "{avoid}\n\n"

        "Propose up to {n_object} object_level tasks, up to {n_component} component_level tasks, "
        "and up to {n_pose} functional_pose tasks. These are upper bounds, not quotas.\n\n"

        "Task families:\n"
        "A) object_level: the object is manipulated mainly as a whole object. "
        "The outcome is an intrinsic change of the object's configuration or contents already held by it. "
        "Generic motion, placement, display, transport, or camera-facing orientation is not enough.\n"
        "B) component_level: one or more existing components change state, configuration, access, attachment, "
        "or relation to another part of the same object. "
        "The component must be visible, listed, or clearly designed to move or separate.\n"
        "C) functional_pose: the object is put into an object-side use pose that enables an external goal. "
        "The task changes the object's own pose, shape, or functional interface, such as being level, upright, tilted, aimed, "
        "spread, taut, opened, or aligned for use. "
        "It does not perform the external action, add the external object/material/content, or depend on a surface or environment.\n\n"

        "Think like a person using the object normally. "
        "Look for designed operations the object visibly supports: open/close, lock/unlock, fold/unfold, roll/unroll, "
        "extend/retract, tighten/loosen, adjust a built-in part, rotate a built-in axis, operate a handle/knob/button/plug/latch, "
        "remove/replace a removable part, release/pour/remove contents already held by the object, "
        "or put the object into a designed functional pose.\n\n"

        "For functional_pose, prefer diverse object-side use poses when possible: "
        "receiving, dispensing, aiming, supporting a load-bearing surface, exposing a working interface, "
        "keeping a surface level, keeping an opening accessible, spreading a flexible body, or keeping it taut. "
        "The goal must describe the object's pose, shape, or interface state, not how many hands hold it. "
        "Both hands should act on the target object itself, but hand placement belongs only in why_bimanual, not in the task or goal. "
        "Do not use functional_pose for display, inspection, placement, storage, camera-facing orientation, generic flipping, "
        "or merely stabilizing/supporting the object without changing its object-side use pose.\n\n"

        "Each task must have a concrete goal written as '<before> -> <after>'. "
        "For object_level and component_level, the changed thing must be the object itself, an existing component, "
        "or contents already held by the object. "
        "For functional_pose, the changed thing must be only the object's own pose, shape, or functional interface. "
        "A goal like '<held by one hand> -> <held by two hands>' is invalid because only the hand arrangement changes.\n\n"

        "Do not propose tasks that mainly add a new liquid, material, object, workpiece, or content. "
        "Do not propose tasks that are mainly grasping, holding without a functional pose, stabilizing, inspecting, showing, cleaning, "
        "placing, moving, preparing, handoff, blocking, obstructing, changing viewpoint/pose for visibility, "
        "or detaching fixed structural parts.\n\n"

        "Prefer component_level tasks when meaningful components exist. "
        "Use functional_pose only for natural bimanual use poses that the object clearly supports. "
        "If the object has several designed operations, include several distinct tasks. "
        "If it has only one or two real operations, return only those.\n\n"

        "Set is_scene true only if the input is a multi-object scene rather than one object.\n"
        "Report size_class from real-world knowledge of the object: "
        "\"hand-sized\", \"two-handed\", or \"furniture-scale\".\n\n"

        "Return JSON only:\n"
        "{{\n"
        "  \"is_scene\": false,\n"
        "  \"size_class\": \"hand-sized | two-handed | furniture-scale\",\n"
        "  \"object_level\": [\n"
        "    {{\"task\": \"...\", \"goal\": \"<before> -> <after>\"}}\n"
        "  ],\n"
        "  \"component_level\": [\n"
        "    {{\"task\": \"...\", \"goal\": \"<before> -> <after>\"}}\n"
        "  ],\n"
        "  \"functional_pose\": [\n"
        "    {{\"task\": \"...\", \"goal\": \"<before> -> <after>\"}}\n"
        "  ]\n"
        "}}\n"
        "Output JSON only."
    ),
}

TASK_BRAINSTORM_RETRY_PROMPT = {
    "system": (
        "You are an annotation assistant for a 3D bimanual affordance dataset. "
        "Find overlooked daily bimanual tasks for one object. "
        "A valid task is a normal user operation that the object is designed to support "
        "without damage, tools, external placement, or completing an external action. "
        "Return only new valid tasks, not variations of existing ones."
    ),

    "user": (
        "Object: {object_name}\n"
        "The images show this object from multiple views.\n"
        "Groundable components:\n"
        "{components}\n\n"

        "Current contents, if explicitly visible or listed:\n"
        "{current_contents}\n\n"

        "Already accepted or proposed tasks:\n"
        "{avoid}\n\n"

        "Do a component sweep: for each visible or listed component, ask whether an ordinary person would operate it by hand. "
        "Look especially for handles, knobs, buttons, plugs, caps, lids, flaps, hinges, latches, clasps, straps, pages, folds, seams, "
        "sliders, adjustable parts, removable parts, or built-in rotating parts. "
        "Skip fixed structural parts.\n\n"

        "Also do a functional-pose sweep: ask whether two hands would naturally put the object into an object-side use pose "
        "for receiving, dispensing, aiming, supporting a load-bearing surface, exposing a working interface, keeping a surface level, "
        "keeping an opening accessible, spreading a flexible body, or keeping it taut. "
        "The task may mention the external goal, but the goal must change the object's own pose, shape, or functional interface. "
        "It must not merely change how the hands hold, support, or stabilize the object, and it must not complete the external action.\n\n"

        "Task families:\n"
        "A) object_level: whole-object intrinsic configuration or already-held-content change. "
        "Generic motion, placement, display, transport, or camera-facing orientation is not enough.\n"
        "B) component_level: existing components change state, configuration, access, attachment, or relation to another part of the same object.\n"
        "C) functional_pose: the object is put into a designed use pose or functional condition. "
        "Both hands should act on the target object itself, and the pose must involve a functional interface of the object.\n\n"

        "Each task must have a concrete '<before> -> <after>' goal verifiable from the object itself. "
        "The task must describe the intended object outcome or object-side use pose, not hand placement or role decomposition.\n\n"

        "Reject tasks involving new objects, new materials, new contents, tools, surfaces, people, environments, "
        "showing, inspecting, cleaning, placing, moving, preparing, blocking, obstructing, stabilizing, handoff, "
        "viewpoint/pose changes for visibility, detaching fixed structural parts, or completing the external action itself. "
        "Reject goals where the only change is hand arrangement, such as one-hand to two-hand holding or "
        "unsupported to hand-stabilized.\n\n"

        "Propose up to {n_object} additional object_level tasks, up to {n_component} additional component_level tasks, "
        "and up to {n_pose} additional functional_pose tasks. Return empty lists if no overlooked valid tasks exist.\n\n"

        "Return JSON only:\n"
        "{{\n"
        "  \"object_level\": [\n"
        "    {{\"task\": \"...\", \"goal\": \"<before> -> <after>\"}}\n"
        "  ],\n"
        "  \"component_level\": [\n"
        "    {{\"task\": \"...\", \"goal\": \"<before> -> <after>\"}}\n"
        "  ],\n"
        "  \"functional_pose\": [\n"
        "    {{\"task\": \"...\", \"goal\": \"<before> -> <after>\"}}\n"
        "  ]\n"
        "}}\n"
        "Output JSON only."
    ),
}

TASK_CURATE_PROMPT = {
    "system": (
        "You are a strict curator for a 3D bimanual affordance dataset. "
        "Keep only realistic daily bimanual tasks. "
        "A valid task is a normal user operation that the object is designed to support "
        "without damage, tools, external placement, or completing an external action. "
        "When uncertain, reject."
    ),

    "user": (
        "Object: {object_name}\n\n"

        "Groundable components:\n"
        "{components}\n\n"

        "Current contents, if explicitly visible or listed:\n"
        "{current_contents}\n\n"

        "Fixed tasks already in the final set. Do not repeat or rephrase them:\n"
        "{universal}\n\n"

        "Raw candidate tasks:\n"
        "object_level candidates:\n"
        "{object_level}\n\n"
        "component_level candidates:\n"
        "{component_level}\n\n"
        "functional_pose candidates:\n"
        "{functional_pose}\n\n"

        "Compose the final task list.\n\n"

        "Task families:\n"
        "A) object_level: the object is manipulated mainly as a whole object. "
        "The outcome is an intrinsic change of the object's configuration or contents already held by it. "
        "Generic motion, placement, display, transport, or camera-facing orientation is not enough.\n"
        "B) component_level: one or more existing components change state, configuration, access, attachment, "
        "or relation to another part of the same object. "
        "The component must be visible, listed, or clearly designed to move or separate.\n"
        "C) functional_pose: the object is put into an object-side use pose that enables an external goal. "
        "The task changes the object's own pose, shape, or functional interface, such as being level, upright, tilted, aimed, "
        "spread, taut, opened, or aligned for use. "
        "It does not perform the external action, add the external object/material/content, or depend on a surface or environment.\n\n"

        "Keep a candidate only if:\n"
        "- its goal is a concrete '<before> -> <after>' change verifiable from the object itself;\n"
        "- it is a normal operation an ordinary person would intentionally do to this object;\n"
        "- the object visibly supports the change without damage;\n"
        "- for object_level or component_level, the changed thing is the object itself, an existing component, "
        "or contents already held by the object;\n"
        "- for functional_pose, the changed thing is only the object's own pose, shape, or functional interface; "
        "both hands act on the target object, but the task/goal must not be about hand arrangement; "
        "the external goal is not completed by the task;\n"
        "- it naturally requires or strongly benefits from two hands;\n"
        "- it is not already covered by a fixed task or another kept task.\n\n"

        "Reject tasks whose main outcome is adding a new liquid, material, object, workpiece, or content. "
        "Reject showing, inspecting, cleaning, placing, storing, transporting, generic flipping, camera-facing orientation, "
        "hand placement, grasping, stabilization, blocking, obstructing, vague preparation, handoff, "
        "new tools/surfaces/environments, detaching fixed structural parts, or completing an external action itself.\n\n"

        "Reject functional_pose tasks whose before-to-after goal only changes the hand arrangement, "
        "such as one-hand to two-hand holding, unsupported to hand-stabilized, or one hand on part A and another hand on part B. "
        "Hand contact details belong in why_bimanual, not in the task or goal.\n\n"

        "For functional_pose, keep diverse pose types when possible instead of many upright/tilt variants. "
        "Prefer poses involving receiving, dispensing, aiming, supporting, exposing a working interface, keeping a surface level, "
        "keeping an opening accessible, or keeping a flexible body taut. "
        "Reject functional_pose tasks that are merely display, inspection, placement, or viewpoint change.\n\n"

        "Prioritize component_level tasks because they are the main focus of the dataset. "
        "Keep object_level tasks when they add meaningful object-intrinsic diversity. "
        "Keep functional_pose tasks only when they are natural bimanual use poses and not display/placement/viewpoint tasks. "
        "Keep up to {n_component} component_level tasks, up to {n_object} object_level tasks, "
        "and up to {n_pose} functional_pose tasks. Return fewer if fewer valid tasks exist.\n\n"

        "Rewrite kept tasks into simple daily task phrases. "
        "Order component_level tasks first, then object_level tasks, then functional_pose tasks.\n\n"

        "Return JSON only:\n"
        "{{\n"
        "  \"tasks\": [\n"
        "    {{\n"
        "      \"task\": \"...\",\n"
        "      \"goal\": \"<before> -> <after>\",\n"
        "      \"why_bimanual\": \"...\",\n"
        "      \"category\": \"object_level | component_level | functional_pose\"\n"
        "    }}\n"
        "  ]\n"
        "}}\n"
        "Output JSON only."
    ),
}


# ---------------------------------------------------------------- per-object pipeline

# Family quota: at most this many whole-object (inter) tasks per object, universal transport INCLUDED,
# so the rest of the budget is GUARANTEED to part-level (intra) tasks - the dataset's core examples.
# Without this quota the intra share measurably collapses (45% -> 30%, part0 audit 2026-07-08).
INTER_CAP = 3


# ---- FACTORED decomposition: call 1 derives the operation-mechanics PLAN (focused physics, no role
# formatting), call 2 GROUNDS that plan into two roles. Isolating the physics keeps the model from
# letting the big formatting prompt dilute its reasoning (probe: latches reasoned correctly when focused).
OP_PLAN_PROMPT = {
    "system": (
        "You are a manipulation-mechanics expert. For ONE task on ONE object you work out ONLY the "
        "physics of performing it - which part moves, which way it must go, where to grip it for "
        "leverage, and what the other hand must hold. Think CONCISELY - a few short reasoning lines are "
        "enough, do NOT over-deliberate or second-guess. You do NOT assign hand-roles yet."
    ),
    "user": (
        "Object: {object_name}\nTask: {task}\nGoal: {goal}\nWhy bimanual: {why_bimanual}\n"
        "Visible parts (use a name from this list verbatim; only parts you can SEE are real):\n{components}\n"
        "Allowed verbs: {role_verbs}\nVerb meanings: {role_gloss}\n\n"
        "STEP 1 - MECHANISM + SCALE. (a) MOVING ELEMENT vs STABLE FRAME: what does the goal relocate or "
        "reconfigure - the whole body, a part, or the CONTENTS inside - and what must stay FIXED as the "
        "reference. If the goal moves the CONTENTS (pour / empty / tip out), the contents are the moving "
        "element and the BODY is the stable frame: one hand anchors the body, the other tilts it - do NOT "
        "have both hands rotate the whole body. Name the rigid BODY and which part, if any, must MOVE - "
        "usually ONE part; a MATCHING PAIR (two flaps, two handles) when the task drives both at once "
        "(or 'nothing moves' for a plain carry/flip/move); (b) the object's real-world SIZE CLASS, "
        "judged from WHAT THE OBJECT IS (views are SIZE-NORMALIZED, do not read size off the image): "
        "hand-sized / two-handed / furniture-scale.\n"
        "STEP 2 - OPERATION MECHANICS (only if a part MOVES):\n"
        "  (i) ALLOWED MOTION: it has ONE way - swing on a hinge (arc) / slide along a track / twist about "
        "an axis through its face. Name which, and where the hinge / track / axis is.\n"
        "  (ii) DIRECTION TO GOAL: which way along that motion takes the part from its CURRENT to its GOAL "
        "state (a closed latch must move UP off its catch; an upright bowl's rim must rotate DOWN to pour; "
        "a shut drawer must come straight OUT). The acting verb is THIS motion - and the verb must produce "
        "force IN that direction: an inward push on a vertical face CANNOT lift (raising by two opposite "
        "faces needs grip/clamp so friction bears the weight, not push).\n"
        "  (iii) PURCHASE: the contact FARTHEST from the hinge/axis, on a rigid surface - the FREE edge of "
        "a lid (never near the hinge), the RIM of a cap (never its flat top, which is on the axis), the "
        "handle/front of a drawer. Reject any spot on the pivot/axis or a floppy surface.\n"
        "  (iv) REACTION: the OTHER hand holds the part that stays put (the body the hinge mounts on, the "
        "jar the cap screws into, the OPPOSITE side) - a DIFFERENT part from the acting one. A true anchor "
        "STAYS STATIONARY while the acting part moves: a contact that travels and rotates WITH the body is "
        "NOT an anchor. For a tilt / pour / empty, the anchor is the BASE / bottom (the pivot that stays "
        "put) while the other hand tilts the upper body. EXCEPTION - a MATCHING PAIR driven together: "
        "there is no stationary anchor; each hand drives its own part of the pair and each is the other's "
        "reaction (set purchase_contact = one part of the pair, reaction_part = its twin).\n"
        "STEP 3 - COORDINATION follows directly from STEP 1: ONE part moves on the otherwise-fixed body -> "
        "'stabilize+actuate' (one hand holds the frame, the other works the part); BOTH hands drive moving "
        "parts at once - a matching PAIR (two flaps, two handles, opposite twists) or two halves pulled "
        "APART - -> 'co-actuate'; NOTHING moves on the object - a "
        "plain lift / carry / flip / push of the whole rigid body - -> 'whole-body'. A pick-up / carry is "
        "always 'whole-body' (the object is one rigid unit; nothing on it moves). A whole-body LIFT is a "
        "CO-LIFT by DEFAULT: two hands grip two OPPOSITE faces (or two handles) and raise together - "
        "because a hand on a smooth face holds weight ONLY as half of a squeeze, so neither hand is ever "
        "passive. Set purchase_contact = one face / handle, reaction_part = the OPPOSITE face / second "
        "handle (BOTH are lifted). The ONE exception: a single DOMINANT grip a hand can hang the whole "
        "LIGHT object from (a mug's handle, a jug's loop) -> purchase_contact = that grip, reaction_part = "
        "the body (other hand only steadies). When unsure, default to the co-lift.\n\n"
        "Output JSON only (replace every <...>):\n"
        "{{\n"
        "  \"size_class\": \"<hand-sized | two-handed | furniture-scale>\",\n"
        "  \"moving_part\": \"<the visible part that moves, or 'none'>\",\n"
        "  \"motion\": \"<hinge-swing | slide | twist | tilt | none>\",\n"
        "  \"direction_to_goal\": \"<which way it must travel, few words>\",\n"
        "  \"acting_verb\": \"<an allowed verb that produces that motion, or 'none' for whole-body>\",\n"
        "  \"purchase_contact\": \"<where to grip: the moving part for an actuation; for a whole-body lift the handle/grip if present, else a body side>\",\n"
        "  \"reaction_part\": \"<the OTHER hand's part: it holds the body still for an actuation, OR is the second lift-point (opposite face / second handle) co-lifted for a shared lift>\",\n"
        "  \"coordination\": \"<stabilize+actuate | whole-body | co-actuate>\"\n"
        "}}"
    ),
}

GROUND_PROMPT = {
    "system": (
        "You convert an OPERATION PLAN into two coordinated, hand-agnostic robot roles. The physics is "
        "already worked out in the plan - you TRANSCRIBE it into roles; do NOT re-reason the physics."
    ),
    "user": (
        "Object: {object_name}\nTask: {task}\nGoal: {goal}\n"
        "Visible parts (a role's target MUST be one of these, verbatim):\n{components}\n"
        "Allowed verbs: {role_verbs}\n\n"
        "OPERATION PLAN (already derived - OBEY it):\n{plan}\n\n"
        "Make EXACTLY two roles, A and B, from the plan:\n"
        "- coordination 'stabilize+actuate': ACTING hand = plan.acting_verb on plan.moving_part at "
        "plan.purchase_contact; OTHER hand = 'hold' plan.reaction_part.\n"
        "- coordination 'whole-body' (lift / carry, nothing moves on the object): a lift needs UPWARD "
        "force, and a hand on a smooth vertical face gives only a SIDEWAYS push - it bears weight ONLY as "
        "half of a two-hand squeeze, so it can NEVER be a passive 'hold'. Therefore the DEFAULT is a "
        "CO-LIFT: BOTH hands 'lift', each gripping an OPPOSITE face (or one of two handles) and raising "
        "together - this is how two hands raise anything that cannot dangle from a single grip. The ONLY "
        "exception: a single DOMINANT grip a hand can hang the whole LIGHT object from - a mug's handle, a "
        "jug's loop - then THAT hand 'lift's the grip and the OTHER merely 'hold's the body "
        "to steady it. When unsure, default to BOTH 'lift' two OPPOSITE faces. furniture-scale that cannot be raised "
        "-> both 'push' the body (verb 'push').\n"
        "- coordination 'co-actuate': BOTH hands actively drive moving parts at the same time - each "
        "works its own part of a matching pair (plan.purchase_contact and plan.reaction_part, e.g. two "
        "flaps, two handles), or the two halves are pulled APART in opposite directions. Neither hand is "
        "a passive 'hold'.\n"
        "Each role: role (allowed verb), target (a visible part, verbatim), contact_region (a specific "
        "spot ON that hand's OWN target; the two regions must be DIFFERENT places - opposite faces for a "
        "symmetric grip), function. Never actuate a part fused/rigid to the body. Every contact_region "
        "must be REACHABLE in the object's natural resting pose: the underside / base face it RESTS ON is "
        "occluded - never place a contact there; a steadying hand resists from the nearest EXPOSED surface "
        "(a side wall, rim, or upper edge). Pin the region in the OBJECT'S OWN frame so the SAME physical "
        "spot is found from ANY view, using any of: its UP-AXIS (top / bottom, rim / base, upper / lower / "
        "mid-height, a bottle's neck / shoulder) - gravity fixes these, the object always rests on its "
        "base; a NAMED part or anatomy (handle, spout, foot); a SURFACE mark (printed text, logo, label, "
        "colour); or a STRUCTURAL feature (seam, hinge, corner, notch, the open mouth) - composed by "
        "relation ('the rim just above the foot', 'the wall below the spout'). NEVER use an AZIMUTHAL "
        "viewer direction (left / right / front / back / 'facing camera' / 'azimuth'): these rotate with "
        "the camera and cannot be re-found on another view, whereas the up-axis does not rotate. If the "
        "object is symmetric with no distinguishing landmark, name a PAIR of opposite faces (interchangeable, "
        "resolved downstream), not one. If a region would equally fit three spots it is too generic.\n"
        "The two roles must do DISTINCT work: both hands doing the SAME action on the SAME SPOT is not "
        "bimanual - one stabilizes while the other actuates. EXCEPTION: a symmetric CO-LIFT (both hands "
        "'lift' two OPPOSITE faces, or the two grips, of one rigid body) IS bimanual - distinct by opposite "
        "location, jointly bearing a load neither hand can raise alone. Write 'relation' as the ACTUAL combined effect "
        "of A and B (their net force/torque toward the goal), read off the two roles - not a generic phrase.\n\n"
        "Output JSON only (replace every <...>):\n"
        "{{\n"
        "  \"roles\": [\n"
        "    {{\"id\": \"A\", \"role\": \"<verb>\", \"target\": \"<part>\", \"contact_region\": \"<spot on A's target>\", \"function\": \"<what A does>\"}},\n"
        "    {{\"id\": \"B\", \"role\": \"<verb>\", \"target\": \"<part>\", \"contact_region\": \"<a different spot, on B's target>\", \"function\": \"<what B does>\"}}\n"
        "  ],\n"
        "  \"relation\": \"<force / motion + spatial dependency between the two hands>\"\n"
        "}}"
    ),
}


def build_query_record(task, plan, plan_txt, dec):
    """Assemble the final query record from a task + its OP_PLAN dict + GROUND dict (roles)."""
    roles = dec.get("roles", [])
    p = plan if isinstance(plan, dict) else {}        # tolerate a parse-failed plan (dict.get -> None)
    return {
        "task": task.get("task"), "query": task.get("query"), "goal": task.get("goal"),
        "why_bimanual": task.get("why_bimanual"), "category": task.get("category"),
        "mechanism": p.get("moving_part"),
        "operation": plan_txt,
        "coordination": p.get("coordination"),
        "roles": roles, **normalize_pair(dec),
        "molmo_queries": [molmo_instruction(r) for r in roles],
        "answer": [r.get("contact_region") or r.get("target") for r in roles],
    }


# ---- BATCHED decompose: each phase runs ONE llm.generate over MANY jobs at once (vLLM continuous
# batching). Verified no cross-request image contamination (scratchpad/vllm_batch_verify.py: 0/17).
def decompose_batch(model, jobs):
    """jobs: list of {name, comps, task, views, labels}. Returns query records aligned to jobs (None=drop).
    Phase 1 = batched OP_PLAN (thinking off); Phase 2 = batched GROUND; Phase 3 = re-batch the GROUNDs that
    echoed placeholders (one retry round). Each phase is ONE model.text_images_batch(...) = one llm.generate
    over the whole list (vLLM continuous-batches internally; no manual chunking -- the official pattern)."""
    if not jobs:
        return []
    verbs = ", ".join(ROLE_VERBS)
    op_items = [{
        "user_text": OP_PLAN_PROMPT["user"].format(
            object_name=j["name"], task=j["task"].get("task", ""), goal=j["task"].get("goal", ""),
            why_bimanual=j["task"].get("why_bimanual", ""), components=components_to_text(j["comps"]),
            role_verbs=verbs, role_gloss=ROLE_GLOSS),
        "image_urls": j["views"], "system_text": OP_PLAN_PROMPT["system"], "labels": j["labels"],
    } for j in jobs]
    plans = [parse_json(o) for o in model.text_images_batch(op_items)]
    plan_txts = [json.dumps(p, ensure_ascii=False) if isinstance(p, dict) else str(p) for p in plans]

    def ground_items(idxs, retry):
        items = []
        for k in idxs:
            j = jobs[k]
            u = GROUND_PROMPT["user"].format(
                object_name=j["name"], task=j["task"].get("task", ""), goal=j["task"].get("goal", ""),
                components=components_to_text(j["comps"]), role_verbs=verbs, plan=plan_txts[k])
            if retry:
                u += ("\n\nOutput REAL verbs, REAL part names from the list, and SPECIFIC contact regions "
                      "- no placeholder text.")
            items.append({"user_text": u, "image_urls": j["views"],
                          "system_text": GROUND_PROMPT["system"], "labels": j["labels"]})
        return items

    decs = [parse_json(o) for o in model.text_images_batch(ground_items(list(range(len(jobs))), False))]
    retry_idx = [k for k, d in enumerate(decs) if not valid_decomp(d.get("roles", []))]
    if retry_idx:
        rout = [parse_json(o) for o in model.text_images_batch(ground_items(retry_idx, True))]
        for kk, k in enumerate(retry_idx):
            decs[k] = rout[kk]

    out = []
    for k, j in enumerate(jobs):
        dec = decs[k]
        if not valid_decomp(dec.get("roles", [])):
            out.append(None)
            continue
        out.append(build_query_record(j["task"], plans[k], plan_txts[k], dec))
    return out


POSE_CAP = 3      # functional_pose is a supplement for low-capacity objects, never the main course


def propose_tasks_batch(model, jobs, n_component=6, max_rounds=3):
    """BATCHED multi-pass propose + ONE strict curate. Round 1: broad batched BRAINSTORM (also reports
    is_scene / size_class). Rounds 2..max_rounds: batched RETRY doing a component sweep + a
    functional-pose sweep (avoid-list = everything already proposed + the fixed universal tasks); an
    object whose retry adds nothing is dry and stops early. Finally ONE batched CURATE composes each
    object's set - the LLM does all validation / semantic dedup; code only exact-string-dedups and
    enforces the INTER_CAP / POSE_CAP family quotas. Three families: object_level(inter) /
    component_level(intra) / functional_pose(pose) - pose tasks change only the object-side use pose.
    jobs: list of {name, comps, views, labels}. Returns task-lists aligned to jobs."""
    fam2cat = {"object_level": "inter", "component_level": "intra", "functional_pose": "pose"}
    fams = tuple(fam2cat.items())
    # stage0 has no separate contents annotation -> the model infers contents from what it sees
    contents_note = "(not separately annotated; infer only from the images and listed components)"
    st = [{"comp_txt": components_to_text(j["comps"]), "inter": [], "intra": [], "pose": [],
           "seen": set(), "is_scene": False, "size_class": "two-handed", "active": True} for j in jobs]

    def pool(s, bs):
        """Accumulate one round's candidates (exact-string dedup only). Returns #added."""
        added = 0
        for fam, cat in fams:
            for t in bs.get(fam) or []:
                if not isinstance(t, dict):                # schema-deviant item must not kill the window
                    continue
                k = str(t.get("task", "")).strip().lower()
                if not k or k in s["seen"]:
                    continue
                s["seen"].add(k)
                s[cat].append(dict(t, category=cat))
                added += 1
        return added

    for rnd in range(max_rounds):
        active = [i for i, s in enumerate(st) if s["active"]]
        if not active:
            break
        prompt = TASK_BRAINSTORM_PROMPT if rnd == 0 else TASK_BRAINSTORM_RETRY_PROMPT
        bs_items = []
        for i in active:
            j, s = jobs[i], st[i]
            # avoid the universal transport tasks from the start (both exist for every size_class)
            uni = [f"pick up the {j['name']}", f"move the {j['name']}"]
            avoid = "; ".join(sorted(s["seen"]) + uni)[:1600]
            bs_items.append({"user_text": prompt["user"].format(
                                 object_name=j["name"], components=s["comp_txt"],
                                 current_contents=contents_note, avoid=avoid,
                                 n_object=4 if rnd == 0 else 3,
                                 n_component=6 if rnd == 0 else 5,
                                 n_pose=3 if rnd == 0 else 4),   # retry offers extra pose candidates;
                                                                 # the curator still keeps <= POSE_CAP
                             "image_urls": j["views"], "system_text": prompt["system"],
                             "labels": j["labels"]})
        bs_out = [parse_json(o) for o in model.text_images_batch(bs_items)]
        for k, i in enumerate(active):
            s, bs = st[i], bs_out[k]
            if "error" in bs:                              # transient parse failure: keep active, the
                continue                                   # next round retries (round 1 fields defaulted)
            if rnd == 0:
                s["is_scene"] = bool(bs.get("is_scene"))
                s["size_class"] = bs.get("size_class", s["size_class"])
            if pool(s, bs) == 0 and rnd > 0:               # a dry RETRY round -> another one won't help
                s["active"] = False

    # ---- ONE batched CURATE: LLM validates + semantically dedups + composes the final set ----
    bases, cur_items = [], []
    for i, j in enumerate(jobs):
        s = st[i]
        base = [] if s["is_scene"] else universal_tasks(j["name"], s["size_class"])
        bases.append(base)
        cur_items.append({"user_text": TASK_CURATE_PROMPT["user"].format(
                              object_name=j["name"], components=s["comp_txt"],
                              current_contents=contents_note,
                              universal=candidates_to_text(base),
                              object_level=candidates_to_text(s["inter"]),
                              component_level=candidates_to_text(s["intra"]),
                              functional_pose=candidates_to_text(s["pose"]),
                              n_component=n_component,
                              n_object=max(0, INTER_CAP - len(base)),
                              n_pose=POSE_CAP),
                          "image_urls": j["views"], "system_text": TASK_CURATE_PROMPT["system"],
                          "labels": j["labels"]})
    cur_out = [parse_json(o) for o in model.text_images_batch(cur_items)]
    out = []
    for base, cur in zip(bases, cur_out):
        existing = {str(t.get("task", "")).strip().lower() for t in base}
        n_cat = {"inter": len(base), "pose": 0}
        cap = {"inter": INTER_CAP, "pose": POSE_CAP}
        tasks = list(base)
        for t in cur.get("tasks") or []:
            if not isinstance(t, dict):
                continue
            k = str(t.get("task", "")).strip().lower()
            if not k or k in existing:                     # exact dup of a fixed task only; semantic
                continue                                   # dedup is the curator's job
            cat = fam2cat.get(t.get("category"), "inter")
            if cat in cap:                                 # hard family caps even if the curator
                if n_cat[cat] >= cap[cat]:                 # disobeys its quotas (intra-collapse /
                    continue                               # pose-flood guards)
                n_cat[cat] += 1
            existing.add(k)
            tasks.append(dict(t, category=cat))
        for t in tasks:
            t["query"] = to_query(t.get("task", ""))
        out.append(tasks)
    return out


def run_batched(model, batch, args, out, done, results):
    """Batched path: process objects in windows of args.batch_size. Per window, propose + decompose each run
    as ONE llm.generate over the whole window (vLLM continuous-batches internally -- no manual chunking).
    Writes the file + frees temp view dirs per window. A window that raises is logged and skipped; its
    objects are simply re-run on the next resume (they were never written, so not in `done`)."""
    window = []   # [{obj, gi, name, comps, clean, labels, views, odir}]

    def write():
        with open(out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def flush():
        if not window:
            return
        t0 = time.time()
        try:
            jobs = [{"name": w["name"], "comps": w["comps"], "views": w["clean"],
                     "labels": w["labels"]} for w in window]
            tasks_per = propose_tasks_batch(model, jobs, n_component=args.n_component,
                                            max_rounds=args.max_rounds)
            dec_jobs = [{"wi": wi, "name": w["name"], "comps": w["comps"], "task": t,
                         "views": w["clean"], "labels": w["labels"]}
                        for wi, w in enumerate(window) for t in tasks_per[wi]]
            recs = decompose_batch(model, dec_jobs)
            per = {wi: [] for wi in range(len(window))}
            for job, rec in zip(dec_jobs, recs):
                if rec:
                    per[job["wi"]].append(rec)
            for wi, w in enumerate(window):
                results.append({"object_id": w["obj"]["object_id"], "object_name": w["name"],
                                "views_used": w["views"], "components": w["comps"],
                                "queries": per[wi]})
                print(f"  [{w['gi']}] {w['name'][:28]:28} {len(per[wi])} queries")
            write()
            print(f"  [window {len(window)} objs] {time.time() - t0:.0f}s")
        except Exception as e:                                 # a bad window shouldn't kill the run
            print(f"  window of {len(window)} failed ({e}); recording error stubs")
            for w in window:                                   # error record (like serial) -> run completes, no re-loop
                results.append({"object_id": w["obj"]["object_id"], "object_name": w["name"], "error": str(e)})
            write()
        for w in window:
            shutil.rmtree(w["odir"], ignore_errors=True)       # free temp views (bounded /tmp)
        window.clear()

    for i, obj in enumerate(batch):
        gi = args.start + i
        if obj["object_id"] in done:
            results.append(done[obj["object_id"]])
            continue
        name, views = obj["object_name"], obj.get("views_used", [])
        if not views or not all(os.path.exists(v) for v in views):
            print(f"  [{gi}] {name[:28]:28} SKIP (views missing)")
            continue
        odir = os.path.join(VIEW_DIR, str(obj["object_id"]))
        pairs = safe_prepare_views(views, odir)
        if len(pairs) < 5:
            print(f"  [{gi}] {name[:28]:28} SKIP (corrupt renders)")
            shutil.rmtree(odir, ignore_errors=True)            # dir was created above -> don't leak it
            continue
        window.append({"obj": obj, "gi": gi, "name": name, "odir": odir, "views": views,
                       "comps": ensure_body_component(name, obj["components"]),
                       "clean": [p for _, p in pairs],
                       "labels": [view_label(views[idx]) for idx, _ in pairs]})
        if len(window) >= args.batch_size:
            flush()
    flush()
    write()                                                    # final write (trailing done objects)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="outputs/stage0_filtered_redesign.kept.json")
    ap.add_argument("--out", default="outputs/stage1_redesign_26b.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--n_component", type=int, default=6)   # component (intra) tasks the curator may keep
                                                            # per object; + 1-2 universal + <=INTER_CAP
                                                            # extra inter -> ~5-8 tasks on rich objects,
                                                            # honest fewer on simple ones
    ap.add_argument("--max_rounds", type=int, default=3)    # brainstorm rounds: 1 broad + (max_rounds-1)
                                                            # coverage retries; a dry retry stops early
    ap.add_argument("--batch_size", type=int, default=1,    # >1: batched path; = objects per window
                    help="objects per window. A window's images must all stay resident in vLLM's encoder "
                         "cache (max_num_batched_tokens/560 imgs) so each object's 8 views encode ONCE and "
                         "reuse across its ~10 calls; bigger windows that overflow the cache re-encode and "
                         "drop back to serial speed. OPTIMUM = 8 WITH --max_num_batched_tokens 40960 (64<73 "
                         "imgs fit -> 5.5s/obj); W=16 over-squeezes KV (~5.9). Default cache: best is W=3 "
                         "(6.6). Measured W=1 10.6, W=3 6.6, W=8+cache 5.5, W=16+cache 5.9, W=32 10.7.")
    ap.add_argument("--gpu_mem", type=float, default=0.85,  # vLLM KV cache headroom; raise for a bigger batch
                    help="vLLM gpu_memory_utilization (bigger -> more KV cache -> larger concurrent batch)")
    ap.add_argument("--max_num_seqs", type=int, default=None,          # official concurrency knob (vLLM default 128)
                    help="vLLM max concurrent requests; raise for throughput, lower if OOM/preemption")
    ap.add_argument("--max_num_batched_tokens", type=int, default=None,  # per-step budget = multi-image encoder budget
                    help="vLLM per-step token budget (= encoder-cache budget); raise (>8192) to relax multi-image prefill")
    ap.add_argument("--max_new_tokens", type=int, default=2048,   # generous cap; greedy stops at EOS so it's ~free
                    help="output token cap. Observed max real output ~258; 2048 = ample no-truncation headroom "
                         "without a large runaway tail (avoid setting to max_model_len)")
    ap.add_argument("--model_id", default=os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-26B-A4B-it"))
    args = ap.parse_args()

    inp = args.inp if os.path.isabs(args.inp) else os.path.join(DATA_GEN, args.inp)
    out = args.out if os.path.isabs(args.out) else os.path.join(DATA_GEN, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["GEMMA_MODEL_ID"] = args.model_id

    os.chdir(REPO_ROOT)
    sys.path.insert(0, BIMANUAL_DIR)
    from gemma import Gemma

    objects = [o for o in json.load(open(inp)) if o.get("keep") and o.get("components")]
    end = args.end if args.end is not None else len(objects)
    batch = objects[args.start:end]
    print(f"stage1: {len(batch)} kept objects [{args.start}:{end}] of {len(objects)} -> {out}")

    model = Gemma(max_new_tokens=args.max_new_tokens, gpu_memory_utilization=args.gpu_mem,   # gpu_mem: vLLM KV headroom
                  max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens)
    results = []
    done = {}
    if os.path.exists(out):                           # resume: keep prior results, skip re-processing
        try:
            for r in json.load(open(out)):
                if r.get("object_id") and not r.get("error"):   # error stubs are NOT done -> re-run them
                    done[r["object_id"]] = r
            if done:
                print(f"  resume: {len(done)} objects already in {out}")
        except Exception:
            pass
    if getattr(model, "backend", None) != "vllm":     # text_images_batch only truly batches on vLLM
        print("  WARNING: backend != vLLM -> batching INACTIVE (serial speed); set GEMMA_BACKEND=vllm", flush=True)
    print(f"  propose+decompose batched, window={args.batch_size}")
    run_batched(model, batch, args, out, done, results)   # batch_size=1 => one object per window (serial-equivalent)
    total_q = sum(len(r.get("queries", [])) for r in results)
    print(f"\nstage1 done: {len(results)} objects, {total_q} queries -> {out}")


if __name__ == "__main__":
    main()
