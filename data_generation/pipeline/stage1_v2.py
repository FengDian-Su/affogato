#!/usr/bin/env python
"""
Stage 1 — task proposal + coordination-aware role decomposition + assemble.

Input: stage0_filter_components.py output (kept objects with components).
Per object:
  B1 brainstorm tasks in two families (inter_object whole-object / intra_object part-level)
  -> B2 rank "most daily-used first" + deterministic quota/backfill assembly (-> TOTAL tasks)
  -> per task: role decomposition into 2 hand-agnostic robot-executable roles (+contact_region)
               + pattern + role-conditioned Molmo instructions
  -> emit a query-ready record per task.

Output: one entry per object: {object_id, object_name, views_used, components, queries:[...]}.
Each query = {task, query, goal, roles, relation, pattern, molmo_queries, answer, ...}.

Reuses bimanual_annotation/{gemma.py, get_component.py}. Run with the **gemma4** env.

  CUDA_VISIBLE_DEVICES=3 \
  /home/michaellee/miniconda3/envs/gemma4/bin/python \
      data_generation/stage1_task_role_assemble.py \
      --in outputs/stage0_filtered.json --out outputs/stage1_dataset.json
"""
import os
import sys
import json
import math
import argparse
import subprocess
import glob
import re
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


# affogato (the single-region dataset we extend) gives human-VERIFIED affordances per object. We inject
# ALL of them as AFFORDANCE EVIDENCE and let the model REASON (principle-guided, NOT keyword-filtered)
# about which are genuine manipulations to rewrite vs perception / body-use / external to ignore. No code
# filter: a substring filter both false-drops real ops ("open to see what's inside" -> dropped on 'see')
# and is exactly the rigid-rule style we avoid -- the guideline teaches the distinction, and the
# downstream RANK reality-test gate enforces it.
_AFF_ROOT = os.path.join(REPO_ROOT, "dataset", "affogato")


def affogato_hint(object_id):
    """All affogato queries for this object as a principle-guided AFFORDANCE-EVIDENCE block, or '' if none."""
    if not object_id:
        return ""
    qs = None
    for qf in glob.glob(os.path.join(_AFF_ROOT, "affogato_all_part*", str(object_id), "queries.json")):
        try:
            d = json.load(open(qf)); rec = d[0] if isinstance(d, list) else d
            qs = rec.get("queries") if isinstance(rec, dict) else None
        except Exception:
            qs = None
        if qs:
            break
    if not qs:
        return ""
    lines = "\n".join(f"  - {q}" for q in qs)
    return (
        "AFFORDANCE EVIDENCE - the affogato dataset (the human-annotated affordance set we extend) "
        "recorded, for THIS exact object, the parts a person points to for everyday actions (verbatim "
        "below). They are human-verified, so TRUST them as evidence of which parts really exist and what "
        "the object truly affords - over any guess from the object's name. REASON about each: if it "
        "describes a genuine MANIPULATION of the object (open / close / pour / press / turn / pull / "
        "carry / squeeze ...), REWRITE it into a bimanual task below. If it is mere PERCEPTION (read / "
        "look at / identify / see), BODY-USE (sit / lean / rest / wear), or needs a SEPARATE item or "
        "surface (place it on a shelf, put an object on it, stack on top), it is NOT a manipulation of "
        "this object alone - use it only to understand the object, do NOT make it a task. Do not copy "
        "any verbatim.\n"
        f"{lines}\n\n"
    )


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
        # its real two-handed operations (open / pour / twist / press) are part-specific -> from brainstorm.
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


_GRASP_WORDS = ("pick up", "pick-up", "carry", "carrying", "lift", "lifting", "move", "moving",
                "transport", "take", "raise", "haul")


def is_grasp_task(task):
    """True for whole-object grasp/transport tasks already covered by the fixed pick-up."""
    t = str(task).lower()
    return any(w in t for w in _GRASP_WORDS)


# Deterministic near-duplicate key: strip the trailing bimanual qualifier clause ('... while holding
# the base', '... using both handles'), intensity adverbs ('vigorously'), and articles, so phrasings of
# the SAME action on the SAME part collapse to one key (a safety net under the RANK semantic dedup).
_DK_CLAUSE = re.compile(r"\s+(while|using|with)\s+.*$")
_DK_ADV = re.compile(r"\b(vigorously|gently|carefully|firmly|slowly|quickly|fully|completely|"
                     r"partially|repeatedly|lightly|slightly|hard)\b")
_DK_FILLER = re.compile(r"\b(a|an|the|its)\b")


def dedup_key(task):
    s = str(task).strip().lower().rstrip(".")
    s = _DK_CLAUSE.sub("", s)
    s = _DK_ADV.sub("", s)
    s = _DK_FILLER.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


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
TASK_BRAINSTORM_PROMPT = {
    "system": (
        "You are a task planner for a two-arm (bimanual) robot. You brainstorm common daily tasks "
        "the robot could do using ONLY its two hands and the object itself - never relying on walls, "
        "surfaces, other objects, or tools."
    ),
    "user": (
        "Object: {object_name}\n"
        "The images are several views of this object from different angles (incl. top-down and underside).\n"
        "Groundable parts (the robot can only contact these named parts):\n"
        "{components}\n\n"
        "{affogato}"
        "Brainstorm daily tasks for THIS object, in TWO families:\n"
        "A) inter_object - WHOLE-OBJECT tasks (object as one rigid unit, both hands together). "
        "Picking it up / carrying it with two hands is ALWAYS added automatically, so do NOT list "
        "plain pick-up / carry / lift / move. Instead propose OTHER whole-object actions ONLY IF they "
        "are genuinely meaningful and common for THIS specific object - e.g. pour from a kettle / "
        "pitcher / bottle, tip out a bin, flip a clamshell case. Do NOT force rotate / flip / reorient "
        "on objects where they serve no real purpose; if no other whole-object action is meaningful, "
        "return an EMPTY inter_object list.\n"
        "B) intra_object - PART-LEVEL tasks: one hand HOLDS the body while the OTHER operates a movable "
        "or functional PART (open/close a lid/flap/door, twist a cap/knob, pull a handle/drawer, press "
        "a button/latch, slide a cover, insert/detach a part). Each MUST name a real part above.\n\n"
        "Favour the operations people CHARACTERISTICALLY do with THIS KIND of object (e.g. a garment is "
        "folded/rolled, a bottle/jar is opened and poured, a book is paged) that its visible parts afford.\n"
        "Give up to {n} tasks in EACH family. Every task must be a COMMON, everyday operation people "
        "genuinely do with this object - keep it SIMPLE, DIRECT, natural; do NOT invent unusual / "
        "contrived tasks just to reach {n}; if only a few exist, give only those. DIVERSITY = genuinely "
        "different ACTIONS / parts, NOT a 'left' vs 'right' mirror variant. Reworded phrasings are fine.\n"
        "Do NOT repeat any of these ALREADY-PROPOSED tasks - propose DIFFERENT ones (or genuinely new "
        "rewordings of yet-uncovered actions): {avoid}\n"
        "Each task MUST:\n"
        "- cause an object state / pose / part change with a CONCRETE end-state (reject vague "
        "'adjust' / 'reposition' / 'arrange' goals)\n"
        "- need TWO hands AT THE SAME TIME (no hand-to-hand handoffs / passing / regrasps)\n"
        "- be doable with BARE HANDS only (no tool needed)\n"
        "- be SELF-CONTAINED: two hands + this object only, no external surface / object / tool\n"
        "- only involve the groundable parts above (intra tasks must name a real part)\n\n"
        "If {object_name} is actually a multi-object SCENE / arrangement (e.g. a table setting, a still "
        "life) rather than ONE rigid object, set is_scene true and return an EMPTY inter_object list "
        "(only part-level tasks on individual items make sense).\n"
        "Also report the object's real-world SIZE CLASS, judged from WHAT THE OBJECT IS (the views are "
        "SIZE-NORMALIZED, do NOT read size off the image): \"hand-sized\" (one hand lifts it), "
        "\"two-handed\" (a person lifts it with two hands), or \"furniture-scale\" (too big/heavy to "
        "lift - it is rolled / pushed, e.g. a chest freezer, large cooler, appliance).\n"
        "Return JSON only:\n"
        "{{\n  \"is_scene\": false,\n  \"size_class\": \"hand-sized | two-handed | furniture-scale\",\n"
        "  \"inter_object\": [ {{\"task\": \"...\", \"goal\": \"<before> -> <after>\"}} ],\n"
        "  \"intra_object\": [ {{\"task\": \"...\", \"goal\": \"<before> -> <after>\"}} ]\n}}\n"
        "Output JSON only."
    ),
}

TASK_RANK_PROMPT = {
    "system": (
        "You curate a two-arm (bimanual) robot manipulation dataset. You validate candidate tasks "
        "and rank them by how common/everyday they are (MOST daily-used first)."
    ),
    "user": (
        "Object: {object_name}\n"
        "Groundable parts:\n{components}\n\n"
        "inter_object candidates (other meaningful whole-object actions):\n{inter}\n\n"
        "intra_object candidates (part-level):\n{intra}\n\n"
        "For EACH family: drop any task that breaks a rule OR is not genuinely meaningful for THIS "
        "object, then RANK survivors MOST DAILY-USED / most common FIRST. A valid task MUST:\n"
        "- need TWO hands at the same time (no handoffs / passing / regrasps)\n"
        "- be SELF-CONTAINED (two hands + this object only; no walls / surfaces / other objects / tools)\n"
        "- REALITY TEST (DROP if it fails any): (1) does a real person do this BARE-HANDED and routinely to "
        "THIS object? (2) does the end-state need anything NOT in the scene (a vessel to pour into, a "
        "second object, a person)? (3) does it require defeating a seal/tape/shrink-wrap (= a tool)? (4) is "
        "it a genuine STATE CHANGE, not 'after' == 'before'? A pour / empty presupposes a real interior "
        "cavity + opening - reject it on a solid or sealed body.\n"
        "- cause an object state / pose / part change, and not be solvable by one hand alone\n"
        "- only involve the groundable parts above\n"
        "- be BARE-HANDS doable (DROP anything needing a tool OR a KEY: wrench/screwdriver/tape/knife, "
        "and padlock/keyed-lock unlock/lock) and have a CONCRETE end-state (DROP vague 'adjust'/"
        "'reposition'/'arrange' tasks)\n"
        "- be a SINGLE coordinated action (DROP compound / sequential tasks that need two ordered steps, "
        "e.g. 'close and latch the lid', 'fold then tuck' - keep only the one operation)\n"
        "For inter_object specifically, DROP any plain pick-up / carry / lift / move (added separately) "
        "and DROP rotate / flip / reorient UNLESS they are genuinely meaningful for this object.\n"
        "Keep tasks simple and natural. If two surviving tasks perform the SAME action on the SAME part - "
        "differing only in wording, an added qualifier, or intensity - keep just the single clearest one; "
        "keep both only when the ACTION or the TARGET PART genuinely differs.\n\n"
        "JUSTIFICATION (why_bimanual): give a CONCRETE physical reason specific to THIS object, in ONE "
        "of two forms: (a) ASYMMETRIC - one hand must HOLD/steady one named part while the OTHER hand "
        "operates a DIFFERENT named part (name both parts); or (b) SYMMETRIC - the object must be gripped "
        "at two DISTINCT, OPPOSITE outer regions at the same time to keep it level/balanced. The bare "
        "words 'large', 'heavy', or 'awkward' are BANNED as a justification - if the only reason you can "
        "give is that the object is big or heavy, and one hand could in fact do it, the task is NOT "
        "bimanual: DROP it.\n\n"
        "Return JSON only (each family ranked, most-daily-used first):\n"
        "{{\n  \"inter_object\": [ {{\"task\": \"...\", \"goal\": \"<before> -> <after>\", "
        "\"why_bimanual\": \"what fails with one hand\"}} ],\n"
        "  \"intra_object\": [ {{\"task\": \"...\", \"goal\": \"<before> -> <after>\", "
        "\"why_bimanual\": \"what fails with one hand\"}} ]\n}}\n"
        "Output JSON only."
    ),
}


# ---------------------------------------------------------------- per-object pipeline
def whole_object_fillers(object_name, size_class):
    """GROUNDED whole-object transforms used ONLY to reach the hard floor on part-poor objects, WITHOUT
    hallucinating a part. Every hand-liftable rigid body can be turned over / reoriented with two hands,
    each grounding to two OPPOSITE faces (a clean co-rotate two-region target) - honest-but-generic, never
    a fake lid/cap. Furniture-scale returns none (can't flip a freezer)."""
    o = str(object_name).strip()
    if "furniture" in str(size_class).lower():
        return []
    return [
        {"task": f"turn over the {o}", "category": "inter",
         "goal": f"{o} resting on its base -> {o} rotated end-over-end to rest on the opposite face",
         "why_bimanual": ("both hands grip two opposite faces and apply matched torque to flip it; one "
                          "hand cannot rotate it without it slipping or dropping")},
        {"task": f"reorient the {o}", "category": "inter",
         "goal": f"{o} at one facing -> {o} rotated in the hands to a new facing",
         "why_bimanual": ("both hands hold two opposite faces and turn it together to change its facing "
                          "while keeping it level")},
    ]


def accumulate_ranked(ranked, inter, intra, seen, seen_keys):
    """Add a RANK round's survivors into the inter/intra pools, deduped (exact string + normalized
    dedup_key + transport filter). Mutates the pools/sets in place; returns how many were added."""
    added = 0
    for fam, pool, cat in (("inter_object", inter, "inter"), ("intra_object", intra, "intra")):
        for t in ranked.get(fam, []):
            k = str(t.get("task", "")).strip().lower(); dk = dedup_key(k)
            if k and dk not in seen_keys and not is_grasp_task(k):   # transport is whole-body, never a pool task
                pool.append(dict(t, category=cat)); seen.add(k); seen_keys.add(dk); added += 1
    return added


def enough_tasks(inter, intra, is_scene, size_class, target_max):
    """Stop condition: scene, or the pools already fill every non-universal slot up to target_max."""
    n_uni = 2 if "two-handed" in str(size_class).lower() else 1   # universal tasks fill that many slots
    return is_scene or len(inter) + len(intra) >= max(0, target_max - n_uni)


def assemble_tasks(name, is_scene, size_class, inter, intra, target_min, target_max):
    """Universal transport leads -> ranked brainstormed fill (cap target_max) -> hard-floor backfill with
    GROUNDED whole-object transforms (never a fabricated part) -> attach the query wrapper. Shared logic."""
    base = [] if is_scene else universal_tasks(name, size_class)
    tasks = (base + inter + intra)[:target_max]
    if not is_scene:
        have = {dedup_key(t.get("task", "")) for t in tasks}
        for f in whole_object_fillers(name, size_class):
            if len(tasks) >= target_min:
                break
            if dedup_key(f["task"]) not in have:
                tasks.append(f); have.add(dedup_key(f["task"]))
        tasks = tasks[:target_max]
    for t in tasks:
        t["query"] = to_query(t.get("task", ""))
    return tasks


# ---- FACTORED decomposition: call 1 derives the operation-mechanics PLAN (focused physics, no role
# formatting), call 2 GROUNDS that plan into two roles. Isolating the physics keeps a 12B from letting
# the big formatting prompt dilute its reasoning (probe: it reasons latches correctly when focused).
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
        "have both hands rotate the whole body. Name the rigid BODY and which SINGLE part, if any, must "
        "MOVE (or 'nothing moves' for a plain carry/flip/move); (b) the object's real-world SIZE CLASS, "
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
        "put) while the other hand tilts the upper body.\n"
        "STEP 3 - COORDINATION follows directly from STEP 1: a PART moves on the otherwise-fixed body -> "
        "'stabilize+actuate' (one hand holds the frame, the other works the part); the body splits into two "
        "halves that pull APART in opposite directions -> 'co-actuate'; NOTHING moves on the object - a "
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
        "- coordination 'co-actuate': both hands actively pull/separate the two halves in OPPOSITE "
        "directions.\n"
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


# ---- BATCHED stage (option B): both propose (propose_tasks_batch) and decompose (decompose_batch) run
# ONE llm.generate over MANY jobs at once (vLLM continuous batching). Same OP_PLAN->GROUND logic as the
# serial path, just batched. Verified no cross-request image contamination (scratchpad/vllm_batch_verify.py:
# 0/17). Records are built by the shared build_query_record, identical to serial.
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


def propose_tasks_batch(model, jobs, target_min=3, target_max=5, max_rounds=2):
    """BATCHED multi-object propose. Each round runs ONE batched BRAINSTORM + ONE batched RANK over all
    ACTIVE objects (each = one model.text_images_batch = one llm.generate over the whole active list; vLLM
    continuous-batches internally), then per-object accumulate + stop-check with INDEPENDENT per-object state
    (no cross-object contamination). jobs: list of {name, comps, views, labels, aff_hint}. Returns task-lists
    aligned to jobs."""
    st = [{"comp_txt": components_to_text(j["comps"]), "inter": [], "intra": [], "seen": set(),
           "seen_keys": set(), "is_scene": False, "size_class": "two-handed", "active": True, "_bs": {}}
          for j in jobs]
    for _ in range(max_rounds):
        active = [i for i, s in enumerate(st) if s["active"]]
        if not active:
            break
        # ---- batched BRAINSTORM over active objects (each with its OWN avoid-list) ----
        bs_items = []
        for i in active:
            j, s = jobs[i], st[i]
            avoid = "; ".join(sorted(s["seen"]))[:1600] if s["seen"] else "(none yet)"
            bs_items.append({"user_text": TASK_BRAINSTORM_PROMPT["user"].format(
                                 object_name=j["name"], components=s["comp_txt"], n=5,
                                 avoid=avoid, affogato=j.get("aff_hint", "")),
                             "image_urls": j["views"], "system_text": TASK_BRAINSTORM_PROMPT["system"],
                             "labels": j["labels"]})
        bs_out = [parse_json(o) for o in model.text_images_batch(bs_items)]
        for k, i in enumerate(active):
            bs = bs_out[k]
            st[i]["is_scene"] = st[i]["is_scene"] or bool(bs.get("is_scene"))
            st[i]["size_class"] = bs.get("size_class", st[i]["size_class"])
            st[i]["_bs"] = bs
        # ---- batched RANK over active objects ----
        rank_items = []
        for i in active:
            j, s = jobs[i], st[i]; bs = s["_bs"]
            rank_items.append({"user_text": TASK_RANK_PROMPT["user"].format(
                                   object_name=j["name"], components=s["comp_txt"],
                                   inter=candidates_to_text(bs.get("inter_object", [])),
                                   intra=candidates_to_text(bs.get("intra_object", []))),
                               "image_urls": j["views"], "system_text": TASK_RANK_PROMPT["system"],
                               "labels": j["labels"]})
        rank_out = [parse_json(o) for o in model.text_images_batch(rank_items)]
        # ---- per-object accumulate + stop ----
        for k, i in enumerate(active):
            s = st[i]
            added = accumulate_ranked(rank_out[k], s["inter"], s["intra"], s["seen"], s["seen_keys"])
            if added == 0 or enough_tasks(s["inter"], s["intra"], s["is_scene"], s["size_class"], target_max):
                s["active"] = False
    return [assemble_tasks(jobs[i]["name"], s["is_scene"], s["size_class"], s["inter"], s["intra"],
                           target_min, target_max) for i, s in enumerate(st)]


def run_batched(model, batch, args, out, done, results):
    """Batched path: process objects in windows of args.batch_size. Per window, propose + decompose each run
    as ONE llm.generate over the whole window (vLLM continuous-batches internally -- no manual chunking).
    Writes the file + frees temp view dirs per window. A window that raises is logged and skipped; its
    objects are simply re-run on the next resume (they were never written, so not in `done`)."""
    window = []   # [{obj, gi, name, comps, clean, labels, views, aff, odir}]

    def write():
        with open(out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def flush():
        if not window:
            return
        t0 = time.time()
        try:
            jobs = [{"name": w["name"], "comps": w["comps"], "views": w["clean"],
                     "labels": w["labels"], "aff_hint": w["aff"]} for w in window]
            tasks_per = propose_tasks_batch(model, jobs, target_min=args.target_min,
                                            target_max=args.target_max, max_rounds=args.max_rounds)
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
                                "views_used": w["views"], "components": w["comps"], "queries": per[wi]})
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
                       "aff": affogato_hint(obj["object_id"]), "clean": [p for _, p in pairs],
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
    ap.add_argument("--target_min", type=int, default=3)    # hard floor (backfilled w/ grounded transforms)
    ap.add_argument("--target_max", type=int, default=5)    # cap on total tasks per object
    ap.add_argument("--max_rounds", type=int, default=2)    # agentic regenerate rounds before giving up
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
                if r.get("object_id"):
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
