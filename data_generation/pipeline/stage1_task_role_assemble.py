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

ROLE_VERBS = ["hold", "support", "push", "pull", "press", "lift", "rotate", "slide", "insert"]
ROLE_GLOSS = (
    "hold = static prehensile grip that immobilizes a part (stabilizer); "
    "support = bear weight from below WITHOUT a grip (non-prehensile stabilizer); "
    "push = non-prehensile lateral force to translate; "
    "pull = grasp then retract along a line; "
    "press = localized normal force into a part that yields/clicks (button/latch/lid); "
    "lift = raise the whole object upward; "
    "rotate = axis-parameterized rotation - covers in-place twist of a knob/cap AND reorient/flip/tilt "
    "of the whole object; "
    "slide = translate a part along its linear/prismatic track; "
    "insert = mate a part into a receptacle/slot"
)
RELATION_EXAMPLES = ["opposing torque", "stabilize and actuate", "balanced grip on opposite sides",
                     "opposing force", "orientation control", "guide and actuate",
                     "constrained translation"]


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
    t = str(task).strip().rstrip(".")
    if t:
        t = t[0].lower() + t[1:]
    return f"How would you {t}?"


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
        "Keep tasks simple and natural; keep rewordings / similar tasks (only drop EXACT duplicates).\n\n"
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

ROLE_DECOMP_PROMPT = {
    "system": (
        "You are a manipulation-mechanics expert. You convert one bimanual task into two coordinated, "
        "hand-agnostic robot roles by FIRST reasoning about how the object physically works, then "
        "assigning each hand the role it must play. Ground every role in a part visible in the views."
    ),
    "user": (
        "Object: {object_name}\n"
        "Task: {task}\n"
        "Goal: {goal}\n"
        "Why bimanual: {why_bimanual}\n"
        "Groundable parts (a role's target MUST be one of these, copied verbatim):\n"
        "{components}\n\n"
        "Allowed verbs: {role_verbs}\n"
        "Verb meanings - {role_gloss}.\n\n"
        "Reason in THREE steps, then emit JSON.\n\n"
        "STEP 1 - MECHANISM + SCALE. State in one line: (a) the rigid BODY (bears the load, does not "
        "deform) and which SINGLE part, if any, must MOVE to reach the goal (a lid swings, a cap "
        "unscrews, a page turns, a drawer slides), or 'nothing moves' for a plain carry; (b) the "
        "object's real-world SIZE CLASS - judge it from WHAT THE OBJECT IS (the views are SIZE-NORMALIZED, "
        "so do NOT read size off the image): hand-sized (one hand fully grasps/lifts it: mug, can, book), "
        "two-handed (one hand cannot lift it but a person can: large box, full pot, crate), or "
        "furniture-scale (not hand-liftable: cooler, freezer, appliance). Use only parts you can SEE.\n\n"
        "STEP 2 - OPERATION MECHANICS. If STEP 1 found a part that MOVES, work out the physics of moving "
        "it BEFORE you name any verb or region. Answer these four lines in order, each in a few words - "
        "the verb and the contact spot must be DERIVED here, not guessed later. (If STEP 1 said 'nothing "
        "moves', write 'whole-body, no part motion' and skip to STEP 3.)\n"
        "  (i) ALLOWED MOTION: how can the moving part move at all? It has ONE way: swing on a hinge "
        "(arc), slide along a track (straight line), or twist about an axis through its face (a cap / "
        "knob). Name which, and where the hinge / track / twist-axis is.\n"
        "  (ii) DIRECTION TO GOAL: along that one motion, which way must the part travel to go from its "
        "CURRENT state to the GOAL state? (a closed latch must move UP off its catch; an upright bowl's "
        "rim must rotate DOWN to pour; a shut drawer must come straight OUT). The acting hand's verb is "
        "THIS motion - pick the allowed verb that produces it.\n"
        "  (iii) PURCHASE (where to push/pull): place the acting contact where it has the most leverage "
        "on that motion - the spot FARTHEST from the hinge or axis, on a rigid surface. For a SWING: the "
        "FREE edge of the part, opposite the hinge (never near the hinge - no leverage). For a TWIST: the "
        "RIM / outer circumference of the cap or knob (never its flat top or centre - that sits ON the "
        "axis and turns nothing). For a SLIDE: a graspable handle / front face you can pull along the "
        "track. Reject any spot on the pivot, on the twist-axis, or on a floppy / non-rigid surface.\n"
        "  (iv) REACTION (the other hand): the acting push/pull will shove the whole object unless "
        "something holds it. The other hand HOLDS the part that must stay put - the body the hinge is "
        "mounted on, the jar the cap screws into, or the OPPOSITE side / cover - so the acting force does "
        "work. It must be a DIFFERENT part from the acting hand's.\n"
        "Worked examples (reasoning -> roles):\n"
        "  - Open a hinged lid: (i) swings on an arc about the rear hinge; (ii) the front of the lid must "
        "travel UP/open -> verb 'pull'; (iii) grip the lid's FRONT free edge, far from the hinge; (iv) "
        "other hand 'hold' the body below the hinge. -> A: pull the lid at its front edge; B: hold the body.\n"
        "  - Unscrew a cap: (i) twists about the axis through the cap's top; (ii) turn it to loosen -> "
        "verb 'rotate'; (iii) grip the cap's RIM / circumference, NOT its flat top (the top is on the "
        "axis and gives no turn); (iv) other hand 'hold' the bottle / jar body it screws into. -> A: "
        "rotate the cap at its rim; B: hold the body.\n"
        "  - Pour from a bowl: (i) the whole bowl tilts about its lip / rim; (ii) the rim must rotate DOWN "
        "so contents leave -> verb 'rotate' (tilt), NOT 'hold'; (iii) grip the bowl wall opposite the "
        "pouring lip for leverage; (iv) other hand 'support' the base from below. -> A: rotate (tilt) the "
        "bowl; B: support the base.\n\n"
        "STEP 3 - DIVISION OF LABOUR. First ask: does ONE hand stay still (a static anchor) while the "
        "other acts (ASYMMETRIC), or do BOTH hands act at the same time (SYMMETRIC)? Then pick ONE "
        "scheme:\n"
        "  (A) STABILIZE + ACTUATE  [asymmetric] - one hand HOLDS the part that must stay still (the "
        "reaction from STEP 2(iv)); the OTHER hand performs the motion on the part that MOVES, using the "
        "verb from STEP 2(ii) at the purchase contact from STEP 2(iii). Use when ONE part moves relative "
        "to a held body (open, close, twist a cap, pull out a drawer, press, tilt-to-pour, turn a page).\n"
        "  (B) WHOLE-BODY  [symmetric, body rigid, no internal motion] - both hands act on the ONE rigid "
        "body to lift / carry / flip / move it. Grip by the SIZE CLASS from STEP 1: hand-sized / "
        "base-cuppable -> one hand SUPPORTS the base / underside from below while the other STEADIES a "
        "handle / rim / upper edge; two-handed-scale whose base one hand CANNOT support (big box, crate) "
        "-> BOTH hands grip two OPPOSITE outer sides or diagonal corners to lift it level (do NOT balance "
        "it on one hand under the centre). To FLIP / tilt the whole body, apply matched torque at those "
        "two opposite grips. To MOVE a FURNITURE-SCALE object that cannot be lifted (cooler, freezer) -> "
        "both hands PUSH its body in the travel direction (or one pushes + one steers): use 'push', "
        "never 'lift'.\n"
        "  (C) CO-ACTUATE  [symmetric, both active] - BOTH hands apply active force AT THE SAME TIME to "
        "separate, split, or deform the object: each hand grips one of the two parts / sides and they "
        "move in OPPOSITE or mirrored directions, with NEITHER hand a static anchor. Use for pull two "
        "halves apart (twist open a shaker / canister), splay open a carton's gable spout, peel two "
        "sides apart, wring, tear, stretch, or snap.\n\n"
        "STEP 4 - GROUND EACH HAND, transcribing STEPS 2-3: the acting hand's role = the verb from "
        "STEP 2(ii) and its contact_region = the purchase spot from STEP 2(iii); the other hand HOLDS / "
        "SUPPORTS the reaction part from STEP 2(iv).\n"
        "  - role: an allowed verb that matches the motion derived in STEP 2; a stabilizing hand uses "
        "'hold' or 'support'. Do NOT assign an actuate verb (rotate / pull / slide / open / press) to a "
        "part that is FUSED / rigid to the body and cannot move relative to it - if STEP 1 found nothing "
        "moves, this is a whole-body operation (B), not a part actuation.\n"
        "  - target: the visible part it contacts (a name from the list, verbatim). A hand that lifts, "
        "carries, or stabilizes must grip LOAD-BEARING structure (body / base / handle) - never a part "
        "that swings or detaches (lid / flap / strap / wheel).\n"
        "  - contact_region: a specific, pointable spot lying ON that hand's own target (the grounder "
        "points exactly here); the two hands' regions must be DIFFERENT places. For a SYMMETRIC two-sided "
        "grip the two regions must be on OPPOSITE faces (~180 deg apart) or diagonal corners - never two "
        "spots on the SAME face.\n"
        "  - function: what that hand accomplishes, consistent with its verb.\n\n"
        "Output JSON only (replace every <...>; never output the angle brackets):\n"
        "{{\n"
        "  \"mechanism\": \"<rigid body; which part moves, or 'nothing moves'>\",\n"
        "  \"operation\": \"<from STEP 2: the part's motion + direction to goal + purchase spot, or 'whole-body'>\",\n"
        "  \"coordination\": \"<stabilize+actuate | co-support | co-actuate>\",\n"
        "  \"roles\": [\n"
        "    {{\"id\": \"A\", \"role\": \"<verb>\", \"target\": \"<part>\", \"contact_region\": \"<spot on A's target>\", \"function\": \"<what A does>\"}},\n"
        "    {{\"id\": \"B\", \"role\": \"<verb>\", \"target\": \"<part>\", \"contact_region\": \"<a different spot, on B's target>\", \"function\": \"<what B does>\"}}\n"
        "  ],\n"
        "  \"relation\": \"<force / motion + spatial dependency between the two hands>\"\n"
        "}}"
    ),
}


# ---------------------------------------------------------------- per-object pipeline
def propose_tasks(model, name, comps, views, labels=None, target_inter=3, target_intra=3, max_rounds=4):
    """Agentic generate->validate loop: each round brainstorm 5 inter + 5 intra (avoiding already-kept
    tasks for diversity), rank/validate, accumulate the qualified ones; regenerate until BOTH families
    reach their target (the fixed whole-object task counts as 1 inter) or max_rounds is hit."""
    comp_txt = components_to_text(comps)
    inter_pool, intra_pool, seen = [], [], set()
    is_scene, size_class = False, "two-handed"
    for _ in range(max_rounds):
        avoid = "; ".join(sorted(seen))[:1600] if seen else "(none yet)"
        bs = parse_json(model.text_images(
            TASK_BRAINSTORM_PROMPT["user"].format(object_name=name, components=comp_txt, n=5, avoid=avoid),
            views, TASK_BRAINSTORM_PROMPT["system"], labels=labels))
        is_scene = is_scene or bool(bs.get("is_scene"))
        size_class = bs.get("size_class", size_class)
        ranked = parse_json(model.text_images(
            TASK_RANK_PROMPT["user"].format(object_name=name, components=comp_txt,
                                            inter=candidates_to_text(bs.get("inter_object", [])),
                                            intra=candidates_to_text(bs.get("intra_object", []))),
            views, TASK_RANK_PROMPT["system"], labels=labels))
        for t in ranked.get("inter_object", []):
            k = str(t.get("task", "")).strip().lower()
            if k and k not in seen and not is_grasp_task(k):
                inter_pool.append(dict(t, category="inter")); seen.add(k)
        for t in ranked.get("intra_object", []):
            k = str(t.get("task", "")).strip().lower()
            if k and k not in seen:
                intra_pool.append(dict(t, category="intra")); seen.add(k)
        n_uni = 2 if "two-handed" in str(size_class).lower() else 1   # universal tasks fill that many inter slots
        enough_inter = is_scene or len(inter_pool) >= max(0, target_inter - n_uni)
        if enough_inter and len(intra_pool) >= target_intra:
            break

    # inter ALWAYS leads with the universal whole-object TRANSPORT tasks ({pick up, move}, or {move} for
    # furniture), unless a multi-object scene
    base_inter = [] if is_scene else universal_tasks(name, size_class)
    inter = (base_inter + inter_pool)[:target_inter]
    intra = intra_pool[:(target_inter + target_intra - len(inter))]   # fill the rest up to total
    tasks = inter + intra
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
        "always 'whole-body' (the object is one rigid unit; nothing on it moves). For a whole-body LIFT the "
        "grip is NOT just 'body': set purchase_contact to the part MADE for lifting - a handle / grip / "
        "bail / loop the object has - whenever one is present (it exists to bear the load); use a body side "
        "ONLY when there is none. reaction_part = the body, or the OPPOSITE side for a featureless object.\n\n"
        "Output JSON only (replace every <...>):\n"
        "{{\n"
        "  \"size_class\": \"<hand-sized | two-handed | furniture-scale>\",\n"
        "  \"moving_part\": \"<the visible part that moves, or 'none'>\",\n"
        "  \"motion\": \"<hinge-swing | slide | twist | tilt | none>\",\n"
        "  \"direction_to_goal\": \"<which way it must travel, few words>\",\n"
        "  \"acting_verb\": \"<an allowed verb that produces that motion, or 'none' for whole-body>\",\n"
        "  \"purchase_contact\": \"<where to grip: the moving part for an actuation; for a whole-body lift the handle/grip if present, else a body side>\",\n"
        "  \"reaction_part\": \"<the visible part the other hand holds still>\",\n"
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
        "plan.purchase_contact; OTHER hand = 'hold'/'support' plan.reaction_part.\n"
        "- coordination 'whole-body' (lift / carry, nothing moves on the object): lifting needs an UPWARD "
        "force that beats gravity. A hand flat on a vertical wall gives only a SIDEWAYS force and cannot "
        "lift by itself; it would lift only by friction, which needs an EQUAL squeeze from the opposite "
        "face. So reason from the CONTACT, not the size: (a) if the object HAS a part MADE for lifting - a "
        "handle / grip / bail / loop the fingers hook through, or a sturdy lip / rim / ledge they curl "
        "under - the lifting hand MUST take it (that part exists to bear the whole load): one hand 'lift's "
        "there and the other 'support'/'hold's the body to steady it (ASYMMETRIC); if there are TWO such "
        "grips, each hand takes one. (b) ONLY when NO such part exists (a smooth box, tin, slab, or a lug "
        "too small to bear the load) do BOTH hands 'lift'/'hold' two OPPOSITE faces AT ONCE, each supplying "
        "the inward squeeze the other needs (SYMMETRIC). furniture-scale that cannot be raised "
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
        "The two roles must do DISTINCT work: if both hands do the SAME action on the SAME part it is not "
        "bimanual - one stabilizes while the other actuates. Write 'relation' as the ACTUAL combined effect "
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


def decompose_task(model, name, comps, task, views, labels=None):
    comp_txt = components_to_text(comps)
    # call 1: focused operation-mechanics plan, WITH thinking on (the reasoning step that matters)
    plan = parse_json(model.text_images(
        OP_PLAN_PROMPT["user"].format(
            object_name=name, task=task.get("task", ""), goal=task.get("goal", ""),
            why_bimanual=task.get("why_bimanual", ""), components=comp_txt,
            role_verbs=", ".join(ROLE_VERBS), role_gloss=ROLE_GLOSS),
        views, OP_PLAN_PROMPT["system"], labels=labels, enable_thinking=True))
    plan_txt = json.dumps(plan, ensure_ascii=False) if isinstance(plan, dict) else str(plan)
    # call 2: ground the plan into two roles (retry once if it echoes placeholders)
    base = GROUND_PROMPT["user"].format(
        object_name=name, task=task.get("task", ""), goal=task.get("goal", ""),
        components=comp_txt, role_verbs=", ".join(ROLE_VERBS), plan=plan_txt)
    dec, roles = {}, []
    for attempt in range(2):
        prompt = base if attempt == 0 else base + ("\n\nOutput REAL verbs, REAL part names from the list, "
                                                   "and SPECIFIC contact regions - no placeholder text.")
        dec = parse_json(model.text_images(prompt, views, GROUND_PROMPT["system"], labels=labels))
        roles = dec.get("roles", [])
        if valid_decomp(roles):
            break
    if not valid_decomp(roles):
        return None   # drop degenerate query rather than emit placeholder garbage
    pattern = normalize_pair(dec)
    return {
        "task":          task.get("task"),
        "query":         task.get("query"),
        "goal":          task.get("goal"),
        "why_bimanual":  task.get("why_bimanual"),
        "category":      task.get("category"),
        "mechanism":     plan.get("moving_part") if isinstance(plan, dict) else None,
        "operation":     plan_txt,
        "coordination":  plan.get("coordination") if isinstance(plan, dict) else None,
        "roles":         roles,
        **pattern,
        "molmo_queries": [molmo_instruction(r) for r in roles],
        "answer":        [r.get("contact_region") or r.get("target") for r in roles],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="outputs/stage0_filtered.kept.json")
    ap.add_argument("--out", default="outputs/stage1_dataset.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--target_inter", type=int, default=3)  # qualified inter tasks (incl. fixed pick-up/move)
    ap.add_argument("--target_intra", type=int, default=3)  # qualified intra tasks
    ap.add_argument("--max_rounds", type=int, default=4)    # agentic regenerate rounds before giving up
    ap.add_argument("--model_id", default=os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-12B-it"))
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
    from get_component import make_grid

    objects = [o for o in json.load(open(inp)) if o.get("keep") and o.get("components")]
    end = args.end if args.end is not None else len(objects)
    batch = objects[args.start:end]
    print(f"stage1: {len(batch)} kept objects [{args.start}:{end}] of {len(objects)} -> {out}")

    model = Gemma(max_new_tokens=768)   # +256 headroom for the STEP 2 operation-mechanics CoT
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
    for i, obj in enumerate(batch):
        name, comps = obj["object_name"], obj["components"]
        if obj.get("object_id") in done:
            results.append(done[obj["object_id"]])
            continue
        comps = ensure_body_component(name, comps)   # guarantee a groundable rigid-body target
        try:
            views = obj.get("views_used", [])
            if not views or not all(os.path.exists(v) for v in views):
                print(f"  [{args.start+i}] {name[:28]:28} SKIP (views missing)")
                continue
            pairs = safe_prepare_views(views, VIEW_DIR)
            if len(pairs) < 5:
                print(f"  [{args.start+i}] {name[:28]:28} SKIP (corrupt renders)")
                continue
            clean = [p for _, p in pairs]
            labels = [view_label(views[idx]) for idx, _ in pairs]
            tasks = propose_tasks(model, name, comps, clean, labels=labels,
                                  target_inter=args.target_inter, target_intra=args.target_intra,
                                  max_rounds=args.max_rounds)
            queries = [q for q in (decompose_task(model, name, comps, t, clean, labels=labels)
                                    for t in tasks) if q]
            results.append({"object_id": obj["object_id"], "object_name": name,
                            "views_used": views, "components": comps, "queries": queries})
            print(f"  [{args.start+i}] {name[:28]:28} {len(queries)} queries")
        except Exception as e:
            results.append({"object_id": obj.get("object_id"), "object_name": name, "error": str(e)})
            print(f"  [{args.start+i}] ERROR {name}: {e}")

        with open(out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    total_q = sum(len(r.get("queries", [])) for r in results)
    print(f"\nstage1 done: {len(results)} objects, {total_q} queries -> {out}")


if __name__ == "__main__":
    main()
