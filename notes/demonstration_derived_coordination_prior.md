# Demonstration-Derived Coordination Prior for Bimanual Affordance Generation

## Motivation

Traditional affordance annotation usually maps an object and a query to a single interaction region:

```text
object + affordance query -> one affordance region
```

For bimanual affordance, this formulation is not enough. The important supervision is not just two independent regions, but a coordinated pair:

```text
object + task -> role A region + role B region + coordination relation
```

The goal is to build a hand-agnostic bimanual dataset where each high-level daily task is decomposed into two robot-executable role affordances. These roles should be grounded in object parts and should explain how the two interactions cooperate physically.

The key idea here is to learn or distill a compact prior from human demonstration videos. Instead of directly learning robot trajectories from human motion, we extract structured bimanual coordination knowledge:

```text
what task is being performed
which object is manipulated
how the two hands interact with the object
which roles are used
which object parts are contacted
what physical relation connects the two roles
```

This produces a demonstration-derived knowledge base that can guide task proposal and role decomposition.

## High-Level Formulation

Given an object observation, the generation pipeline should be:

```text
object image / render
-> task proposal
-> task decomposition into two hand-agnostic roles
-> role-specific grounding with Molmo
-> paired 3D heatmap regions
```

The demonstration-derived prior is used before grounding:

```text
human videos
-> structured role-pair knowledge base
-> compact prompt prior
-> better task proposal and role decomposition
```

For the first version, we do not use retrieval or RAG. The knowledge base is distilled into a compact table of canonical bimanual coordination patterns and provided directly in the prompt.

## Role Vocabulary

Roles should be simple robot-executable verbs. Avoid overly semantic or task-level labels. The role name should describe what a robot hand can physically do.

Recommended minimal role set:

```text
hold
support
lift
push
pull
press
rotate
slide
tilt
insert
separate
align
```

Notes:

- `open` should usually be a task, not a role. Opening can be implemented by `pull`, `rotate`, `slide`, `lift`, or `separate`.
- `grasp` is usually an execution/contact mode, not a role. Both `hold` and `rotate` may require grasping.
- `pick` and `place` are full manipulation skills, but they are less useful as bimanual role labels.

Each role should keep only a few essential fields:

```json
{
  "role": "hold",
  "target": "bottle body",
  "function": "resist rotation"
}
```

## Coordination Relations

The coordination relation describes why the two role regions form a pair. This should be a compact label or phrase, not a long explanation.

Recommended relation set:

```text
stabilize and actuate
opposing force
opposing torque
shared support
orientation control
guide and actuate
constrained translation
part separation
```

Examples:

```text
hold + rotate   -> opposing torque
hold + pull     -> stabilize and actuate
support + support -> shared support
support + tilt  -> orientation control
pull + pull     -> opposing force
align + insert  -> guide and actuate
hold + slide    -> constrained translation
hold + separate -> part separation
```

## Knowledge Base Entry

Each human demonstration clip should be converted into a structured JSON entry:

```json
{
  "task": "open the bottle",
  "object": "bottle",
  "roles": [
    {
      "role": "hold",
      "target": "bottle body",
      "function": "resist rotation"
    },
    {
      "role": "rotate",
      "target": "bottle cap",
      "function": "apply torque"
    }
  ],
  "relation": "opposing torque",
  "evidence": {
    "source": "human demonstration video",
    "dataset": "EPIC-Kitchens",
    "clip_id": "...",
    "timestamp": [12.3, 18.7],
    "caption": "a person opens a bottle",
    "confidence": 0.86
  }
}
```

The evidence fields are useful for auditing, but they do not need to be passed into the generation prompt. The prompt should use only the distilled canonical patterns.

## Canonical Pattern Table

After collecting many video-derived entries, merge them into a smaller table of common bimanual patterns:

```json
[
  {
    "object_type": "bottle or jar",
    "task": "open",
    "roles": [
      {"role": "hold", "target": "body", "function": "keep object stable"},
      {"role": "rotate", "target": "cap or lid", "function": "apply torque"}
    ],
    "relation": "opposing torque"
  },
  {
    "object_type": "box, tray, or large container",
    "task": "carry",
    "roles": [
      {"role": "support", "target": "one side or bottom", "function": "share load"},
      {"role": "support", "target": "opposite side or bottom", "function": "share load"}
    ],
    "relation": "shared support"
  },
  {
    "object_type": "package, bag, or wrapper",
    "task": "tear open",
    "roles": [
      {"role": "pull", "target": "one edge or flap", "function": "apply pulling force"},
      {"role": "pull", "target": "opposite edge or flap", "function": "apply opposing pulling force"}
    ],
    "relation": "opposing force"
  },
  {
    "object_type": "container with liquid",
    "task": "pour",
    "roles": [
      {"role": "support", "target": "handle or body", "function": "support object weight"},
      {"role": "tilt", "target": "body or spout side", "function": "control pouring direction"}
    ],
    "relation": "orientation control"
  },
  {
    "object_type": "object with sliding cover or latch",
    "task": "slide open",
    "roles": [
      {"role": "hold", "target": "base or body", "function": "keep object stable"},
      {"role": "slide", "target": "cover or latch", "function": "translate movable part"}
    ],
    "relation": "constrained translation"
  },
  {
    "object_type": "object with socket, slot, or opening",
    "task": "insert",
    "roles": [
      {"role": "align", "target": "socket, slot, or opening", "function": "guide insertion target"},
      {"role": "insert", "target": "insertable part", "function": "move part into opening"}
    ],
    "relation": "guide and actuate"
  }
]
```

This table is the compact demonstration-derived prior for the first version.

## Video-to-KB Extraction Pipeline

Candidate video sources:

```text
Ego4D
EPIC-Kitchens
YouTube how-to videos
other egocentric or instructional manipulation videos
```

Proposed extraction flow:

```text
video
-> sample clips
-> use VLM to find bimanual single-object interactions
-> caption the task
-> describe each hand's interaction with the object
-> map free-form descriptions to the fixed role vocabulary
-> output structured JSON
-> validate and canonicalize
-> add to knowledge base
```

The VLM prompt should ask for:

```text
1. whether the clip contains two-hand manipulation of one primary object
2. the daily task being performed
3. the primary object
4. what each hand contacts
5. what each hand does physically
6. the role pair using the fixed role vocabulary
7. the coordination relation
8. whether the example is reliable enough to keep
```

## Example VLM Prompt

```text
You are analyzing a human manipulation video clip.

Identify whether the clip shows two hands coordinating to manipulate one primary object.
If it does not, return {"valid": false}.

If it does, output a JSON object with:
- task: a short daily-life task description
- object: the manipulated object
- roles: exactly two hand-agnostic role affordances
- relation: the physical coordination relation between the two roles
- evidence: a short caption and confidence score

Use only these role verbs:
hold, support, lift, push, pull, press, rotate, slide, tilt, insert, separate, align.

Keep role fields minimal:
- role
- target
- function

Prefer physical functions such as:
resist motion, apply torque, apply pulling force, share load, guide alignment,
translate movable part, control orientation.
```

Expected output:

```json
{
  "valid": true,
  "task": "open the bottle",
  "object": "bottle",
  "roles": [
    {
      "role": "hold",
      "target": "bottle body",
      "function": "resist rotation"
    },
    {
      "role": "rotate",
      "target": "bottle cap",
      "function": "apply torque"
    }
  ],
  "relation": "opposing torque",
  "evidence": {
    "caption": "a person holds the bottle body with one hand and twists the cap with the other",
    "confidence": 0.86
  }
}
```

## Validation and Canonicalization

Raw VLM outputs should be cleaned before entering the knowledge base.

Reject examples if:

- the task can be naturally completed with one hand
- there is no single primary manipulated object
- one hand is only idle or incidental
- both roles touch the same part without a meaningful coordination relation
- the output uses a role outside the fixed vocabulary
- the target part is too vague to ground later
- the relation is not physically meaningful

Canonicalize examples by mapping:

```text
twist -> rotate
grip / grasp / keep steady -> hold
carry / cradle -> support
push down -> press
move cover sideways -> slide
open by pulling apart -> separate or pull + pull
```

The goal is not to preserve every detail of the human motion. The goal is to preserve the reusable bimanual coordination pattern.

## Using the Prior Without RAG

For the first version, do not build retrieval. Instead:

1. Mine many video-derived entries.
2. Merge them into a small canonical pattern table.
3. Put the table directly into the task proposal / decomposition prompt.
4. Ask the LLM to choose the closest pattern when generating each task.

The generation prompt should say:

```text
Use the demonstration-derived bimanual coordination patterns below as prior knowledge.
For each visible object, propose daily-life bimanual tasks.
Each task must require two complementary role affordances.
For each task, choose the closest coordination pattern.
Then output exactly two hand-agnostic roles.
```

The output for dataset generation should be:

```json
{
  "task": "open the bottle",
  "roles": [
    {
      "id": "A",
      "role": "hold",
      "target": "bottle body",
      "function": "resist rotation"
    },
    {
      "id": "B",
      "role": "rotate",
      "target": "bottle cap",
      "function": "apply torque"
    }
  ],
  "relation": "opposing torque"
}
```

This output is then converted into role-specific Molmo queries:

```text
Point to the bottle body where a robot hand should hold it to resist rotation.
Point to the bottle cap where a robot hand should rotate it to apply torque.
```

## Why This Is More Than Prompt Engineering

The method is not simply using a few examples in the prompt. The important step is distillation:

```text
human manipulation videos
-> many noisy VLM annotations
-> canonical role-pair patterns
-> compact coordination prior
-> task-conditioned bimanual affordance generation
```

The contribution can be framed as:

> We distill common bimanual coordination patterns from human demonstration videos into a compact role-pair prior. This prior guides task proposal and task decomposition, enabling generation of hand-agnostic paired affordance regions for robot manipulation.

This keeps the system simple while still grounding the role-pair design in observed human demonstrations.

## Open Design Questions

- How many canonical patterns are enough for broad daily objects?
- Should the KB keep object-specific patterns only, or also abstract object types such as "capped container" and "sliding mechanism"?
- Should low-confidence video annotations be used for statistics, or discarded entirely?
- Should role pair frequency be used to rank task proposals?
- How strict should the validator be when one task can be done with either one or two hands?

## Suggested V1 Scope

V1 should avoid over-engineering:

```text
no RAG
no video model training
no robot trajectory learning
no left/right hand assignment
```

V1 should implement:

```text
fixed role vocabulary
fixed relation vocabulary
video-derived canonical pattern table
task proposal prompt using the pattern table
role decomposition prompt using the pattern table
validation rules for rejecting weak bimanual examples
Molmo grounding with role-specific target queries
```

This gives a clean first-stage system:

```text
object observation
-> demonstration-guided task proposal
-> role-pair decomposition
-> hand-agnostic paired affordance heatmaps
```
