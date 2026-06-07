"""
Pilot: run the CURRENT generator (get_component -> get_affordance -> get_question) on a sample of
daily-used objects, using the daily_used_to_affogato.json mapping (abs paths) instead of the
electronics mapping. Faithful to the teammate's 3-stage chain; only the data source + paths differ.

Run from repo root with the `mm` env, e.g.:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python bimanual_annotation/run_daily_used_pilot.py
"""
import os
import sys
import json
import random

# make `from gemma import Gemma` (inside the teammate's modules) resolve, while cwd stays repo root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MAPPING = "data_generation/dataset/daily_used_to_affogato.json"
OUT     = "data_generation/output_daily_used_pilot.json"
N       = int(os.environ.get("PILOT_N", "15"))
SEED    = int(os.environ.get("PILOT_SEED", "7"))


def dry_check(samples):
    """Validate path wiring WITHOUT loading any model."""
    from get_component import load_object_name, sample_views
    print("=== DRY CHECK (no model) ===")
    ok = 0
    for ex in samples:
        name = load_object_name(ex["dst"])
        views = sample_views(ex["src"], 4)
        exist = sum(os.path.exists(v) for v in views)
        print(f"  {ex['object_id'][:12]}  name={name!r:30}  views={len(views)} exist={exist}  src_ok={os.path.isdir(ex['src'])}")
        if name != "unknown object" and exist == 4:
            ok += 1
    print(f"  -> wired OK: {ok}/{len(samples)}")
    return ok == len(samples)


def main():
    data = [d for d in json.load(open(MAPPING)) if d.get("dst")]
    random.seed(SEED)
    sample = random.sample(data, N)

    if "--dry" in sys.argv:
        dry_check(sample)
        return

    # quick wiring check before paying for model loads
    if not dry_check(sample):
        print("WARNING: some samples are not fully wired; continuing anyway.")

    from get_component import ComponentExtractor
    from get_affordance import AffordanceExtractor
    from get_question import QuestionGenerator

    ce = ComponentExtractor()
    ae = AffordanceExtractor()
    qg = QuestionGenerator()

    results = []
    for i, ex in enumerate(sample):
        rec = ce.predict(ex["src"], ex["dst"])      # {object_name, views_used, operable, final_components}
        rec["object_id"] = ex["object_id"]

        if rec.get("operable") and rec.get("final_components"):
            aff = ae.reason_affordance(rec["object_name"], rec["final_components"])
            rec["affordance"] = aff
            try:
                rec["question"] = qg.generate_question(
                    rec["object_name"], aff["reason"], aff["affordance_parts"]
                )
                rec["answer"] = aff["affordance_parts"]
            except Exception as e:
                rec["question"], rec["answer"] = None, None
                rec["error"] = f"question/answer failed: {e}"

        results.append(rec)
        comps = [c.get("name") for c in rec.get("final_components", []) if isinstance(c, dict)]
        print(f"\n[{i+1}/{N}] {rec.get('object_name')}  operable={rec.get('operable')}")
        print(f"    components: {comps}")
        print(f"    answer:     {rec.get('answer')}")
        print(f"    question:   {rec.get('question')}")
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    op = sum(1 for r in results if r.get("operable"))
    print(f"\nsaved -> {OUT}   ({op}/{len(results)} operable)")


if __name__ == "__main__":
    main()
