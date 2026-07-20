#!/usr/bin/env python
"""
Stage 0 refinement — one entry point for the two QC screens and for APPLYING their result.

Screens (independent by design; a model family must never verify its own filtering decisions):
  semantic  judge_semantic.py     Qwen3.5-35B-A3B re-examines every kept object's renders
  geometry  check_geometry_stats.py  alpha-silhouette stats (flat sheet / ground slab / empty)

Measured on stage0_full (2026-07-20, 6,174 flags arbitrated by Claude on the renders):
  both screens agree  93%真junk | semantic only 75% | geometry only 40%
So the screens are a RECALL net, not a verdict: arbitrate flags before applying them. A residual
check on 150 double-passed objects found 18% junk still escapes both screens - the screens shrink
the problem, they do not close it.

  # 1. screen everything (writes judge_qwen_part*.json + geomflags.json + qc_flags.json)
  python verify/refine_stage0.py screen --gpu 0
  # 2. (arbitrate qc_flags.json however you like - Claude image review is the calibrated path)
  # 3. apply an arbitrated junk list to the ORIGINAL part jsons, in place (backup first)
  python verify/refine_stage0.py apply --junk refinement_junk_list.json
  python verify/refine_stage0.py apply --junk ... --dry-run     # preview only
  python verify/refine_stage0.py restore                       # undo the last apply
  python verify/refine_stage0.py status

`apply` flips keep -> false on the listed objects, records WHY on the record
(qc_junk_type / qc_evidence / qc_note / skip_reason), and regenerates every .kept.json so the
files stage1 reads are already clean. Records are never deleted: the full json keeps the audit
trail, and `restore` puts the pre-apply state back from the backup.
"""
import os
import sys
import json
import shutil
import argparse
import subprocess
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))                 # data_generation/verify/
DATA_GEN = os.path.dirname(HERE)
DEFAULT_DIR = os.path.join(DATA_GEN, "outputs", "stage0", "daily_used")


def part_files(d):
    """Every stage0_part{N}.json in the dir, N-ordered (part count varies per category)."""
    import glob, re
    out = []
    for f in glob.glob(os.path.join(d, "stage0_part*.json")):
        m = re.search(r"stage0_part(\d+)\.json$", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def load_kept(d):
    kept = {}
    for p, f in part_files(d):
        for r in json.load(open(f)):
            if r.get("keep"):
                kept[r["object_id"]] = (p, r)
    return kept


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------- screen
def cmd_screen(args):
    d = args.dir
    py = sys.executable
    if not args.skip_geometry:
        print("== geometry screen ==", flush=True)
        subprocess.run([py, os.path.join(HERE, "check_geometry_stats.py"),
                        "--dir", d, "--workers", str(args.workers)], check=True)
    if not args.skip_semantic:
        print("== semantic screen (this is the long one) ==", flush=True)
        subprocess.run([py, os.path.join(HERE, "judge_semantic.py"),
                        "--dir", d, "--gpu", str(args.gpu), "--judge_model", args.judge_model],
                       check=True)
    merge_flags(d)


def merge_flags(d):
    """Combine both screens into qc_flags.json: one record per flagged object + its evidence."""
    kept = load_kept(d)
    sem = {}
    for p, _ in part_files(d):
        f = os.path.join(d, f"judge_qwen_part{p}.json")
        if os.path.exists(f):
            for r in json.load(open(f)):
                sem[r["object_id"]] = r
    geom = {}
    gf = os.path.join(d, "geomflags.json")
    if os.path.exists(gf):
        geom = {r["object_id"]: r for r in json.load(open(gf))}

    out = []
    for oid, (p, rec) in kept.items():
        s, g = sem.get(oid, {}), geom.get(oid)
        s_junk = s.get("verdict") == "junk"
        if not s_junk and not g:
            continue
        out.append({"object_id": oid, "part": p, "object_name": rec.get("object_name"),
                    "evidence": "both" if (s_junk and g) else ("semantic" if s_junk else "geometry"),
                    "semantic_types": s.get("junk_types") or [], "semantic_notes": s.get("notes"),
                    "semantic_actual": s.get("actual_object"),
                    "geometry_flags": g["flags"] if g else [],
                    "components": [c.get("name") for c in (rec.get("components") or [])],
                    "views": rec.get("views_used")})
    out.sort(key=lambda r: (r["part"], r["evidence"]))
    write_json(os.path.join(d, "qc_flags.json"), out)
    print(f"qc_flags.json: {len(out)} flagged of {len(kept)} kept "
          f"({len(out)/max(len(kept),1)*100:.1f}%)  {Counter(r['evidence'] for r in out)}")


# ---------------------------------------------------------------- apply
def cmd_apply(args):
    d = args.dir
    junk_path = args.junk if os.path.isabs(args.junk) else os.path.join(d, args.junk)
    junk = {r["object_id"]: r for r in json.load(open(junk_path))}
    print(f"junk list: {len(junk)} object_ids from {os.path.basename(junk_path)}")

    bak = os.path.join(d, "pre_apply_backup")
    if not args.dry_run:
        if os.path.isdir(bak) and not args.force:
            sys.exit(f"backup already exists ({bak}); `restore` first, or pass --force to overwrite")
        os.makedirs(bak, exist_ok=True)

    applied, missing, per_part = 0, 0, Counter()
    for p, f in part_files(d):
        recs = json.load(open(f))
        n_hit = 0
        for r in recs:
            j = junk.get(r.get("object_id"))
            if not j or not r.get("keep"):
                continue
            r["keep"] = False
            r["skip_reason"] = f"qc-junk: {j.get('junk_type', 'unspecified')}"
            r["qc_junk_type"] = j.get("junk_type")
            r["qc_evidence"] = j.get("evidence")
            r["qc_note"] = j.get("what_i_see")
            n_hit += 1
        kept = [r for r in recs if r.get("keep")]
        kf = os.path.splitext(f)[0] + ".kept.json"
        per_part[p] = n_hit
        applied += n_hit
        print(f"  part{p}: -{n_hit:>4} -> kept {len(kept)}")
        if args.dry_run:
            continue
        shutil.copy2(f, os.path.join(bak, os.path.basename(f)))
        if os.path.exists(kf):
            shutil.copy2(kf, os.path.join(bak, os.path.basename(kf)))
        write_json(f, recs)
        write_json(kf, kept)

    seen = set()
    for _, f in part_files(d):
        for r in json.load(open(f)):
            seen.add(r.get("object_id"))
    missing = len([oid for oid in junk if oid not in seen])
    print(f"\n{'DRY RUN - nothing written' if args.dry_run else 'applied'}: "
          f"{applied} objects flipped to keep=false"
          + (f"; {missing} listed ids not found in the parts" if missing else ""))
    if not args.dry_run:
        print(f"backup of the pre-apply files -> {bak}  (undo: refine_stage0.py restore)")


def cmd_restore(args):
    bak = os.path.join(args.dir, "pre_apply_backup")
    if not os.path.isdir(bak):
        sys.exit("no backup to restore")
    n = 0
    for fn in sorted(os.listdir(bak)):
        shutil.copy2(os.path.join(bak, fn), os.path.join(args.dir, fn))
        n += 1
    shutil.rmtree(bak)
    print(f"restored {n} files from backup; backup removed")


def cmd_status(args):
    d = args.dir
    tot = kept = junked = 0
    types, reasons = Counter(), Counter()
    for _, f in part_files(d):
        for r in json.load(open(f)):
            tot += 1
            if r.get("keep"):
                kept += 1
            if r.get("qc_junk_type"):
                junked += 1
                types[r["qc_junk_type"]] += 1
                reasons[r.get("qc_evidence", "?")] += 1
    print(f"records {tot} | kept {kept} ({kept/tot*100:.1f}%) | qc-removed {junked}")
    if types:
        print("removed by type:    ", dict(types.most_common()))
        print("removed by evidence:", dict(reasons.most_common()))
    for f in ("qc_flags.json", "geomflags.json", "refinement_junk_list.json"):
        p = os.path.join(d, f)
        print(f"  {f:28} {'-' if not os.path.exists(p) else str(len(json.load(open(p)))) + ' records'}")
    print(f"  pre_apply_backup            {'present (restore available)' if os.path.isdir(os.path.join(d, 'pre_apply_backup')) else '-'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen", help="run both screens + merge flags")
    s.add_argument("--gpu", type=int, default=0)
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--judge_model", default="qwen", choices=["qwen", "gemma"])
    s.add_argument("--skip-semantic", action="store_true")
    s.add_argument("--skip-geometry", action="store_true")
    s.set_defaults(func=cmd_screen)

    m = sub.add_parser("merge", help="re-merge existing screen outputs into qc_flags.json")
    m.set_defaults(func=lambda a: merge_flags(a.dir))

    a = sub.add_parser("apply", help="apply an arbitrated junk list to the original part jsons")
    a.add_argument("--junk", default="refinement_junk_list.json")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--force", action="store_true", help="overwrite an existing backup")
    a.set_defaults(func=cmd_apply)

    r = sub.add_parser("restore", help="undo the last apply")
    r.set_defaults(func=cmd_restore)

    st = sub.add_parser("status", help="show current keep/qc counts")
    st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
