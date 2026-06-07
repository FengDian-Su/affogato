#!/usr/bin/env python
"""
build_gallery.py — bundle the rendered heatmap PNGs into ONE self-contained HTML.

Images are embedded as base64 data URIs (no external files, no scripts), so the page
renders in any browser even offline / when opened directly. Per-object metrics from
compare_results_fixed are shown as captions.

Usage:
  python build_gallery.py --images images --csv compare_results_fixed/per_query_metrics.csv \
         --out gallery.html
"""
import os, csv, glob, base64, argparse
from collections import defaultdict


def per_object_metrics(csv_path):
    rows = defaultdict(list)
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("mode") != "visible":
                continue
            rows[r["object_id"]].append(r)
    out = {}
    for oid, rs in rows.items():
        def m(k):
            vals = [float(r[k]) for r in rs if r.get(k) not in (None, "", "None")]
            return sum(vals) / len(vals) if vals else float("nan")
        out[oid] = {"auc": m("auc"), "spearman": m("spearman"), "cc": m("cc"),
                    "sim": m("sim"), "coverage": m("coverage")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="assets/images")
    ap.add_argument("--csv", default="results/compare_results_fixed/per_query_metrics.csv")
    ap.add_argument("--out", default="galleries/gallery.html")
    ap.add_argument("--title", default="AFFOGATO reproduction — GT vs ours (frame-aligned)")
    args = ap.parse_args()

    pngs = sorted(glob.glob(os.path.join(args.images, "*.png")))
    if not pngs:
        print(f"no PNGs in {args.images}"); return
    metrics = per_object_metrics(args.csv)

    # overall mean
    allm = {k: [] for k in ("auc", "spearman", "cc", "sim")}
    for v in metrics.values():
        for k in allm:
            allm[k].append(v[k])
    mean = {k: (sum(x) / len(x) if x else float("nan")) for k, x in allm.items()}

    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>{args.title}</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:24px;background:#111;color:#eee}}
 h1{{font-size:20px}} h2{{font-size:15px;margin:6px 0 2px}}
 .sub{{color:#aaa;font-size:13px;margin-bottom:18px;line-height:1.5}}
 .card{{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:10px 12px;margin:18px 0}}
 .meta{{color:#9cf;font-size:13px;margin:2px 0 8px}}
 img{{width:100%;height:auto;border-radius:4px;background:#000}}
 code{{color:#fc9}}
</style></head><body>
<h1>{args.title}</h1>
<div class="sub">
 Each card: <b>top row = AFFOGATO GT heatmap</b>, <b>bottom row = our reproduction</b>,
 one column per query, on the affogato 16384 points, 45&deg; angled top-down view.
 Color = affordance score (red = high), per-panel normalized — compare <i>where</i> the
 hot region is, not absolute color.<br>
 Overall (visible points, 8 objects): <code>AUC {mean['auc']:.2f}</code> ·
 <code>CC {mean['cc']:.2f}</code> · <code>Spearman {mean['spearman']:.2f}</code> ·
 <code>SIM {mean['sim']:.2f}</code>.
</div>"""]

    for p in pngs:
        oid = os.path.splitext(os.path.basename(p))[0]
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        mt = metrics.get(oid)
        cap = (f"AUC {mt['auc']:.2f} · CC {mt['cc']:.2f} · Spearman {mt['spearman']:.2f} · "
               f"SIM {mt['sim']:.2f} · coverage {mt['coverage']*100:.0f}%") if mt else ""
        parts.append(f'<div class="card"><h2>{oid}</h2><div class="meta">{cap}</div>'
                     f'<img src="data:image/png;base64,{b64}"></div>')

    parts.append("</body></html>")
    html = "\n".join(parts)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}  ({len(html)/1e6:.1f} MB, {len(pngs)} objects, self-contained)")


if __name__ == "__main__":
    main()
