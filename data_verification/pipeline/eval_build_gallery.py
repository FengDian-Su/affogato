#!/usr/bin/env python
"""Build a paged local HTML gallery of the 1000 sampled stage2 results with Claude ground-truth
scores. Each card: task + roles + two score badges (axis1 task/role, axis2 heatmap) + reasons,
the RGB strip, and the heatmap render. References PNGs by relative path (open locally). Sorts worst-
first so problem cases surface. Filters by each axis score / coordination / category.

Labels are read from labels_axis1.json / labels_axis2.json ({ "<idx>": {"score","reason"} }) if
present; missing labels render as "pending". Regenerate after labeling completes.

  python build_gallery.py            # -> gallery_pXX.html + index
"""
import os, json, html

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "daily_used")
PER = 100

recs = json.load(open(f"{D}/labeling_records.json"))
def load(p):
    try:
        return {int(k): v for k, v in json.load(open(p)).items()}
    except Exception:
        return {}
a1 = load(f"{D}/labels_axis1.json")
a2 = load(f"{D}/labels_axis2.json")

BADGE = {0: ("BAD", "#e5484d"), 1: ("OK", "#f2a33c"), 2: ("GOOD", "#30a46c"), None: ("—", "#556")}
def badge(lbl):
    t, c = BADGE.get(lbl["score"] if lbl else None, BADGE[None])
    return f'<span class="b" style="background:{c}">{t}</span>'

def rel(p):  # path relative to the gallery html (lives in D)
    return os.path.relpath(p, D)

# sort worst-first: sum of available scores ascending; unlabeled last-ish (treat as 9)
def key(r):
    i = r["idx"]; s1 = a1.get(i, {}).get("score"); s2 = a2.get(i, {}).get("score")
    return ((s1 if s1 is not None else 9) + (s2 if s2 is not None else 9), i)
recs_sorted = sorted(recs, key=key)

CSS = """<style>
:root{--bg:#0f1216;--card:#171b21;--ink:#e6e8ec;--mut:#9aa3af;--line:#242a33}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:19px}.sub{color:var(--mut);font-size:13px}
.bar{padding:10px 24px;position:sticky;top:0;background:#0f1216ee;border-bottom:1px solid var(--line);
backdrop-filter:blur(6px);display:flex;gap:14px;flex-wrap:wrap;align-items:center;font-size:13px}
.bar b{color:var(--mut);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:16px;padding:18px 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.hd{padding:11px 14px;border-bottom:1px solid var(--line)}
.task{font-weight:700;font-size:15px}.oid{color:var(--mut);font:11px ui-monospace,Menlo,monospace}
.tags{margin-top:3px}.tag{display:inline-block;background:#20262e;color:#b9c2cd;border-radius:5px;
padding:0 7px;margin-right:5px;font-size:11px}
.b{display:inline-block;color:#fff;border-radius:5px;padding:1px 8px;font-size:11px;font-weight:800}
.scores{display:flex;gap:18px;padding:10px 14px;border-bottom:1px solid var(--line);font-size:12.5px}
.sc b{color:var(--mut);font-weight:600;margin-right:5px}.rsn{color:#c4ccd6;font-style:italic;margin-top:3px}
.roles{padding:9px 14px;font-size:12.5px;border-bottom:1px solid var(--line)}
.role{margin:2px 0}.rid{font-weight:800;color:#7db8ff}.reg{color:#ffd479}
.imgs img{width:100%;display:block;background:#000}
.imgs .cap{color:var(--mut);font-size:11px;padding:4px 14px}
.summary{padding:12px 24px;border-bottom:1px solid var(--line);background:#12161c}
.statrow{display:flex;align-items:center;gap:12px;margin:5px 0}
.stlbl{width:74px;color:var(--mut);font-size:12px;text-align:right}
.stbar{flex:0 0 320px;height:14px;border-radius:4px;overflow:hidden;display:flex;background:#222}
.stnum{font-size:12px}
.coordrow{margin-top:8px;color:var(--mut);font-size:12px}
.chip{display:inline-block;background:#1b2129;border:1px solid var(--line);border-radius:999px;
padding:2px 10px;margin:0 5px 5px 0}
.crosstab{margin-top:12px}.ctttl{color:var(--mut);font-size:12px;margin-bottom:5px}
.crosstab table{border-collapse:collapse;font-size:13px}
.crosstab th{color:var(--mut);font-weight:600;font-size:12px}
.crosstab td{border:1px solid var(--line);font-weight:700}
.crosstab td:hover{outline:2px solid #7db8ff}
</style>"""

JS = """<script>
function flt(){let a1=v('f1'),a2=v('f2'),co=v('fco'),ca=v('fca'),n=0;
document.querySelectorAll('.card').forEach(c=>{
 let ok=(a1=='*'||c.dataset.a1==a1)&&(a2=='*'||c.dataset.a2==a2)
   &&(co=='*'||c.dataset.co==co)&&(ca=='*'||c.dataset.ca==ca);
 c.style.display=ok?'':'none'; if(ok)n++;});
 let s=document.getElementById('shown'); if(s)s.textContent=n+' shown';}
function v(id){return document.getElementById(id).value}
function setf(t,h){document.getElementById('f1').value=t;document.getElementById('f2').value=h;flt();}
</script>"""

SCORE_LABEL = {0: "bad", 1: "ok", 2: "good"}


def opt(name, vals, score=False):
    if score:
        o = '<option value="*">all</option>' + "".join(
            f'<option value="{s}">{SCORE_LABEL[s]}</option>' for s in vals)
    else:
        o = '<option value="*">all</option>' + "".join(f"<option>{x}</option>" for x in vals)
    return f'<b>{name}</b><select id="{name.split()[0].lower()}" onchange="flt()">{o}</select>'

def card(r):
    i = r["idx"]; l1 = a1.get(i); l2 = a2.get(i)
    roles = "".join(
        f'<div class="role"><span class="rid">{x["id"]}</span> {html.escape(x["role"])} '
        f'@ <span class="reg">{html.escape(x["contact_region"])}</span> '
        f'<span style="color:#7f8896">(target {html.escape(str(x["target"]))})</span></div>'
        for x in r["roles"])
    def sc(name, lbl):
        r_ = f'<div class="rsn">{html.escape(lbl["reason"])}</div>' if lbl else ''
        return f'<div class="sc"><b>{name}</b>{badge(lbl)}{r_}</div>'
    d = (f'data-a1="{l1["score"] if l1 else ""}" data-a2="{l2["score"] if l2 else ""}" '
         f'data-co="{r["coordination"]}" data-ca="{r["category"]}"')
    return f"""<div class="card" {d}>
<div class="hd"><div class="task">#{i} · {html.escape(r["task"])}</div>
<div class="oid">{r["object_id"]} · {html.escape(r["object_name"])}</div>
<div class="tags"><span class="tag">{r["coordination"]}</span><span class="tag">{r["category"]}</span>
<span class="tag">cov {r["coverage"]}</span></div></div>
<div class="scores"><div>{sc("task/role", l1)}</div><div>{sc("heatmap", l2)}</div></div>
<div class="roles">{roles}</div>
<div class="imgs"><div class="cap">object scan (8 views)</div><img loading="lazy" src="{rel(r["rgb_png"])}">
<div class="cap">affordance heatmap · A=orange, B=tiffany · 8-view turntable</div>
<img loading="lazy" src="{rel(r["heat_png"])}"></div></div>"""

npages = (len(recs_sorted) + PER - 1) // PER
n1 = len(a1); n2 = len(a2)
from collections import Counter
dist1 = Counter(a1[i]["score"] for i in a1); dist2 = Counter(a2[i]["score"] for i in a2)

def stat_bar(name, d):
    """visual bad|ok|good proportion bar + counts."""
    n = max(sum(d.values()), 1)
    seg = "".join(f'<span style="display:inline-block;height:100%;width:{d[s]/n*100:.1f}%;'
                  f'background:{col}"></span>' for s, col in ((0, "#e5484d"), (1, "#f2a33c"), (2, "#30a46c")))
    return (f'<div class="statrow"><span class="stlbl">{name}</span>'
            f'<span class="stbar">{seg}</span>'
            f'<span class="stnum"><b style="color:#e5484d">bad {d[0]} ({d[0]/n*100:.0f}%)</b> · '
            f'<b style="color:#f2a33c">ok {d[1]} ({d[1]/n*100:.0f}%)</b> · '
            f'<b style="color:#30a46c">good {d[2]} ({d[2]/n*100:.0f}%)</b></span></div>')

# per-coordination axis2 mini-table (where stage2 grounding is strong/weak)
coord_rows = []
for cval in ["whole-body", "stabilize+actuate", "co-actuate"]:
    idxs = [i for i in a2 if str(recs[i]["coordination"]) == cval]
    if not idxs:
        continue
    c = Counter(a2[i]["score"] for i in idxs); n = len(idxs)
    coord_rows.append(f'<span class="chip">{cval}: <b style="color:#30a46c">{c[2]/n*100:.0f}% good</b> '
                      f'/ <b style="color:#e5484d">{c[0]/n*100:.0f}% bad</b> (n={n})</span>')
# clickable cross-tab: task/role (rows) x heatmap (cols); click a cell to filter both axes to it
both_pairs = [(a1[i]["score"], a2[i]["score"]) for i in a2 if i in a1]
ct = {(t, h): sum(1 for x, y in both_pairs if x == t and y == h) for t in (0, 1, 2) for h in (0, 1, 2)}
CTCOL = {0: "#e5484d", 1: "#f2a33c", 2: "#30a46c"}
NB = max(len(both_pairs), 1)
ctm_row = {t: sum(ct[(t, h)] for h in (0, 1, 2)) for t in (0, 1, 2)}   # task marginals
ctm_col = {h: sum(ct[(t, h)] for t in (0, 1, 2)) for h in (0, 1, 2)}   # heat marginals
def ct_cell(t, h):
    n = ct[(t, h)]; pct = n / NB * 100; shade = min(0.12 + n / max(ct.values()) * 0.6, 0.85)
    return (f'<td onclick="setf(\'{t}\',\'{h}\')" '
            f'title="task/role={SCORE_LABEL[t]}, heatmap={SCORE_LABEL[h]}: {n} results ({pct:.1f}% of {NB})" '
            f'style="cursor:pointer;text-align:center;padding:7px 12px;'
            f'background:rgba(125,184,255,{shade:.2f})">{pct:.1f}%</td>')
ct_head = ("".join(f'<th style="color:{CTCOL[h]};padding:4px 12px">heat {SCORE_LABEL[h]}</th>' for h in (0, 1, 2))
           + '<th style="color:#9aa3af;padding:4px 12px">Σ task</th>')
ct_body = "".join(
    f'<tr><th style="color:{CTCOL[t]};text-align:right;padding:4px 8px">task {SCORE_LABEL[t]}</th>'
    + "".join(ct_cell(t, h) for h in (0, 1, 2))
    + f'<td style="text-align:center;padding:7px 12px;color:#9aa3af">{ctm_row[t]/NB*100:.0f}%</td></tr>'
    for t in (0, 1, 2))
ct_foot = ('<tr><th style="color:#9aa3af;text-align:right;padding:4px 8px">Σ heat</th>'
           + "".join(f'<td style="text-align:center;padding:7px 12px;color:#9aa3af">{ctm_col[h]/NB*100:.0f}%</td>'
                     for h in (0, 1, 2)) + '<td style="text-align:center;color:#9aa3af">100%</td></tr>')
CROSSTAB = (f'<div class="crosstab"><div class="ctttl">cross-tab (% of {NB}) · click a cell to filter '
            f'· task/role &#215; heatmap · margins Σ show each axis\'s own distribution</div>'
            f'<table><tr><th></th>{ct_head}</tr>{ct_body}{ct_foot}</table></div>')
SUMMARY = (f'<div class="summary">{stat_bar("task/role", dist1)}{stat_bar("heatmap", dist2)}'
           f'<div class="coordrow">heatmap by coordination — {" ".join(coord_rows)}</div>'
           f'{CROSSTAB}</div>')

for pg in range(npages):
    chunk = recs_sorted[pg*PER:(pg+1)*PER]
    nav = " ".join(f'<a href="gallery_p{p}.html" style="color:{"#fff" if p==pg else "#7db8ff"};'
                   f'margin-right:8px">{p}</a>' for p in range(npages))
    bar = ('<div class="bar">' + opt("f1 task/role", [0,1,2], score=True)
           + opt("f2 heatmap", [0,1,2], score=True)
           + opt("fco coord", sorted({str(r["coordination"]) for r in recs}))
           + opt("fca cat", sorted({str(r["category"]) for r in recs}))
           + '<span id="shown" style="color:#7db8ff;font-weight:700"></span>'
           + '<button onclick="setf(\'*\',\'*\')" '
           'style="background:#242a34;color:#e6e8ec;border:1px solid #3b4352;border-radius:6px;'
           'padding:3px 9px;cursor:pointer">reset</button></div>')
    hdr = (f'<header><h1>stage2 result review — daily_used (1000 sampled, seed 42)</h1>'
           f'<div class="sub">worst-first. axis1 task/role labeled {n1}/1000 '
           f'(bad {dist1[0]} / ok {dist1[1]} / good {dist1[2]}); '
           f'axis2 heatmap labeled {n2}/1000 (bad {dist2[0]} / ok {dist2[1]} / good {dist2[2]}). '
           f'page {pg} · {nav}</div></header>')
    body = "".join(card(r) for r in chunk)
    open(f"{D}/gallery_p{pg}.html", "w").write(
        f"<!doctype html><meta charset=utf-8>{CSS}{JS}{hdr}{SUMMARY}{bar}<div class='grid'>{body}</div>")
# single-page filterable view (all cards) so the filters + cross-tab slice the WHOLE 1000, not one page
bar_all = ('<div class="bar">' + opt("f1 task/role", [0,1,2], score=True)
           + opt("f2 heatmap", [0,1,2], score=True)
           + opt("fco coord", sorted({str(r["coordination"]) for r in recs}))
           + opt("fca cat", sorted({str(r["category"]) for r in recs}))
           + '<span id="shown" style="color:#7db8ff;font-weight:700"></span>'
           + '<button onclick="setf(\'*\',\'*\')" style="background:#242a34;color:#e6e8ec;'
           'border:1px solid #3b4352;border-radius:6px;padding:3px 9px;cursor:pointer">reset</button></div>')
hdr_all = (f'<header><h1>stage2 result review — daily_used (all 1000, seed 42)</h1>'
           f'<div class="sub">single-page filterable view · worst-first · '
           f'<a href="gallery_p0.html" style="color:#7db8ff">paged version</a></div></header>')
body_all = "".join(card(r) for r in recs_sorted)
open(f"{D}/gallery.html", "w").write(
    f"<!doctype html><meta charset=utf-8>{CSS}{JS}{hdr_all}{SUMMARY}{bar_all}<div class='grid'>{body_all}</div>")
print(f"wrote {npages} paged + gallery.html (all 1000) -> {D}/gallery.html")
print(f"axis1 {n1}/1000 labeled {dict(dist1)}; axis2 {n2}/1000 labeled {dict(dist2)}")
print(f"cross-tab task x heat: {dict(ct)}")
