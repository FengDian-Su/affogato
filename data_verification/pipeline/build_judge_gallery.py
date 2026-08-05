#!/usr/bin/env python3
"""Side-by-side gallery of a judge run against the gold labels, for eyeballing whether the judge's
0/1/2 assignments are sensible on their own merits - not only whether they match gold.

  python pipeline/build_judge_gallery.py \
      --judged .../stage1_multi/judged.jsonl --axis stage1 \
      --out quality_evaluation/claude_pilot_1000/judge_gallery.html

Cards carry the judge's score, its sub-scores when present, and its reason, next to the gold score,
fault and reason. Filters and a clickable gold x judge cross-tab operate over the whole page.
"""
import os
import json
import argparse
import collections

SC = {0: "bad", 1: "ok", 2: "good"}
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "quality_evaluation/claude_pilot_1000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--axis", choices=("stage1", "stage2"), default="stage1")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-page", type=int, default=100)
    a = ap.parse_args()
    base = os.path.dirname(os.path.abspath(a.out))

    def jl(p):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

    man = {r["sample_id"]: r for r in jl(f"{a.root}/manifest.jsonl")}
    gname = "metadata_labels.gold.jsonl" if a.axis == "stage1" else "heatmap_labels.gold.jsonl"
    gold = {r["sample_id"]: r for r in jl(f"{a.root}/{gname}")}
    judged = jl(a.judged)

    def img(p):
        return "imgs/" + p.split("/daily_used/")[-1] if os.path.islink(
            os.path.join(base, "imgs")) else os.path.relpath(p, base)

    rows = []
    for j in judged:
        sid = j["sample_id"]
        m, g = man[sid], gold[sid]
        rm = json.load(open(m["meta_path"])).get("roles", [])

        def role(x):
            return (f"<b>{x.get('role')}</b> {x.get('target')} &mdash; "
                    f"<i>{x.get('contact_region')}</i> &rarr; {x.get('function')}")
        sub = j.get("sub_scores") or {}
        rows.append({
            "sid": sid, "name": m["object_name"], "task": m["task"],
            "roles": "<br>".join(f"{'A' if i == 0 else 'B'}: {role(x)}" for i, x in enumerate(rm[:2])),
            "img": img(m["rgb_png"] if a.axis == "stage1" else m["heat_png"]),
            "p": j["score"] if j["score"] is not None else 1, "pr": j.get("reason") or "",
            "pf": j.get("fault") or "none",
            "sub": " ".join(f'<span class="sub s{v}">{k} {v}</span>' for k, v in sub.items()),
            "g": g["overall_score"], "gr": g.get("reason") or "", "gf": g.get("fault") or "none",
        })
    rows.sort(key=lambda r: (r["g"], r["p"], r["sid"]))

    n = len(rows)
    ct = collections.Counter((r["g"], r["p"]) for r in rows)
    gd = collections.Counter(r["g"] for r in rows)
    pd_ = collections.Counter(r["p"] for r in rows)
    agree = sum(ct[(k, k)] for k in (0, 1, 2))

    def bar(d):
        return "".join(f'<div class="seg s{k}" style="width:{d[k]/n*100:.1f}%">{SC[k]} {d[k]}</div>'
                       for k in (0, 1, 2))
    ctrows = ""
    for gg in (0, 1, 2):
        tds = "".join(f'<td class="ct{" diag" if gg == pp else ""}" data-g="{gg}" data-p="{pp}">'
                      f'<b>{ct[(gg, pp)]}</b><span>{ct[(gg, pp)]/n*100:.1f}%</span></td>'
                      for pp in (0, 1, 2))
        ctrows += f'<tr><th>gold {SC[gg]}</th>{tds}<th class="mg">{gd[gg]}</th></tr>'
    ctrows += ('<tr><th>&Sigma;</th>' + "".join(f'<th class="mg">{pd_[p]}</th>' for p in (0, 1, 2))
               + f'<th class="mg">{n}</th></tr>')

    cards = "".join(
        f'''<div class="card" data-g="{r['g']}" data-p="{r['p']}" data-pf="{r['pf']}">
<div class="hd"><span class="pill s{r['g']}">gold {SC[r['g']]}</span>
<span class="pill s{r['p']}">judge {SC[r['p']]}</span>
<b>{r['name']}</b> &mdash; {r['task']}</div>
<div class="body"><div class="txt">
<div class="roles">{r['roles']}</div>
<div class="j"><span class="tag s{r['p']}">judge</span>{r['pr']}
{('<em>' + r['pf'] + '</em>') if r['pf'] != 'none' else ''}{('<div class="subs">' + r['sub'] + '</div>') if r['sub'] else ''}</div>
<div class="j"><span class="tag s{r['g']}">gold</span>{r['gr']}
{('<em>' + r['gf'] + '</em>') if r['gf'] != 'none' else ''}</div>
<div class="dim mono">{r['sid']}</div></div>
<figure><img loading="lazy" src="{r['img']}"></figure></div></div>''' for r in rows)

    faults = "".join(f'<option value="{k}">{k} ({v})</option>' for k, v in
                     collections.Counter(r["pf"] for r in rows if r["pf"] != "none").most_common())

    HEAD = f'''<!doctype html><meta charset="utf-8"><title>judge vs gold &mdash; {a.axis}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:0;background:#fafafa;color:#222}}
header{{background:#fff;border-bottom:1px solid #ddd;padding:14px 20px}}
.sticky{{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid #ddd;padding:8px 20px;
box-shadow:0 1px 4px rgba(0,0,0,.06)}}
h1{{margin:0 0 10px;font-size:16px}} .dim{{color:#888}} .mono{{font-family:ui-monospace,monospace;font-size:11px}}
.bars{{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start}}
.barwrap{{flex:1;min-width:250px}} .bar{{display:flex;height:22px;border-radius:4px;overflow:hidden;font-size:11px}}
.seg{{display:flex;align-items:center;justify-content:center;color:#fff}}
.s0{{background:#d9534f}} .s1{{background:#e8a33d}} .s2{{background:#4c9e6f}}
table.ct{{border-collapse:collapse;font-size:12px}} table.ct td,table.ct th{{border:1px solid #ddd;padding:5px 10px;text-align:center}}
td.ct{{cursor:pointer}} td.ct:hover{{background:#eef4ff}} td.ct.diag{{background:#eef7f0}}
td.ct span{{display:block;color:#999;font-size:10px}} th.mg{{background:#f4f4f4;color:#666}}
.ctrls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}} select,button{{font:13px system-ui;padding:4px 8px}}
main{{padding:16px 20px;display:flex;flex-direction:column;gap:14px}}
.card{{background:#fff;border:1px solid #e2e2e2;border-radius:7px;overflow:hidden}}
.hd{{padding:8px 12px;border-bottom:1px solid #eee;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.pill{{color:#fff;border-radius:11px;padding:2px 9px;font-size:11px}}
.body{{display:flex;flex-direction:column;gap:12px;padding:12px}}
figure{{margin:0}} figure img{{width:100%;display:block;border:1px solid #eee;border-radius:4px}}
.roles{{font-size:12px;background:#f7f7f7;padding:7px 9px;border-radius:5px;line-height:1.5}}
.j{{font-size:12px;margin-top:6px}} .tag{{color:#fff;border-radius:4px;padding:1px 6px;margin-right:6px;font-size:11px}}
.j em{{display:block;color:#a33;font-style:normal;font-size:11px;margin-top:3px}}
.subs{{margin-top:4px}} .sub{{display:inline-block;color:#fff;border-radius:3px;padding:0 5px;margin-right:4px;font-size:10px}}
</style>
<header><h1>judge vs gold &mdash; {a.axis} &mdash; {os.path.basename(os.path.dirname(a.judged))}
<span class="dim">({n} dev samples, agreement {agree}/{n} = {agree/n:.0%})</span></h1>
<div class="bars">
<div class="barwrap"><div class="dim">gold</div><div class="bar">{bar(gd)}</div>
<div class="dim" style="margin-top:8px">judge</div><div class="bar">{bar(pd_)}</div></div>
<table class="ct"><tr><th></th><th>judge bad</th><th>judge ok</th><th>judge good</th><th class="mg">&Sigma;</th></tr>{ctrows}</table>
</div></header>
<div class="sticky"><div class="ctrls">
gold <select id="fg"><option value="">all</option><option value="0">bad</option><option value="1">ok</option><option value="2">good</option></select>
judge <select id="fp"><option value="">all</option><option value="0">bad</option><option value="1">ok</option><option value="2">good</option></select>
judge fault <select id="ff"><option value="">all</option>{faults}</select>
<button id="agree">agree only</button><button id="dis">disagree only</button>
<button id="reset">reset</button><span id="count" class="dim"></span>NAVSLOT
</div></div>
'''
    TAIL = '''<script>
const $=s=>document.querySelector(s), cards=[...document.querySelectorAll('.card')];
let mode='';
function apply(){
  const g=$('#fg').value,p=$('#fp').value,f=$('#ff').value; let n=0;
  for(const c of cards){
    let ok=(!g||c.dataset.g===g)&&(!p||c.dataset.p===p)&&(!f||c.dataset.pf===f);
    if(ok&&mode==='agree') ok=c.dataset.g===c.dataset.p;
    if(ok&&mode==='dis')   ok=c.dataset.g!==c.dataset.p;
    c.style.display=ok?'':'none'; if(ok)n++;
  }
  $('#count').textContent=n+' shown';
}
for(const id of ['fg','fp','ff']) $('#'+id).addEventListener('change',()=>{mode='';apply();});
$('#agree').addEventListener('click',()=>{mode='agree';apply();});
$('#dis').addEventListener('click',()=>{mode='dis';apply();});
$('#reset').addEventListener('click',()=>{mode='';for(const id of ['fg','fp','ff'])$('#'+id).value='';apply();});
for(const td of document.querySelectorAll('td.ct')) td.addEventListener('click',()=>{
  mode=''; $('#fg').value=td.dataset.g; $('#fp').value=td.dataset.p; $('#ff').value=''; apply();
  window.scrollTo({top:document.querySelector('header').offsetHeight,behavior:'smooth'});
});
apply();
</script>'''

    stem, ext = os.path.splitext(a.out)
    per = a.per_page or n
    chunks = [rows[i:i + per] for i in range(0, n, per)]
    names = [f"{os.path.basename(stem)}_p{i+1}{ext}" for i in range(len(chunks))]

    def nav(cur):
        links = " ".join(f'<a href="{nm}" class="{"cur" if i == cur else ""}">{i+1}</a>'
                         for i, nm in enumerate(names))
        return (f'<span style="margin-left:14px">page: {links} | '
                f'<a href="{os.path.basename(a.out)}">all&nbsp;{n}</a></span>')

    NAVCSS = ('<style>.ctrls a{padding:2px 7px;border:1px solid #ddd;border-radius:4px;'
              'text-decoration:none;color:#333}.ctrls a.cur{background:#333;color:#fff}</style>')
    card_of = {r["sid"]: c for r, c in zip(rows, cards.split('<div class="card"')[1:])}
    for i, (chunk, nm) in enumerate(zip(chunks, names)):
        body = "".join('<div class="card"' + card_of[r["sid"]] for r in chunk)
        open(os.path.join(base, nm), "w").write(
            HEAD.replace("NAVSLOT", nav(i)) + NAVCSS + f'<main>{body}</main>' + TAIL)
    open(a.out, "w").write(HEAD.replace("NAVSLOT", nav(-1)) + NAVCSS + f'<main>{cards}</main>' + TAIL)
    print(f"-> {a.out} ({n} cards, agreement {agree/n:.0%}) + {len(chunks)} pages")


if __name__ == "__main__":
    main()
