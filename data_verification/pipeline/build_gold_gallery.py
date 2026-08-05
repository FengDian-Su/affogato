#!/usr/bin/env python3
"""Single-page gallery for the re-labelled Claude gold set (both axes).

  python pipeline/build_gold_gallery.py \
      --dir quality_evaluation/claude_pilot_1000 --out .../gold_gallery.html

Shows, per sample: the 8-view RGB montage and the 8-view heatmap render side by side, the task and
both hand roles, and the new metadata / heatmap score with its reason and fault tag. Filters and a
clickable cross-tab operate over all 1000 cards at once (images are lazy-loaded).
"""
import os
import json
import argparse
import collections

SC = {0: "bad", 1: "ok", 2: "good"}


def rel(path, base):
    """Prefer the in-directory `imgs` symlink so the page never references anything above its own
    folder - webview previews refuse paths that escape the workspace root."""
    p = os.path.relpath(path, base)
    for tag in ("/rgb/", "/renders/"):
        if tag in path and os.path.islink(os.path.join(base, "imgs")):
            return "imgs" + tag + os.path.basename(path)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="claude_pilot_1000 directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-page", type=int, default=100,
                    help="also emit <out>_pN.html chunks this size (0 = single page only)")
    a = ap.parse_args()
    D = os.path.abspath(a.dir)
    base = os.path.dirname(os.path.abspath(a.out))

    man = {json.loads(l)["sample_id"]: json.loads(l) for l in open(f"{D}/manifest.jsonl")}
    meta = {json.loads(l)["sample_id"]: json.loads(l) for l in open(f"{D}/metadata_labels.gold.jsonl")}
    heat = {json.loads(l)["sample_id"]: json.loads(l) for l in open(f"{D}/heatmap_labels.gold.jsonl")}

    rows = []
    for sid in sorted(meta, key=lambda s: (meta[s]["overall_score"], heat[s]["overall_score"], s)):
        m, g, h = man[sid], meta[sid], heat[sid]
        rm = json.load(open(m["meta_path"])).get("roles", [])

        def role(x):
            return (f"<b>{x.get('role')}</b> {x.get('target')} &mdash; "
                    f"<i>{x.get('contact_region')}</i> &rarr; {x.get('function')}")
        roles = "<br>".join(f"{'A' if i == 0 else 'B'}: {role(x)}" for i, x in enumerate(rm[:2]))
        rows.append({
            "sid": sid, "name": m["object_name"], "task": m["task"],
            "cat": m.get("category"), "coord": m.get("coordination"), "roles": roles,
            "rgb": rel(m["rgb_png"], base), "heat": rel(m["heat_png"], base),
            "m": g["overall_score"], "mr": g["reason"], "mf": g["fault"], "grp": g["group"],
            "h": h["overall_score"], "hr": h["reason"], "hf": h["fault"],
        })

    md = collections.Counter(r["m"] for r in rows)
    hd = collections.Counter(r["h"] for r in rows)
    ct = collections.Counter((r["m"], r["h"]) for r in rows)
    n = len(rows)
    mf = collections.Counter(r["mf"] for r in rows if r["m"] == 0)
    hf = collections.Counter(r["hf"] for r in rows if r["h"] < 2)

    def bar(dist, cls):
        return "".join(
            f'<div class="seg s{k}" style="width:{dist[k]/n*100:.1f}%" title="{SC[k]} {dist[k]}">'
            f'{SC[k]} {dist[k]}</div>' for k in (0, 1, 2))

    cells = []
    for mm in (0, 1, 2):
        for hh in (0, 1, 2):
            v = ct[(mm, hh)]
            cells.append(f'<td class="ct" data-m="{mm}" data-h="{hh}">'
                         f'<b>{v/n*100:.1f}%</b><span>{v}</span></td>')
    ctrows = ""
    for i, mm in enumerate((0, 1, 2)):
        tot = sum(ct[(mm, h)] for h in (0, 1, 2))
        ctrows += (f'<tr><th>task/role {SC[mm]}</th>' + "".join(cells[i*3:i*3+3])
                   + f'<th class="mg">{tot/n*100:.1f}%</th></tr>')
    ctrows += ('<tr><th>&Sigma;</th>'
               + "".join(f'<th class="mg">{sum(ct[(m, h)] for m in (0,1,2))/n*100:.1f}%</th>'
                         for h in (0, 1, 2)) + '<th class="mg">100%</th></tr>')

    cards_by_sample = {}
    cards = []
    for r in rows:
        cards.append(f'''<div class="card" data-m="{r['m']}" data-h="{r['h']}" data-mf="{r['mf']}" data-hf="{r['hf']}">
<div class="hd"><span class="pill s{r['m']}">task/role {SC[r['m']]}</span>
<span class="pill s{r['h']}">heatmap {SC[r['h']]}</span>
<b>{r['name']}</b> &mdash; {r['task']} <span class="dim">[{r['cat']} / {r['coord']}]</span></div>
<div class="body"><div class="imgs">
<figure><img loading="lazy" src="{r['rgb']}"><figcaption>object &mdash; 8 views</figcaption></figure>
<figure><img loading="lazy" src="{r['heat']}"><figcaption>heatmap &mdash; orange A / teal B</figcaption></figure>
</div><div class="txt">
<div class="roles">{r['roles']}</div>
<div class="j"><span class="tag s{r['m']}">task/role {r['m']}</span>{r['mr']}
{'<em>' + r['mf'] + '</em>' if r['mf'] != 'none' else ''}</div>
<div class="j"><span class="tag s{r['h']}">heatmap {r['h']}</span>{r['hr']}
{'<em>' + r['hf'] + '</em>' if r['hf'] != 'none' else ''}</div>
<div class="dim mono">{r['sid']}<br>{r['grp']}</div>
</div></div></div>''')

    faults = ("".join(f'<option value="{k}">{k} ({v})</option>' for k, v in mf.most_common()),
              "".join(f'<option value="{k}">{k} ({v})</option>' for k, v in hf.most_common()))

    def page(body, nav):
        return HEAD + nav + f'<main id="main">{body}</main>' + TAIL

    HEAD = f'''<!doctype html><meta charset="utf-8"><title>Claude gold 1000 &mdash; re-labelled</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:0;background:#fafafa;color:#222}}
header{{background:#fff;border-bottom:1px solid #ddd;padding:14px 20px}}
/* only a slim strip is pinned - a sticky block taller than the viewport hides the whole page */
.sticky{{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid #ddd;
padding:8px 20px;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
.sticky .ctrls{{margin-top:0}}
h1{{margin:0 0 10px;font-size:17px}} .dim{{color:#888}} .mono{{font-family:ui-monospace,monospace;font-size:11px}}
.bars{{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:10px}}
.barwrap{{flex:1;min-width:280px}} .bar{{display:flex;height:22px;border-radius:4px;overflow:hidden;font-size:11px}}
.seg{{display:flex;align-items:center;justify-content:center;color:#fff;white-space:nowrap}}
.s0{{background:#d9534f}} .s1{{background:#e8a33d}} .s2{{background:#4c9e6f}}
table.ct{{border-collapse:collapse;font-size:12px}} table.ct td,table.ct th{{border:1px solid #ddd;padding:5px 9px;text-align:center}}
td.ct{{cursor:pointer}} td.ct:hover{{background:#eef4ff}} td.ct span{{display:block;color:#999;font-size:10px}}
th.mg{{background:#f4f4f4;color:#666}}
.ctrls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}}
select,button{{font:13px system-ui;padding:4px 8px}}
main{{padding:16px 20px;display:flex;flex-direction:column;gap:14px}}
.card{{background:#fff;border:1px solid #e2e2e2;border-radius:7px;overflow:hidden}}
.hd{{padding:8px 12px;border-bottom:1px solid #eee;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.pill{{color:#fff;border-radius:11px;padding:2px 9px;font-size:11px}}
/* images stacked vertically at full card width so each 8-view montage is legible */
.body{{display:flex;flex-direction:column;gap:12px;padding:12px}}
.imgs{{display:flex;flex-direction:column;gap:12px;order:2}} .imgs figure{{margin:0}}
.imgs img{{width:100%;display:block;border:1px solid #eee;border-radius:4px;background:#fff}}
figcaption{{font-size:11px;color:#999;margin-top:3px}}
.txt{{order:1;display:flex;flex-direction:column;gap:8px}}
.roles{{font-size:12px;background:#f7f7f7;padding:7px 9px;border-radius:5px;line-height:1.5}}
.j{{font-size:12px}} .tag{{color:#fff;border-radius:4px;padding:1px 6px;margin-right:6px;font-size:11px}}
.j em{{display:block;color:#a33;font-style:normal;font-size:11px;margin-top:3px}}
</style>
<header><h1>Claude gold &mdash; 1000 daily_used samples, re-labelled 2026-08-05
<span class="dim">(corrected bimanual rule; heatmap extent judged against the description)</span></h1>
<div class="bars">
<div class="barwrap"><div class="dim">task / role</div><div class="bar">{bar(md,"m")}</div></div>
<div class="barwrap"><div class="dim">heatmap</div><div class="bar">{bar(hd,"h")}</div></div>
<table class="ct"><tr><th></th><th>heat bad</th><th>heat ok</th><th>heat good</th><th class="mg">&Sigma;</th></tr>{ctrows}</table>
</div></header>
<div class="sticky"><div class="ctrls">
task/role <select id="fm"><option value="">all</option><option value="0">bad</option><option value="1">ok</option><option value="2">good</option></select>
heatmap <select id="fh"><option value="">all</option><option value="0">bad</option><option value="1">ok</option><option value="2">good</option></select>
task/role fault <select id="fmf"><option value="">all</option>{faults[0]}</select>
heatmap fault <select id="fhf"><option value="">all</option>{faults[1]}</select>
<button id="reset">reset</button><span id="count" class="dim"></span>
</div>NAVSLOT</div>
'''
    TAIL = '''<script>
const $=s=>document.querySelector(s), cards=[...document.querySelectorAll('.card')];
function apply(){
  const m=$('#fm').value,h=$('#fh').value,mf=$('#fmf').value,hf=$('#fhf').value;
  let n=0;
  for(const c of cards){
    const ok=(!m||c.dataset.m===m)&&(!h||c.dataset.h===h)&&(!mf||c.dataset.mf===mf)&&(!hf||c.dataset.hf===hf);
    c.style.display=ok?'':'none'; if(ok)n++;
  }
  $('#count').textContent=n+' / '+cards.length+' on this page';
}
for(const id of ['fm','fh','fmf','fhf']) $('#'+id).addEventListener('change',apply);
$('#reset').addEventListener('click',()=>{for(const id of ['fm','fh','fmf','fhf'])$('#'+id).value='';apply();});
for(const td of document.querySelectorAll('td.ct')) td.addEventListener('click',()=>{
  $('#fm').value=td.dataset.m; $('#fh').value=td.dataset.h; $('#fmf').value=''; $('#fhf').value=''; apply();
  window.scrollTo({top:document.querySelector('header').offsetHeight,behavior:'smooth'});
});
apply();
</script>'''

    stem, ext = os.path.splitext(a.out)
    per = a.per_page or len(cards)
    pages = [cards[i:i + per] for i in range(0, len(cards), per)]
    names = [f"{os.path.basename(stem)}_p{i+1}{ext}" for i in range(len(pages))]

    def nav(cur):
        links = " ".join(f'<a href="{nm}" class="{"cur" if i == cur else ""}">{i+1}</a>'
                         for i, nm in enumerate(names))
        full = f'<a href="{os.path.basename(a.out)}">all&nbsp;{len(cards)}</a>'
        return (f'<div class="ctrls nav">page: {links} &nbsp;|&nbsp; {full}'
                f'<span class="dim">&nbsp;&nbsp;sorted worst-first (task/role, then heatmap)</span></div>')

    NAVCSS = ('<style>.nav a{padding:2px 7px;border:1px solid #ddd;border-radius:4px;'
              'text-decoration:none;color:#333;background:#fff}.nav a.cur{background:#333;color:#fff}</style>')
    for i, (chunk, nm) in enumerate(zip(pages, names)):
        open(os.path.join(os.path.dirname(a.out) or ".", nm), "w").write(
            HEAD.replace("NAVSLOT", nav(i)) + NAVCSS
            + f'<main id="main">{"".join(chunk)}</main>' + TAIL)
    open(a.out, "w").write(HEAD.replace("NAVSLOT", nav(-1)) + NAVCSS
                           + f'<main id="main">{"".join(cards)}</main>' + TAIL)
    print(f"-> {a.out}  ({n} cards, {os.path.getsize(a.out)/1024:.0f} KB)")
    print(f"-> {len(pages)} paged files {names[0]} .. {names[-1]} "
          f"({os.path.getsize(os.path.join(os.path.dirname(a.out) or '.', names[0]))/1024:.0f} KB each)")


if __name__ == "__main__":
    main()
