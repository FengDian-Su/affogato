#!/usr/bin/env python
"""
Emit judge-ready review inputs for stage1 records: one labeled view-montage (.jpg)
and one text block (.txt) per object, so an adversarial judge can score each role
decomposition against the actual object.

  python viz/build_review_inputs.py --in outputs/stage1_dataset.json --out review/inputs
"""
import os
import io
import json
import math
import argparse
from PIL import Image, ImageDraw, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def view_tag(view_path):
    """Short viewpoint tag from a gObjaverse view's camera json (TOP / UNDER / elev/az)."""
    d = os.path.dirname(view_path)
    idx = os.path.basename(d)
    try:
        o = json.load(open(os.path.join(d, f"{idx}.json"))).get("origin")
        r = math.sqrt(sum(t * t for t in o)) or 1.0
        elev = math.degrees(math.asin(o[2] / r))
        az = math.degrees(math.atan2(o[1], o[0])) % 360
    except Exception:
        return "view"
    if elev > 60:
        return "TOP-DOWN"
    if elev < -60:
        return "UNDERSIDE"
    return f"elev{round(elev)} az{round(az)}"


def montage(views, cols=4, cell=320):
    """Grid montage of all views with a viewpoint tag burned into each cell."""
    imgs = []
    for p in views:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell, cell))
            imgs.append((im, view_tag(p)))
        except Exception:
            pass
    if not imgs:
        return None
    rows = (len(imgs) + cols - 1) // cols
    W, H = cols * cell, rows * (cell + 18)
    g = Image.new("RGB", (W, H), (245, 245, 247))
    dr = ImageDraw.Draw(g)
    for k, (im, tag) in enumerate(imgs):
        r, c = divmod(k, cols)
        x, y = c * cell, r * (cell + 18)
        dr.rectangle([x, y, x + cell, y + 16], fill=(20, 24, 30))
        dr.text((x + 4, y + 3), f"[{k}] {tag}", fill=(220, 230, 240))
        g.paste(im, (x + (cell - im.size[0]) // 2, y + 18))
    return g


def role_line(r):
    return (f"  - Hand {r.get('id')}: VERB={r.get('role')} | TARGET={r.get('target')} | "
            f"REGION={r.get('contact_region')} | FUNCTION={r.get('function')}")


def obj_text(o):
    L = [f"OBJECT: {o.get('object_name')}   (id {str(o.get('object_id'))[:12]})",
         "COMPONENTS (the ONLY parts the model was told exist):"]
    for c in o.get("components", []):
        L.append(f"  - {c.get('name')}: {c.get('interaction','')}")
    L.append("")
    L.append("The montage image shows the actual object from labeled viewpoints "
             "(oblique ring + TOP-DOWN + UNDERSIDE). Use it to check whether each part "
             "exists and whether the contact region is reachable/graspable for the stated verb.")
    L.append("")
    for j, q in enumerate(o.get("queries", [])):
        L.append(f"QUERY {j}  [{q.get('category')}]  pattern={q.get('pattern')}  "
                 f"symmetric={q.get('symmetric')}")
        L.append(f"  TASK: {q.get('task')}")
        L.append(f"  GOAL: {q.get('goal')}")
        L.append(f"  WHY-BIMANUAL: {q.get('why_bimanual')}")
        L.append(f"  RELATION: {q.get('relation')}")
        for r in q.get("roles", []):
            L.append(role_line(r))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/stage1_dataset.json")
    ap.add_argument("--out", default="review/inputs")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    dg = os.path.dirname(here)
    inp = args.inp if os.path.isabs(args.inp) else os.path.join(dg, args.inp)
    out = args.out if os.path.isabs(args.out) else os.path.join(dg, args.out)
    os.makedirs(out, exist_ok=True)

    objs = [o for o in json.load(open(inp)) if o.get("queries")]
    manifest = []
    for i, o in enumerate(objs):
        oid = str(o.get("object_id"))[:12].replace("/", "_")
        base = f"{i:02d}_{oid}"
        with open(os.path.join(out, base + ".txt"), "w") as f:
            f.write(obj_text(o))
        g = montage(o.get("views_used", []))
        img_path = ""
        if g is not None:
            img_path = os.path.join(out, base + ".jpg")
            g.save(img_path, format="JPEG", quality=82)
        manifest.append({"i": i, "object_name": o.get("object_name"),
                         "n_queries": len(o.get("queries", [])),
                         "txt": os.path.join(out, base + ".txt"), "img": img_path})
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2, ensure_ascii=False)
    print(f"wrote {len(objs)} objects -> {out}  "
          f"({sum(m['n_queries'] for m in manifest)} queries)")


if __name__ == "__main__":
    main()
