#!/usr/bin/env python
"""
find_object_index.py — 由 object_id 反查它在 stage1 json 裡的第幾筆 (0-based index)。

pipeline/stage2_v2.py 是用 stage1 記錄的 index (--start/--end) 來切批次，
所以知道「正在跑的 object_id」對應的 index，就能知道批次跑到哪、還剩多少。
注意 stage1 分成 part0/part1 兩檔，index 是各檔內的 0-based 序號。

用法：
    # 查一個或多個 object_id (可接短前綴)
    python viz/find_object_index.py 8fd4afdf bc2fe4e8

    # 換一個 stage1 檔 (part1 的物件要用 part1 才查得到)
    python viz/find_object_index.py --stage1 outputs/stage1_v2/stage1_part1.json 8fd4afdf

    # 從 stdin 讀 (每行一個 id)，例如 ls 正在跑的輸出目錄
    ls outputs/bimanual_grounding_v2 | python viz/find_object_index.py -

    # 掃已完成的輸出目錄，回報目前批次跑到的 index 範圍 (min/max) = 進度
    python viz/find_object_index.py --progress outputs/bimanual_grounding_v2
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))            # data_generation/viz/
DATA_GEN = os.path.dirname(HERE)                             # data_generation/


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(DATA_GEN, p)


def _obj_mtime(root, object_id):
    """該 object 底下最新的 scores.npz mtime = 完成時間 (無則 None)。"""
    d = os.path.join(root, object_id)
    if not os.path.isdir(d):
        return None
    best = None
    for t in os.listdir(d):
        sp = os.path.join(d, t, "scores.npz")
        if os.path.exists(sp):
            m = os.path.getmtime(sp)
            best = m if best is None or m > best else best
    return best


def load_index(stage1_path):
    """回傳 (records, id2idx)。id2idx: 完整 object_id -> index。"""
    records = json.load(open(stage1_path))
    id2idx = {r.get("object_id"): i for i, r in enumerate(records)}
    return records, id2idx


def lookup(query, records, id2idx):
    """精確比對優先；否則當成前綴比對。回傳符合的 index 清單。"""
    if query in id2idx:                      # 完整 id 命中
        return [id2idx[query]]
    return [i for i, r in enumerate(records)  # 前綴比對 (貼短 id 時方便)
            if str(r.get("object_id", "")).startswith(query)]


def fmt(i, records):
    r = records[i]
    nq = len(r.get("queries", []))
    return (f"  [{i:4d}/{len(records)}]  {r.get('object_id')}  "
            f"{r.get('object_name', '?'):<28}  ({nq} tasks)")


def main():
    ap = argparse.ArgumentParser(description="由 object_id 反查 stage1 json 的 index")
    ap.add_argument("ids", nargs="*",
                    help="一個或多個 object_id (可短前綴)；用 '-' 從 stdin 逐行讀")
    ap.add_argument("--stage1", default="outputs/stage1_v2/stage1_part0.json",
                    help="stage1 json (預設 part0；part1 的物件要另外指定 --stage1)")
    ap.add_argument("--progress", metavar="DIR",
                    help="掃這個輸出目錄下的 object 子資料夾，回報 index 範圍 (跑到哪)")
    ap.add_argument("--shards", metavar="A-B,C-D,...",
                    help="搭配 --progress：用 index 區間切 shard，逐段回報 done/frontier/速率/ETA "
                         "(例: 0-619,619-1238,1238-1858)")
    args = ap.parse_args()

    stage1 = _resolve(args.stage1)
    records, id2idx = load_index(stage1)
    print(f"stage1: {stage1}  ({len(records)} records)\n")

    # --- 進度模式：掃輸出目錄 ---
    if args.progress:
        root = _resolve(args.progress)
        done_ids = [d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d))]
        idxs = sorted(id2idx[o] for o in done_ids if o in id2idx)
        unknown = [o for o in done_ids if o not in id2idx]

        # --- 逐 shard 回報 (需要 mtime 算速率/ETA) ---
        if args.shards:
            idx2id = {i: r["object_id"] for i, r in enumerate(records)}
            print(f"{'shard':<8}{'range':<14}{'done':>6}{'total':>7}{'pct':>7}"
                  f"{'frontier':>10}{'min/obj':>9}   ETA")
            for k, seg in enumerate(args.shards.split(",")):
                a, b = (int(x) for x in seg.split("-"))
                inseg = [i for i in idxs if a <= i < b]
                tot = b - a
                if not inseg:
                    print(f"p{k+1:<7}{seg:<14}{0:>6}{tot:>7}{0.0:>6.1f}%{'—':>10}{'—':>9}   未開始")
                    continue
                front = max(inseg)
                mts = sorted(_obj_mtime(root, idx2id[i]) for i in inseg
                             if _obj_mtime(root, idx2id[i]))
                win = mts[-20:]                       # 只看最近 ~20 筆 = 當前吞吐 (避免舊閒置拉歪)
                rate = ((win[-1] - win[0]) / 60 / (len(win) - 1)) if len(win) > 1 else None
                remain = tot - len(inseg)
                eta = (f"{remain * rate / 60:.1f}h ({remain} 筆)"
                       if rate else f"{remain} 筆")
                rate_s = f"{rate:.1f}" if rate else "—"
                print(f"p{k+1:<7}{seg:<14}{len(inseg):>6}{tot:>7}"
                      f"{100*len(inseg)/tot:>6.1f}%{front:>10}{rate_s:>9}   {eta}")
            print(f"\n總完成 {len(idxs)} 個" +
                  (f"；⚠ {len(unknown)} 個不在此 stage1" if unknown else ""))
            return

        if not idxs:
            print(f"[progress] {root}: 沒有任何 object 對得上這個 stage1")
        else:
            print(f"[progress] {root}: 有輸出的 object {len(idxs)} 個，"
                  f"index 範圍 {idxs[0]}..{idxs[-1]}")
            covered = set(idxs)
            gaps = [i for i in range(idxs[0], idxs[-1] + 1) if i not in covered]
            print(f"           連續區間內缺 {len(gaps)} 筆" +
                  (f"（前幾筆: {gaps[:10]}）" if gaps else "（無空洞）"))
        if unknown:
            print(f"           ⚠ {len(unknown)} 個輸出目錄不在此 stage1（檔案版本不同？）: {unknown[:5]}")
        return

    # --- 從 stdin 讀 ---
    queries = list(args.ids)
    if not queries or queries == ["-"]:
        queries = [ln.strip() for ln in sys.stdin if ln.strip()]

    for q in queries:
        hits = lookup(q, records, id2idx)
        if not hits:
            print(f"'{q}': 查無此 object_id")
        elif len(hits) == 1:
            print(f"'{q}' ->")
            print(fmt(hits[0], records))
        else:
            print(f"'{q}' -> {len(hits)} 筆符合（前綴）:")
            for i in hits:
                print(fmt(i, records))


if __name__ == "__main__":
    main()
