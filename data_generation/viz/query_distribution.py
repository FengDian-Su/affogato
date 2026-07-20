# import json
# import matplotlib.pyplot as plt
# from collections import Counter

# with open("outputs/stage1_part0_backup_0625.json", "r") as f:
#     data = json.load(f)

# query_counts = [len(obj.get("queries", [])) for obj in data]
# counter = Counter(query_counts)

# print(f"總物件數：{len(data)}")
# print(f"總 query 數：{sum(query_counts)}")
# print(f"平均 query 數：{sum(query_counts) / len(data):.2f}")
# print(f"最少：{min(query_counts)}，最多：{max(query_counts)}")
# print(f"\n分布：")
# for k in sorted(counter):
#     print(f"  {k} queries: {counter[k]} 個物件")

# fig, ax = plt.subplots(figsize=(10, 5))
# xs = sorted(counter.keys())
# ys = [counter[x] for x in xs]
# ax.bar(xs, ys, color="steelblue", edgecolor="white")
# ax.set_xlabel("Number of queries per object")
# ax.set_ylabel("Number of objects")
# ax.set_title("Query count distribution — stage1_part0_backup_0625")
# ax.set_xticks(xs)
# for x, y in zip(xs, ys):
#     ax.text(x, y + 0.3, str(y), ha="center", va="bottom", fontsize=9)

# plt.tight_layout()
# plt.savefig("viz/query_distribution.png", dpi=150)
# plt.show()
# print("圖片已存為 query_distribution.png")

# --------------------------------------------------------------------------

import json

with open("outputs/stage0_part0.kept.json", "r") as f:
    data = json.load(f)

subset = data[:100]

with open("outputs/stage0_test.json", "w") as f:
    json.dump(subset, f, indent=2, ensure_ascii=False)

print(f"完成，共寫入 {len(subset)} 筆 -> outputs/stage0_test.json")