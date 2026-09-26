import json
from pathlib import Path

# PRICE paper Table 3: Q-ERROR percentiles over all rows (50% and 95%)
paper = {
    "stats":    {"pg": (1.87, 73.35), "price_pre": (2.48, 25.84), "price_ft": (1.41, 7.07)},
    "genome":   {"pg": (1.17, 383.09), "price_pre": (1.65, 15.67), "price_ft": (1.41, 4.27)},
    "ergastf1": {"pg": (1.66, 27.47), "price_pre": (1.52, 14.37), "price_ft": (1.49, 7.22)},
}

runs = {
    "PG": "runs/baseline-pg-20260926/summary.json",
    "FOMAML s42": "runs/fomaml-s42-paired-20260926/summary.json",
    "FOMAML s43": "runs/fomaml-s43-paired-20260926/summary.json",
    "ERM s42": "runs/erm-s42-paired-20260926/summary.json",
    "ERM s43": "runs/erm-s43-paired-20260926/summary.json",
}

print(f"{'method':13s} {'db':10s} {'K':>4s} {'q50':>7s} {'q95_all':>8s} {'q95_uniq':>9s}")
for name, path in runs.items():
    r = json.loads(Path(path).read_text())
    for row in r["aggregate"]:
        db, k = row["database"], row["k"]
        if k in (0, 5, 20, 100):
            a = row["across_support_runs"]
            print(f"{name:13s} {db:10s} {k:4d} {a['all_rows_q50']['mean']:7.3f} "
                  f"{a['all_rows_q95']['mean']:8.3f} {a['unique_query_q95']['mean']:9.3f}")

print("\n=== PRICE paper Table 3 (Q-ERROR, all rows) ===")
for db, d in paper.items():
    print(f"{db:10s} PG {d['pg']} | PRICE-pretrained {d['price_pre']} | PRICE-finetuned {d['price_ft']}")
