#!/usr/bin/env python3
"""Extract R2 results from physics JSON files into a CSV."""

import csv
import json
from pathlib import Path

data_dir = Path("data")
rows = []

# Per-model files (physics_{model}_test.json)
for f in sorted(data_dir.glob("physics_*_test.json")):
    if f.name == "physics_all_test.json":
        continue
    d = json.load(open(f))
    model = d["model"]
    for size, stats in d.get("sizes", {}).items():
        row = {
            "model": model,
            "size": size,
            "n_samples": stats.get("n_samples"),
            "embedding_dim": stats.get("embedding_dim"),
            "r2_mean": stats.get("r2_mean"),
            "r2_se": stats.get("r2_se"),
        }
        for prop, val in stats.get("r2_per_property", {}).items():
            row[f"r2_{prop}"] = val
        rows.append(row)

# Also pull any models from physics_all_test.json not already covered
all_models = {r["model"] for r in rows}
d = json.load(open(data_dir / "physics_all_test.json"))
for model, sizes in d["models"].items():
    if model in all_models:
        continue
    for size, stats in sizes.items():
        row = {
            "model": model,
            "size": size,
            "n_samples": stats.get("n_samples"),
            "embedding_dim": stats.get("embedding_dim"),
            "r2_mean": stats.get("r2_mean"),
            "r2_se": stats.get("r2_se"),
        }
        for prop, val in stats.get("r2_per_property", {}).items():
            row[f"r2_{prop}"] = val
        rows.append(row)

fields = list(dict.fromkeys(k for row in rows for k in row))

out = data_dir / "physics_r2_results.csv"
with open(out, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {out}")
for r in rows:
    print(f"  {r['model']:12s} {r['size']:10s} dim={r['embedding_dim']}  r2={float(r['r2_mean']):.4f}")
