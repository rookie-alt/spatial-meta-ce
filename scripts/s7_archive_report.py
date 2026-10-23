"""Rebuild descriptive S7 overview from archived JSON; no cache/model access."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

FILES = (
    "status.json",
    "summary.json",
    "protocol.json",
    "verification.json",
    "legalacts.json",
    "generalization.json",
    "sample-plan.json",
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def overview(rows, train):
    keys = [(r["database"], r["cohort"]) for r in rows]
    expected = {(db, c) for db in train for c in ("seen", "unseen")}
    expected |= {
        (db, "source_validation_full") for db in ("chembl", "legalacts", "telstra")
    }
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Missing, extra or duplicate database/cohort reports")
    lookup = {
        (r["database"], r["cohort"]): {s["bucket"]: s for s in r["statistics"]}
        for r in rows
    }
    for row in rows:
        cells = row["statistics"]
        names = [c["bucket"] for c in cells]
        if len(names) != len(set(names)) or "all" not in names:
            raise ValueError("Missing or duplicate statistics bucket")
        entry = lookup[(row["database"], row["cohort"])]
        if "four_plus" not in entry:
            if cells != [{"bucket": "all", "n": 0, "status": "below_min_cell_n"}]:
                raise ValueError("Missing four_plus statistics for nonempty cohort")
            entry["four_plus"] = {
                "bucket": "four_plus",
                "n": 0,
                "status": "below_min_cell_n",
            }
    result = []
    for bucket in ("all", "four_plus"):
        for arm in ("A", "B"):
            ratios = [
                lookup[(db, "unseen")][bucket][arm]["q95"]
                / lookup[(db, "seen")][bucket][arm]["q95"]
                for db in train
                if all(
                    lookup[(db, c)][bucket]["status"] == "descriptive"
                    for c in ("seen", "unseen")
                )
            ]
            result.append(
                {
                    "bucket": bucket,
                    "arm": arm,
                    "database_count": len(ratios),
                    "median_within_database_q95_unseen_div_seen": statistics.median(
                        ratios
                    )
                    if ratios
                    else None,
                    **({"status": "no_eligible_databases"} if not ratios else {}),
                }
            )
    changes = []
    for bucket in ("all", "four_plus"):
        for cohort in ("seen", "unseen"):
            values = [
                lookup[(db, cohort)][bucket]["q95_relative_change_b_vs_a"]
                for db in train
                if lookup[(db, cohort)][bucket]["status"] == "descriptive"
            ]
            changes.append(
                {
                    "bucket": bucket,
                    "cohort": cohort,
                    "database_count": len(values),
                    "improved_databases": sum(x < 0 for x in values),
                    "worse_databases": sum(x > 0 for x in values),
                    "median_relative_q95_change_b_vs_a": statistics.median(values)
                    if values
                    else None,
                    **({"status": "no_eligible_databases"} if not values else {}),
                }
            )
    return {
        "purpose": "posthoc-descriptive-ratios-not-formal-endpoint-or-selection",
        "within_database_ratios": result,
        "b_vs_a_direction_counts": changes,
        "warning": "Sample sizes/composition vary; ratios are descriptive, not tests of significance or causality.",
    }


def main(root):
    output = root / "实验文档/results/s7-fit-generalization-20261007"
    reports = root / "reports/s7-fit-generalization-20261007"
    derived = [
        output / "overview.json",
        output / "database-matrix.csv",
        reports / "archive-verification.json",
    ]
    for path in derived:
        if path.is_symlink() or (
            path.exists() and (not path.is_file() or path.stat().st_nlink != 1)
        ):
            raise ValueError(f"Derived output symlink/hardlink alias forbidden: {path}")
    hashes = load(reports / "server-archive-sha256.json")
    for name in FILES:
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"Archive server hash mismatch: {name}")
    status, summary, proof = (
        load(output / name)
        for name in ("status.json", "summary.json", "verification.json")
    )
    if (
        status["status"] != "completed"
        or summary["status"] != "completed"
        or summary["training"]
        or summary["target_data_read"]
        or summary["adaptation"]
        or not summary["independent_verification_passed"]
        or not proof["passed"]
        or summary["s6_decision"] != "no_candidate"
    ):
        raise ValueError("S7 completion/scope/verification mismatch")
    for name in FILES[1:]:
        if status["output_sha256"][name] != hashes[name]:
            raise ValueError(
                "Parent output binding differs from independently archived file"
            )
    config = load(root / "configs/data.json")
    gen, plan = (
        load(output / name) for name in ("generalization.json", "sample-plan.json")
    )
    keys = [(r["database"], r["cohort"]) for r in plan["cohorts"]]
    expected = {(db, c) for db in config["splits"]["train"] for c in ("seen", "unseen")}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete sample cohort plan")
    if sum(r["sample_unique"] for r in plan["cohorts"]) != summary["inference_queries"]:
        raise ValueError("Sample query budget mismatch")
    result = overview(gen["rows"], config["splits"]["train"])
    result["source_sha256"] = hashes
    result["seen_pool_unique"] = sum(
        r["pool_unique"] for r in plan["cohorts"] if r["cohort"] == "seen"
    )
    result["unseen_pool_unique"] = sum(
        r["pool_unique"] for r in plan["cohorts"] if r["cohort"] == "unseen"
    )
    result["inference_queries"] = summary["inference_queries"]
    (output / "overview.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    columns = [
        "database",
        "cohort",
        "bucket",
        "n",
        "status",
        "a_q95",
        "b_q95",
        "pg_q95",
        "b_vs_a_relative_change",
    ]
    with (output / "database-matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in gen["rows"]:
            for cell in row["statistics"]:
                if cell["bucket"] not in ("all", "four_plus"):
                    continue
                writer.writerow(
                    {
                        "database": row["database"],
                        "cohort": row["cohort"],
                        "bucket": cell["bucket"],
                        "n": cell["n"],
                        "status": cell["status"],
                        "a_q95": cell.get("A", {}).get("q95"),
                        "b_q95": cell.get("B", {}).get("q95"),
                        "pg_q95": cell.get("PG", {}).get("q95"),
                        "b_vs_a_relative_change": cell.get(
                            "q95_relative_change_b_vs_a"
                        ),
                    }
                )
    for name in FILES:
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"Raw evidence changed during archive rebuild: {name}")
    (reports / "archive-verification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "raw_server_byte_sha256": hashes,
                "source_cohorts_complete": True,
                "statistics_cohorts_complete": True,
                "derived_sha256": {
                    n: hashlib.sha256((output / n).read_bytes()).hexdigest()
                    for n in ("overview.json", "database-matrix.csv")
                },
                "large_predictions_archived_locally": False,
                "checkpoint_downloaded": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    main(parser.parse_args().root.resolve())
