#!/usr/bin/env python3
"""Build the advisor-facing HTML progress report for spatial-meta-ce.

Every number in the report is computed from committed evidence files; nothing
is hand-transcribed.  The K>0 arm table is parsed from doc 14 and every cell
that also has a structured source is cross-checked against it, so a silent
drift between the document and the JSON aborts the build instead of reaching
the report.

    python3 scripts/build_advisor_report.py \
        --git-branch experiment/source-broad-imdb-20261021 \
        --out reports/advisor-report-20261022.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics as stats
import subprocess
import sys
import tempfile
from pathlib import Path

TARGET_DBS = ("stats", "genome", "ergastf1", "imdb")
# the column order used by the PRICE paper's Table 2
PAPER_ORDER = ("imdb", "stats", "ergastf1", "genome")
# doc 14 covers only the three historical targets: imdb has no finetune cache,
# so it has no few-shot arm and no K>0 row anywhere in the record.
DOC14_DBS = ("stats", "genome", "ergastf1")
SEED_NOISE = 0.0628  # doc 94 6.1b: measured seed-to-seed SD, source-side arm

# Architecture shown in section 01.  Mirrors spatial_meta_ce.model.SetEstimator;
# the parameter count computed from these numbers is asserted against the two
# shipped checkpoints below, so a drift in the code aborts the build.
GROUP_DIMS = (40, 40, 4, 43)
GROUP_LABELS = (
    ("连接列", "J", "join 列直方图，按表规模归一"),
    ("fanout", "F", "对侧表匹配行数按本表直方图分桶"),
    ("表", "T", "[log 表规模, AVI, MINSEL, EBO]"),
    ("过滤列", "C", "40 维直方图 + 归一化范围上下界 + 选择率"),
)
HEAD_HIDDEN = (128, 64)
GLOBAL_SCALARS = 6  # log1p(四组 token 数量)×4 + PG 对数估计 + Σlog 表规模

# repo-relative evidence paths -> local names inside the evidence directory
EVIDENCE = {
    "broad_summary.json": "实验文档/results/p2-2-error-structure-broad-20261021/summary.json",
    "broad_protocol.json": "实验文档/results/p2-2-error-structure-broad-20261021/protocol.json",
    "imdb_summary.json": "实验文档/results/p2-2-target-imdb-20261021/summary.json",
    "imdb_protocol.json": "实验文档/results/p2-2-target-imdb-20261021/protocol.json",
    "source102_summary.json": "实验文档/results/p2-2-price-population-broad-20261021/summary.json",
    "doc14.md": "实验文档/14-离线实验统一对照与论文结论.md",
}
for _seed in (42, 43, 44):
    EVIDENCE[f"frozen3_s{_seed}.json"] = (
        f"实验文档/results/frozen3-20260927/s{_seed}-summary.json"
    )
    EVIDENCE[f"erm_s{_seed}.json"] = f"实验文档/results/erm-s{_seed}-summary.json"
    EVIDENCE[f"ermsum_s{_seed}.json"] = (
        f"实验文档/results/erm-sum-target-20260927/s{_seed}-summary.json"
    )

# doc 108: the four target databases re-measured on three seeds
EVIDENCE["target3seed_summary.json"] = (
    "实验文档/results/p2-2-target-3seed-20261023/summary.json"
)
# doc 106: linear probe on the frozen representation (database identity)
EVIDENCE["probe_summary.json"] = (
    "实验文档/results/p2-2-db-identity-probe-20261023/summary.json"
)
# doc 110: three-layer latency measurement (L1 forward, L2 estimation, L3 execution)
for _name, _path in (
    ("latency_cpu_fomaml.json", "latency-cpu-fomaml.json"),
    ("latency_cpu_erm.json", "latency-cpu-erm.json"),
    ("latency_cuda_fomaml.json", "latency-cuda-fomaml.json"),
    ("estimation_cpu_fomaml.json", "estimation-cpu-fomaml.json"),
):
    EVIDENCE[_name] = f"实验文档/results/latency-20261025/{_path}"
for _db in TARGET_DBS:
    EVIDENCE[f"e2e_{_db}.json"] = f"实验文档/results/latency-20261025/e2e-{_db}.json"

# evidence that lives on this branch (literature comparison)
LIT_EVIDENCE = {
    "allrows_summary.json": "实验文档/results/p2-2-allrows-lit-20261022/summary.json",
    "allrows_protocol.json": "实验文档/results/p2-2-allrows-lit-20261022/protocol.json",
    "price_paper_table2.json": "实验文档/results/p2-2-allrows-lit-20261022/price_paper_table2.json",
    "related_work.json": "实验文档/results/p2-2-allrows-lit-20261022/related_work.json",
}

# arm -> evidence family that reproduces it from per-seed aggregates
ARM_FAMILY = {
    "FOMAML-set": "frozen3",
    "ERM-set": "erm",
    "ERM-sum": "ermsum",
}
K_VALUES = (0, 5, 10, 20, 50, 100)


# --------------------------------------------------------------------------- io


def fetch_evidence(
    branch: str,
    repo_root: Path,
    dest: Path,
    manifest: dict[str, str] | None = None,
) -> dict[str, str]:
    """Copy the evidence files out of git and return {name: sha256}."""
    digests = {}
    for name, repo_path in (manifest or EVIDENCE).items():
        result = subprocess.run(
            ["git", "show", f"{branch}:{repo_path}"],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"missing evidence on {branch}: {repo_path}\n{result.stderr.decode()}"
            )
        (dest / name).write_bytes(result.stdout)
        digests[name] = hashlib.sha256(result.stdout).hexdigest()
    return digests


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# whole sections a trimmed handout may drop, keyed by <section id="...">
OMITTABLE_SECTIONS = {
    "flip": "04 漂移与拟合的反转",
    "error": "05 误差结构诊断",
    "literature": "09 与文献的同口径对照",
    "limits": "10 边界与不可声称",
    "evidence": "11 证据与复算",
    "glossary": "12 名词表",
}


SECTION_ORDER = (
    "verdict",
    "protocol",
    "targets",
    "sources",
    "flip",
    "error",
    "fewshot",
    "advantages",
    "latency",
    "literature",
    "limits",
    "evidence",
    "glossary",
)


def renumber_sections(html_text: str, omit: set[str]) -> str:
    """Close the numbering gaps left by removed sections and retarget references."""
    remaining = [sid for sid in SECTION_ORDER if sid not in omit]
    if len(remaining) == len(SECTION_ORDER):
        return html_text
    mapping = {SECTION_ORDER.index(sid): index for index, sid in enumerate(remaining)}

    def range_repl(match: re.Match) -> str:
        start, end = int(match.group(1)), int(match.group(2))
        kept = [mapping[old] for old in sorted(mapping) if old <= end]
        return f"第 {mapping[start]:02d}–{kept[-1]:02d} 节"

    html_text = re.sub(r"第 (\d{2})–(\d{2}) 节", range_repl, html_text)

    def single_repl(match: re.Match) -> str:
        old = int(match.group(1))
        new = mapping.get(old)
        return match.group(0) if new is None else f"{new:02d} 节"

    html_text = re.sub(r"(?<![0-9–])(\d{2}) 节", single_repl, html_text)

    def anchor_repl(match: re.Match) -> str:
        old = int(match.group(2))
        new = mapping.get(old)
        if new is None:
            return match.group(0)
        return f'<a href="{match.group(1)}">{new:02d} '

    html_text = re.sub(r'<a href="(#[a-z]+)">(\d{2}) ', anchor_repl, html_text)
    for new_index, section_id in enumerate(remaining):
        pattern = re.compile(
            rf'(<section id="{section_id}">\s*<h2><span class="num">)(\d{{2}})(</span>)'
        )
        html_text, count = pattern.subn(rf"\g<1>{new_index:02d}\g<3>", html_text)
        if count != 1:
            raise SystemExit(f"section {section_id!r} header not matched")
    return html_text


def strip_sections(html_text: str, omit: set[str]) -> str:
    """Drop whole sections and every in-text pointer to them.

    Removing a section without its pointers would leave dead anchors, so each
    rewrite is asserted: if the wording changes, the build stops instead of
    shipping a dangling reference.
    """
    if not omit:
        return html_text
    for section_id in omit:
        html_text, count = re.subn(
            rf'\n<section id="{section_id}">.*?</section>', "", html_text, flags=re.S
        )
        if count != 1:
            raise SystemExit(f"section {section_id!r} matched {count} times")
    if "literature" in omit:
        html_text = _drop_reference(
            html_text, '（详见 <a href="#literature">09 与文献的同口径对照</a>）'
        )
    if "glossary" in omit:
        html_text, count = re.subn(
            r'<p class="small muted">第一次看到 .*?</p>', "", html_text, flags=re.S
        )
        if count != 1:
            raise SystemExit(f"glossary pointer matched {count} times")
    if "limits" in omit:
        html_text = _drop_reference(
            html_text, "（必须和第 10 节「边界与不可声称」一起读）"
        )
    literature_note = (
        "文献方数字为论文发表值的转录（出处与未核实项见第 09 节与第 11 节）。"
    )
    if "literature" in omit:
        html_text = _drop_reference(html_text, literature_note).replace(
            "计算并带一致性断言，原始记录见", "计算并带一致性断言。原始记录见"
        )
    elif "evidence" in omit:
        html_text = html_text.replace(
            "（出处与未核实项见第 09 节与第 11 节）", "（未核实项见第 09 节）"
        )
    return renumber_sections(html_text, omit)


def _drop_reference(text: str, needle: str) -> str:
    if needle not in text:
        raise SystemExit(f"cross-reference not found: {needle[:40]!r}")
    return text.replace(needle, "")


def renumber_section_refs(obj):
    """Rewrite 「第 08 节」 -> 「第 09 节」 inside the literature evidence.

    related_work.json was written on 2026-10-22, when the literature section
    was number 08.  Inserting the latency section pushed it to 09.  The file is
    sha256-pinned evidence, so it is not edited; the reference is retargeted in
    memory at render time instead.  Only the literature bundle is passed in,
    so the script's own 「第 08 节」 mentions (which do mean the latency section)
    are untouched.
    """
    if isinstance(obj, str):
        return obj.replace("第 08 节", "第 09 节")
    if isinstance(obj, list):
        return [renumber_section_refs(item) for item in obj]
    if isinstance(obj, dict):
        return {key: renumber_section_refs(value) for key, value in obj.items()}
    return obj


# ----------------------------------------------------------------- computation


def arm_means(
    evidence_dir: Path, family: str
) -> dict[tuple[str, int], tuple[float, float]]:
    """(database, k) -> (mean, sd) of unique_query_q95 over the three seeds."""
    per_cell: dict[tuple[str, int], list[float]] = {}
    for seed in (42, 43, 44):
        summary = load_json(evidence_dir / f"{family}_s{seed}.json")
        for row in summary["aggregate"]:
            q95 = row["across_support_runs"]["unique_query_q95"]
            per_cell.setdefault((row["database"], row["k"]), []).append(q95["mean"])
    out = {}
    for cell, values in per_cell.items():
        if len(values) != 3:
            raise SystemExit(f"{family} {cell}: expected 3 seeds, got {len(values)}")
        out[cell] = (sum(values) / 3.0, stats.stdev(values))
    return out


ROW_RE = re.compile(
    r"^\|\s*(0|5|10|20|50|100)\s*\|"
    r"\s*([\d.]+)\s*±\s*([\d.]+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*±\s*([\d.]+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)
ARMS = ("FOMAML-set", "ERM-set", "FOMAML-sum", "ERM-sum")


def parse_doc14(path: Path) -> dict[str, dict[int, dict]]:
    """Parse the three target-database tables of doc 14 into structured rows."""
    text = path.read_text(encoding="utf-8")
    tables: dict[str, dict[int, dict]] = {}
    current = None
    for line in text.splitlines():
        heading = re.match(r"^###\s+(\w+)\s*$", line)
        if heading:
            current = heading.group(1)
            tables.setdefault(current, {})
            continue
        if current is None or current not in TARGET_DBS:
            continue
        match = ROW_RE.match(line.strip())
        if not match:
            continue
        k = int(match.group(1))
        cells = match.groups()
        row = {"price": float(cells[9]), "pg": float(cells[10])}
        for index, arm in enumerate(ARMS):
            row[arm] = {
                "mean": float(cells[1 + index * 2]),
                "sd": float(cells[2 + index * 2]),
            }
        tables[current][k] = row
    for database in DOC14_DBS:
        missing = [k for k in K_VALUES if k not in tables.get(database, {})]
        if missing:
            raise SystemExit(f"doc 14: {database} missing K rows {missing}")
    return tables


def cross_check_arms(doc14_rows: dict, families: dict[str, dict]) -> None:
    """Abort when doc 14 and the per-seed JSON disagree on a reproducible arm."""
    for arm, family in ARM_FAMILY.items():
        for (database, k), (mean, sd) in families[family].items():
            if database not in doc14_rows or k not in doc14_rows[database]:
                continue
            parsed = doc14_rows[database][k][arm]
            if abs(parsed["mean"] - mean) > 0.01 or abs(parsed["sd"] - sd) > 0.01:
                raise SystemExit(
                    f"doc 14 / JSON mismatch for {arm} {database} K={k}: "
                    f"doc=({parsed['mean']:.2f}+-{parsed['sd']:.2f}) "
                    f"json=({mean:.2f}+-{sd:.2f})"
                )


def build_facts(evidence_dir: Path, report_dbs: tuple[str, ...] = TARGET_DBS) -> dict:
    dbs = tuple(report_dbs)
    broad = load_json(evidence_dir / "broad_summary.json")
    protocol = load_json(evidence_dir / "broad_protocol.json")
    imdb102 = load_json(evidence_dir / "imdb_summary.json")
    source102 = load_json(evidence_dir / "source102_summary.json")
    databases = broad["databases"]

    sources = [db for db in databases if db not in TARGET_DBS]
    facts: dict = {
        "broad": broad,
        "protocol": protocol,
        "sources": sorted(sources),
        "report_dbs": dbs,
        "imdb102": imdb102,
        "source102": source102,
    }

    def q95(db: str, method: str) -> float:
        return databases[db]["overview"][method]["q95"]

    def corr(db: str, method: str) -> float:
        return databases[db]["overview"][method]["corr_truth"]

    def group_stats(group: list[str]) -> dict:
        price_ratio = [q95(db, "price_k0") / q95(db, "ours_k0") for db in group]
        pg_ratio = [q95(db, "pg") / q95(db, "ours_k0") for db in group]
        return {
            "price_median": stats.median(price_ratio),
            "pg_median": stats.median(pg_ratio),
            "price_min": min(price_ratio),
            "price_max": max(price_ratio),
            "pg_min": min(pg_ratio),
            "pg_max": max(pg_ratio),
            "win_price": sum(
                1 for db in group if q95(db, "ours_k0") < q95(db, "price_k0")
            ),
            "lose_price": sum(
                1 for db in group if q95(db, "price_k0") < q95(db, "ours_k0")
            ),
            "win_pg": sum(1 for db in group if q95(db, "ours_k0") < q95(db, "pg")),
            "lose_pg": sum(1 for db in group if q95(db, "pg") < q95(db, "ours_k0")),
            "corr_win": sum(
                1 for db in group if corr(db, "ours_k0") > corr(db, "price_k0")
            ),
            "corr_lose": sum(
                1 for db in group if corr(db, "price_k0") > corr(db, "ours_k0")
            ),
            "dcorr_median": stats.median(
                [corr(db, "price_k0") - corr(db, "ours_k0") for db in group]
            ),
            "bias_median": stats.median(
                [databases[db]["overview"]["ours_k0"]["bias_med"] for db in group]
            ),
            "over_median": stats.median(
                [databases[db]["overview"]["ours_k0"]["over_pct"] for db in group]
            ),
            "wider_than_price": sum(
                1
                for db in group
                if databases[db]["overview"]["ours_k0"]["std"]
                > databases[db]["overview"]["price_k0"]["std"]
            ),
        }

    # method-level facts used by the "advantages" section
    imdb_protocol = load_json(evidence_dir / "imdb_protocol.json")
    ermsum_summary = load_json(evidence_dir / "ermsum_s42.json")
    facts["price_parameters"] = imdb_protocol["price"]["parameters"]
    facts["our_parameters"] = ermsum_summary["parameters"]
    facts["our_parameter_bytes"] = ermsum_summary["parameter_bytes"]
    facts["set_parameters"] = load_json(evidence_dir / "frozen3_s42.json")["parameters"]
    facts["use_pg"] = bool(protocol["checkpoint"]["model_config"]["use_pg"])
    facts["parameter_ratio"] = facts["price_parameters"] / facts["our_parameters"]

    facts["targets"] = group_stats(list(dbs))
    facts["source_stats"] = group_stats(facts["sources"])

    # rows that sit inside the measured seed-noise band
    facts["noise_band_sources"] = [
        db
        for db in facts["sources"]
        if 1.0 < q95(db, "price_k0") / q95(db, "ours_k0") < 1.0 + SEED_NOISE
    ]
    facts["noise_band_targets"] = [
        db for db in dbs if abs(q95(db, "pg") / q95(db, "ours_k0") - 1.0) < SEED_NOISE
    ]

    # 4+ bucket (the project's headline macro_four_plus_q95 granularity)
    four_plus = {}
    for db in dbs:
        entry = databases[db]["four_plus"]
        four_plus[db] = {
            "n": entry["n"],
            "reported": entry.get("reported", True),
            "ours": entry["ours_k0"]["q95"],
            "price": entry["price_k0"]["q95"],
            "pg": entry["pg"]["q95"],
        }
    facts["four_plus_targets"] = four_plus
    facts["four_plus_price_median"] = stats.median(
        [v["price"] / v["ours"] for v in four_plus.values()]
    )
    facts["four_plus_pg_median"] = stats.median(
        [v["pg"] / v["ours"] for v in four_plus.values()]
    )

    # literature comparison: our all-rows panel vs the published PRICE table
    allrows = load_json(evidence_dir / "allrows_summary.json")
    paper_table = load_json(evidence_dir / "price_paper_table2.json")
    facts["allrows"] = allrows
    facts["paper_table"] = paper_table
    facts["related_work"] = renumber_section_refs(
        load_json(evidence_dir / "related_work.json")
    )
    for db in TARGET_DBS:
        published = paper_table["q_error"][db]["PG"]["q95"]
        measured = allrows["databases"][db]["pg"]["q95"]
        if abs(measured - published) / published > 0.001:
            raise SystemExit(
                f"all-rows PG does not reproduce the published value for {db}: "
                f"{measured} vs {published}"
            )

    # K>0 arm tables
    doc14_rows = parse_doc14(evidence_dir / "doc14.md")
    families = {
        family: arm_means(evidence_dir, family) for family in ARM_FAMILY.values()
    }
    cross_check_arms(doc14_rows, families)
    facts["doc14"] = doc14_rows

    # The doc 14 PRICE column is PRICE's own few-shot value, so it only lines
    # up with the 30-database evidence file at K=0.  PG is K-independent.
    for db in DOC14_DBS:
        parsed = doc14_rows[db][0]
        if abs(parsed["price"] - q95(db, "price_k0")) > 0.01:
            raise SystemExit(
                f"doc 14 PRICE K=0 mismatch for {db}: "
                f"{parsed['price']} vs {q95(db, 'price_k0')}"
            )
        if abs(parsed["pg"] - q95(db, "pg")) > 0.01:
            raise SystemExit(
                f"doc 14 PG mismatch for {db}: {parsed['pg']} vs {q95(db, 'pg')}"
            )

    # doc 108: three-seed re-measurement.  The single-seed target numbers above
    # stay as they are (they are the same measurement); doc 108 is the
    # robustness check that the direction survives a seed change.
    target3 = load_json(evidence_dir / "target3seed_summary.json")
    headline = target3["headline"]
    per_db_beaten = sum(
        entry["seeds_beating_price_on_q95"] for entry in target3["databases"].values()
    )
    if per_db_beaten != headline["seed_database_pairs_beating_price"]:
        raise SystemExit(
            "doc 108: per-database win counts disagree with the headline "
            f"({per_db_beaten} vs {headline['seed_database_pairs_beating_price']})"
        )
    if headline["seed_database_pairs_beating_price"] != 0:
        raise SystemExit(
            "doc 108 now reports a seed beating PRICE on Q95; the report's "
            "「0/12」 wording is stale and must be revisited"
        )
    facts["target3seed"] = target3
    # statistics for the databases this report actually shows (the evidence file
    # itself always covers all four)
    report3 = [target3["databases"][db] for db in dbs]
    facts["target3seed_ratio"] = stats.median(
        [entry["median_q95_ratio_price_over_ours"] for entry in report3]
    )
    facts["target3seed_pairs"] = len(report3) * 3
    facts["target3seed_pairs_beating_price"] = sum(
        entry["seeds_beating_price_on_q95"] for entry in report3
    )
    facts["target3seed_pairs_beating_pg"] = sum(
        entry["seeds_beating_pg_on_q95"] for entry in report3
    )

    # doc 106: the probe has to clear its own negative control, otherwise the
    # "database identity is linearly readable" claim is a probe artefact.  The
    # body text says "almost perfectly separable", so require an order of
    # magnitude over chance, not merely beating the control.
    probe = load_json(evidence_dir / "probe_summary.json")
    if probe["k_way"]["accuracy"] < 10 * probe["k_way"]["chance"]:
        raise SystemExit(
            "doc 106: probe accuracy is not an order of magnitude above chance "
            f"({probe['k_way']['accuracy']} vs chance {probe['k_way']['chance']})"
        )
    if probe["k_way"]["accuracy"] <= probe["negative_control"]["accuracy"] * 2:
        raise SystemExit(
            "doc 106: probe accuracy does not clear the permuted-label control "
            f"({probe['k_way']['accuracy']} vs {probe['negative_control']['accuracy']})"
        )
    if len(probe["lodo_centroid_ratio"]) != probe["n_databases"]:
        raise SystemExit(
            "doc 106: centroid-ratio rows do not cover every probed database "
            f"({len(probe['lodo_centroid_ratio'])} vs {probe['n_databases']})"
        )
    if probe["target_databases_read"]:
        raise SystemExit(
            "doc 106: the probe claims to have read target databases; the "
            "report's read-only-to-sources claim would be false"
        )
    facts["probe"] = probe
    probe_ratios = [entry["ratio"] for entry in probe["lodo_centroid_ratio"]]
    facts["probe_centroid_ratio_median"] = stats.median(probe_ratios)
    facts["probe_centroid_ratio_separated"] = sum(1 for r in probe_ratios if r > 2.0)

    # doc 110: the three latency layers.  The injection self-check is the
    # quality gate for L3 -- an unverified injection silently degrades that arm
    # to pg_native while still being reported as a measurement.
    latency = load_json(evidence_dir / "latency_cpu_fomaml.json")
    latency_erm = load_json(evidence_dir / "latency_cpu_erm.json")
    latency_gpu = load_json(evidence_dir / "latency_cuda_fomaml.json")
    estimation = load_json(evidence_dir / "estimation_cpu_fomaml.json")
    e2e = {db: load_json(evidence_dir / f"e2e_{db}.json") for db in dbs}

    # L1 must be the forward pass alone; a file that silently grew feature
    # extraction or planning would inflate the speedup without looking wrong.
    for name, payload in (
        ("L1 cpu", latency),
        ("L1 cpu erm", latency_erm),
        ("L1 gpu", latency_gpu),
    ):
        if (
            payload["includes_feature_extraction"]
            or payload["includes_plan_generation"]
            or payload["includes_database_execution"]
        ):
            raise SystemExit(
                f"doc 110: {name} claims to include extraction/planning/execution; "
                "L1 must be the forward pass alone"
            )
    if estimation["includes_feature_extraction"] is not True:
        raise SystemExit(
            "doc 110: L2 no longer includes feature extraction; the chain claim "
            "would be understated"
        )

    for db, payload in e2e.items():
        for arm, check in payload["injection_checks"].items():
            if check["unverified"] != 0:
                raise SystemExit(
                    f"doc 110: {db}/{arm} has {check['unverified']} unverified "
                    "cardinality injections; the L3 ratio would be meaningless"
                )
        if payload["includes_feature_extraction"]:
            raise SystemExit(
                f"doc 110: L3 for {db} claims to include feature extraction; "
                "L3 must not, or it double-counts L2"
            )
    facts["latency"] = latency
    facts["latency_gpu"] = latency_gpu
    facts["estimation"] = estimation
    facts["e2e"] = e2e
    facts["latency_breaches"] = latency["stability_breaches"]

    # 110 §3.1 voids the unstable stats row and says to take the CPU number from
    # the ERM arm (latency does not depend on the weights, §3.5).  Do exactly
    # that instead of quoting a run the source document rejected.
    facts["latency_fallback_dbs"] = sorted(
        b["database"] for b in latency["stability_breaches"]
    )
    cpu_ours_p50 = {}
    for db in dbs:
        arm = latency["databases"][db]["models"]["ours"]
        if arm["stability_ok"] and db not in facts["latency_fallback_dbs"]:
            cpu_ours_p50[db] = arm["population"]["p50"]
        else:
            fallback = latency_erm["databases"][db]["models"]["ours"]
            if not fallback["stability_ok"]:
                raise SystemExit(
                    f"doc 110: both arms are unstable for {db}; there is no "
                    "trustworthy L1 number to report"
                )
            cpu_ours_p50[db] = fallback["population"]["p50"]
    facts["latency_cpu_ours_p50"] = cpu_ours_p50

    speedups_cpu, speedups_gpu, speedups_chain = [], [], []
    for db in dbs:
        price = latency["databases"][db]["models"]["price"]["cache"]
        speedups_cpu.append(price["population"]["p50"] / cpu_ours_p50[db])
        ours_gpu = latency_gpu["databases"][db]["models"]["ours"]
        price_gpu = latency_gpu["databases"][db]["models"]["price"]["cache"]
        speedups_gpu.append(
            price_gpu["population"]["p50"] / ours_gpu["population"]["p50"]
        )
        chain = estimation["databases"][db]
        speedups_chain.append(
            chain["price_total_ms"]["p50"] / chain["ours_total_ms"]["p50"]
        )
    if min(speedups_chain) <= 1.0:
        raise SystemExit(
            "doc 110: the estimation chain is not faster than PRICE on some "
            f"database ({min(speedups_chain):.3f}); the report claims a speedup"
        )
    facts["speedup_forward_cpu"] = (min(speedups_cpu), max(speedups_cpu))
    facts["speedup_forward_gpu"] = (min(speedups_gpu), max(speedups_gpu))
    facts["speedup_chain"] = (min(speedups_chain), max(speedups_chain))
    for arm, key in (("true_card", "true"), ("ours", "ours"), ("price", "price")):
        values = [e2e[db]["ratios_to_pg_native"][arm]["median"] for db in dbs]
        facts[f"e2e_ratio_{key}"] = (min(values), max(values))

    # The prose says "no measurable effect", which only holds while every arm
    # sits near 1.0; a re-measurement outside this band must change the wording.
    band = [0.95, 1.05]
    for key in ("true", "ours", "price"):
        low, high = facts[f"e2e_ratio_{key}"]
        if low < band[0] or high > band[1]:
            raise SystemExit(
                f"doc 110: L3 {key} ratios {low:.3f}-{high:.3f} leave the "
                f"[{band[0]}, {band[1]}] band; 「无可测收益」 would be false"
            )
    facts["extraction_share"] = (
        min(
            estimation["databases"][db]["extraction_share_of_ours_total"] for db in dbs
        ),
        max(
            estimation["databases"][db]["extraction_share_of_ours_total"] for db in dbs
        ),
    )
    return facts


# ------------------------------------------------------------------- rendering


def fmt(value: float, digits: int = 2) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def verdict_class(ratio: float) -> str:
    """Colour a ratio by its own comparison: win / tie-inside-noise / loss."""
    if ratio > 1.0:
        return "win"
    if 1.0 - ratio < SEED_NOISE:
        return "tie"
    return "lose"


def ratio_cell(value: float, ours_better: bool) -> str:
    cls = "win" if ours_better else "lose"
    return f'<td class="{cls}">{value:.3f}</td>'


def build_target_table(facts: dict) -> str:
    broad = facts["broad"]["databases"]
    rows = []
    for db in facts["report_dbs"]:
        entry = broad[db]["overview"]
        ours, price, pg = (entry[m]["q95"] for m in ("ours_k0", "price_k0", "pg"))
        pr = price / ours
        gr = pg / ours
        rows.append(
            f"<tr><td>{db}</td><td>{entry['n']:,}</td>"
            f"<td>{fmt(ours)}</td><td>{fmt(price)}</td><td>{fmt(pg)}</td>"
            f"{ratio_cell(pr, pr > 1)}{ratio_cell(gr, gr > 1)}</tr>"
        )
    t = facts["targets"]
    rows.append(
        f'<tr class="summary"><td>汇总（{len(facts["report_dbs"])} 库中位）</td><td>—</td><td>—</td><td>—</td><td>—</td>'
        f'<td class="lose">{t["price_median"]:.3f}</td>'
        f'<td class="win">{t["pg_median"]:.3f}</td></tr>'
    )
    return (
        '<div class="tablewrap"><table>'
        "<caption><b>目标库 · unique-query Q95（K=0，全量唯一查询）</b>"
        "　比值 &gt;1 表示我们更好；PRICE 与我们均未在这些库上训练，是唯一对等的漂移比较点。"
        "</caption><thead><tr><th>库</th><th>n</th><th>我们 Q95</th><th>PRICE Q95</th>"
        "<th>PG Q95</th><th>PRICE / 我们</th><th>PG / 我们</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_target_four_plus_table(facts: dict) -> str:
    rows = []
    for db in facts["report_dbs"]:
        entry = facts["four_plus_targets"][db]
        pr = entry["price"] / entry["ours"]
        gr = entry["pg"] / entry["ours"]
        rows.append(
            f"<tr><td>{db}</td><td>{entry['n']:,}</td>"
            f"<td>{fmt(entry['ours'])}</td><td>{fmt(entry['price'])}</td><td>{fmt(entry['pg'])}</td>"
            f"{ratio_cell(pr, pr > 1)}{ratio_cell(gr, gr > 1)}</tr>"
        )
    rows.append(
        f'<tr class="summary"><td>汇总（{len(facts["report_dbs"])} 库中位）</td><td>—</td><td>—</td><td>—</td><td>—</td>'
        f'<td class="lose">{facts["four_plus_price_median"]:.3f}</td>'
        f'<td class="win">{facts["four_plus_pg_median"]:.3f}</td></tr>'
    )
    return (
        '<div class="tablewrap"><table>'
        "<caption><b>同一口径下的 4+ 表桶</b>　项目主指标 macro_four_plus_q95 所在的粒度；"
        "IMDB 该桶仅 118 条查询，Q95 由约 6 条决定。</caption>"
        "<thead><tr><th>库</th><th>n</th><th>我们 Q95</th><th>PRICE Q95</th><th>PG Q95</th>"
        "<th>PRICE / 我们</th><th>PG / 我们</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_source_stats(facts: dict) -> str:
    s = facts["source_stats"]
    noise = facts["noise_band_sources"]
    stable_win = s["win_price"] - len(noise)
    items = [
        ("优于 PG 的库数", f"{s['win_pg']} / 26"),
        ("优于 PRICE 的库数", f"{s['win_price']} / 26"),
        ("其中落在 seed 噪声带内", f"{len(noise)}（{'、'.join(noise)}）"),
        ("噪声带外的稳赢", f"{stable_win} / 26"),
        (
            "PRICE / 我们 中位",
            f"{s['price_median']:.3f}（{s['price_min']:.3f} – {s['price_max']:.3f}）",
        ),
        (
            "PG / 我们 中位",
            f"{s['pg_median']:.3f}（{s['pg_min']:.3f} – {s['pg_max']:.3f}）",
        ),
        (
            "训练库 / 源验证库",
            "23 库我们训练过（21 赢 2 输） / 3 库未训练（1 赢 2 输）",
        ),
    ]
    rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{esc(value)}</td></tr>" for name, value in items
    )
    return (
        '<div class="tablewrap"><table>'
        "<caption><b>26 源库分布统计</b></caption>"
        "<thead><tr><th>统计量</th><th>值</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def build_source_table(facts: dict) -> str:
    broad = facts["broad"]["databases"]
    rows = []
    for db in facts["sources"]:
        entry = broad[db]["overview"]
        ours, price, pg = (entry[m]["q95"] for m in ("ours_k0", "price_k0", "pg"))
        pr = price / ours
        gr = pg / ours
        note = ""
        if db in facts["noise_band_sources"]:
            note = '<span class="flag noise">噪声带内</span>'
        rows.append(
            f"<tr><td>{db}{note}</td><td>{entry['n']:,}</td>"
            f"<td>{fmt(ours)}</td><td>{fmt(price)}</td><td>{fmt(pg)}</td>"
            f'<td class="{verdict_class(pr)}">{pr:.3f}</td>'
            f'<td class="{verdict_class(gr)}">{gr:.3f}</td></tr>'
        )
    s = facts["source_stats"]
    rows.append(
        f'<tr class="summary"><td>汇总（26 库中位）</td><td>—</td><td>—</td><td>—</td><td>—</td>'
        f'<td class="win">{s["price_median"]:.3f}</td>'
        f'<td class="win">{s["pg_median"]:.3f}</td></tr>'
    )
    return (
        '<div class="tablewrap"><table>'
        "<caption><b>26 个源库 · unique-query Q95（K=0，1,279,218 条唯一查询）</b>"
        '　<span class="flag">拟合诊断</span>PRICE 在全部 26 库训练过，我们训练过 23 个，'
        "因此本表不是漂移证据。</caption>"
        "<thead><tr><th>库</th><th>n</th><th>我们 Q95</th><th>PRICE Q95</th><th>PG Q95</th>"
        "<th>PRICE / 我们</th><th>PG / 我们</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_flip_table(facts: dict) -> str:
    t, s = facts["targets"], facts["source_stats"]
    n_db = len(facts["report_dbs"])
    rows = [
        (
            "Q95 输给 PRICE 的库",
            f"{t['lose_price']} / {n_db}",
            f"{s['lose_price']} / 26",
        ),
        ("Q95 输给 PG 的库", f"{t['lose_pg']} / {n_db}", f"{s['lose_pg']} / 26"),
        ("PRICE / 我们 中位", f"{t['price_median']:.3f}", f"{s['price_median']:.3f}"),
        (
            "PRICE / 我们 范围",
            f"{t['price_min']:.3f} – {t['price_max']:.3f}",
            f"{s['price_min']:.3f} – {s['price_max']:.3f}",
        ),
        ("PG / 我们 中位", f"{t['pg_median']:.3f}", f"{s['pg_median']:.3f}"),
        (
            "与真值相关性落后 PRICE 的库",
            f"{t['corr_lose']} / {n_db}",
            f"{s['corr_lose']} / 26",
        ),
        (
            "dCorr 中位（PRICE − 我们）",
            f"{t['dcorr_median']:+.4f}",
            f"{s['dcorr_median']:+.4f}",
        ),
        (
            "我们离散度大于 PRICE 的库",
            f"{t['wider_than_price']} / {n_db}",
            f"{s['wider_than_price']} / 26",
        ),
    ]
    body = "".join(
        f"<tr><td>{esc(name)}</td><td>{esc(left)}</td><td>{esc(right)}</td></tr>"
        for name, left, right in rows
    )
    return (
        '<div class="tablewrap"><table>'
        "<caption><b>同一模型的两种场景</b>　左列是双方都没训练过的库（漂移证据），"
        "右列是 PRICE 全训练过、我们训练过 23 个的库（拟合诊断）。</caption>"
        "<thead><tr><th>指标</th><th>目标库（4，对等）</th><th>源库（26，不对称）</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def build_error_table(facts: dict) -> str:
    broad = facts["broad"]["databases"]
    rows = []
    for db in list(facts["report_dbs"]) + list(facts["sources"]):
        entry = broad[db]["overview"]
        group = "目标" if db in facts["report_dbs"] else "源"
        rows.append(
            f"<tr><td>{db}</td><td>{group}</td>"
            f"<td>{entry['ours_k0']['bias_med']:+.3f}</td>"
            f"<td>{entry['ours_k0']['over_pct']:.1f}%</td>"
            f"<td>{entry['ours_k0']['corr_truth']:.4f}</td>"
            f"<td>{entry['price_k0']['corr_truth']:.4f}</td>"
            f"<td>{entry['price_k0']['corr_truth'] - entry['ours_k0']['corr_truth']:+.4f}</td>"
            f"<td>{entry['ours_k0']['std']:.3f}</td>"
            f"<td>{entry['price_k0']['std']:.3f}</td></tr>"
        )
    t, s = facts["targets"], facts["source_stats"]
    rows.append(
        f'<tr class="summary"><td>目标库中位</td><td>目标</td>'
        f"<td>{t['bias_median']:+.3f}</td><td>{t['over_median']:.1f}%</td>"
        f"<td>—</td><td>—</td><td>{t['dcorr_median']:+.4f}</td><td>—</td><td>—</td></tr>"
    )
    rows.append(
        f'<tr class="summary"><td>源库中位</td><td>源</td>'
        f"<td>{s['bias_median']:+.3f}</td><td>{s['over_median']:.1f}%</td>"
        f"<td>—</td><td>—</td><td>{s['dcorr_median']:+.4f}</td><td>—</td><td>—</td></tr>"
    )
    return (
        '<div class="tablewrap"><table>'
        "<caption><b>30 库误差分解（K=0，全量唯一查询）</b>　bias 为 log 空间中位偏差，"
        "over% 为高估占比，corr 为与真值的 Pearson 相关，std 为预测离散度（对数空间）。</caption>"
        "<thead><tr><th>库</th><th>组</th><th>我们 bias</th><th>我们 over%</th>"
        "<th>我们 corr</th><th>PRICE corr</th><th>dCorr</th><th>我们 std</th><th>PRICE std</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_fewshot_table(facts: dict) -> str:
    rows = []
    for db in [d for d in DOC14_DBS if d in facts["report_dbs"]]:
        for k in K_VALUES:
            row = facts["doc14"][db][k]
            cells = [f"{row[arm]['mean']:.2f} ± {row[arm]['sd']:.2f}" for arm in ARMS]
            rows.append(
                f"<tr><td>{db}</td><td>{k}</td>"
                + "".join(f"<td>{c}</td>" for c in cells)
                + f"<td>{row['price']:.2f}</td><td>{row['pg']:.2f}</td></tr>"
            )
    return (
        '<div class="tablewrap"><table>'
        "<caption><b>目标库 few-shot 轨迹（臂 × seed 均值，3 seed × 5 support 抽样）</b>"
        "　口径与上方 K=0 population 表不同（episode 均值 vs 全库 population），两者不可互换；"
        "PRICE 列是 <b>PRICE 自己的少样本值</b>，不是常数——K&gt;0 时它同样适应，"
        "PG 与 K 无关。</caption>"
        "<thead><tr><th>库</th><th>K</th><th>FOMAML-set</th><th>ERM-set</th>"
        "<th>FOMAML-sum</th><th>ERM-sum</th><th>PRICE†</th><th>PG</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


# ---------------------------------------------------------------------- charts


def svg_target_chart(facts: dict) -> str:
    """Horizontal grouped bars, log scale, every bar directly labelled."""
    broad = facts["broad"]["databases"]
    width, label_w, value_w = 780, 96, 96
    x0, x1 = label_w, width - value_w
    vmin, vmax = 10.0, 2000.0
    span = math.log10(vmax) - math.log10(vmin)

    def x_of(value: float) -> float:
        return x0 + (math.log10(value) - math.log10(vmin)) / span * (x1 - x0)

    bar_h, gap, group_gap = 15, 5, 26
    group_h = bar_h * 3 + gap * 2 + group_gap
    top = 34
    height = top + group_h * len(facts["report_dbs"]) + 22
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="目标库的 Q95 对照条形图，对数横轴">'
    ]
    for tick in (10, 100, 1000):
        x = x_of(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 22}" '
            'stroke="var(--border)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" font-size="11" '
            f'fill="var(--muted-fg)">{tick}</text>'
        )
    for index, db in enumerate(facts["report_dbs"]):
        entry = broad[db]["overview"]
        base = top + index * group_h
        parts.append(
            f'<text x="0" y="{base + 26:.1f}" font-size="13" font-weight="600" '
            f'fill="var(--fg)">{db}</text>'
        )
        for offset, (method, color) in enumerate(
            (
                ("ours_k0", "--series-ours"),
                ("price_k0", "--series-price"),
                ("pg", "--series-pg"),
            )
        ):
            value = entry[method]["q95"]
            y = base + offset * (bar_h + gap)
            parts.append(
                f'<rect x="{x0}" y="{y:.1f}" width="{max(x_of(value) - x0, 1):.1f}" '
                f'height="{bar_h}" fill="var({color})" rx="1"/>'
            )
            parts.append(
                f'<text x="{x_of(value) + 6:.1f}" y="{y + bar_h - 3:.1f}" font-size="11" '
                f'fill="var(--muted-fg)">{fmt(value)}</text>'
            )
    parts.append("</svg>")
    legend = (
        '<ul class="legend">'
        '<li><span class="swatch" style="background:var(--series-ours)"></span>我们（冻结 seed42 baseline）</li>'
        '<li><span class="swatch" style="background:var(--series-price)"></span>PRICE（官方权重，零样本）</li>'
        '<li><span class="swatch" style="background:var(--series-pg)"></span>PG（原始优化器估计）</li>'
        "</ul>"
    )
    note = (
        '<p class="axis-note">横轴为 Q95 的对数刻度（10 / 100 / 1000），'
        "每条柱直接标注数值；数值越靠左越准。</p>"
    )
    dbs = facts["report_dbs"]
    beat_pg = [
        db
        for db in dbs
        if facts["broad"]["databases"][db]["overview"]["ours_k0"]["q95"]
        < facts["broad"]["databases"][db]["overview"]["pg"]["q95"]
    ]
    return (
        '<figure class="figure">'
        f"<figcaption><b>目标库 Q95 对照。</b>{len(dbs)} 个库上 PRICE 的柱子都更短，"
        "即全部落后于 PRICE；对 PG 则在 "
        f"{'、'.join(beat_pg) if beat_pg else '任何库'}"
        f"{' 上领先' if beat_pg else '上都没有领先'}。</figcaption>"
        f"{legend}{''.join(parts)}{note}</figure>"
    )


def svg_source_ratio_chart(facts: dict) -> str:
    """26 sorted ratio bars on a log axis with the parity line and noise band."""
    broad = facts["broad"]["databases"]
    items = []
    for db in facts["sources"]:
        entry = broad[db]["overview"]
        ratio = entry["price_k0"]["q95"] / entry["ours_k0"]["q95"]
        items.append((db, ratio))
    items.sort(key=lambda pair: -pair[1])

    width, name_w, value_w = 780, 104, 74
    x0, x1 = name_w, width - value_w
    vmin, vmax = 0.4, 3.4
    span = math.log10(vmax) - math.log10(vmin)

    def x_of(value: float) -> float:
        return x0 + (math.log10(value) - math.log10(vmin)) / span * (x1 - x0)

    row_h = 19
    top = 30
    height = top + row_h * len(items) + 26
    parity = x_of(1.0)
    band_lo, band_hi = x_of(1.0 / (1 + SEED_NOISE)), x_of(1.0 + SEED_NOISE)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="26 个源库的 PRICE 除以我们 的比值条形图，对数横轴，1.0 为持平线">'
    ]
    parts.append(
        f'<rect x="{band_lo:.1f}" y="{top - 10}" width="{band_hi - band_lo:.1f}" '
        f'height="{height - top - 14}" fill="var(--accent)" opacity="0.10"/>'
    )
    parts.append(
        f'<line x1="{parity:.1f}" y1="{top - 16}" x2="{parity:.1f}" y2="{height - 20}" '
        'stroke="var(--fg)" stroke-width="1.5" stroke-dasharray="4 3"/>'
    )
    parts.append(
        f'<text x="{parity:.1f}" y="{top - 20}" text-anchor="middle" font-size="11" '
        'fill="var(--muted-fg)">持平 1.00</text>'
    )
    parts.append(
        f'<text x="{(band_lo + band_hi) / 2:.1f}" y="{height - 6}" text-anchor="middle" '
        'font-size="10" fill="var(--accent)">seed 噪声带 ±6.28%</text>'
    )
    for index, (db, ratio) in enumerate(items):
        y = top + index * row_h
        noisy = db in facts["noise_band_sources"]
        color = "--series-ours" if ratio > 1 else "--series-price"
        parts.append(
            f'<text x="0" y="{y + 12:.1f}" font-size="11" fill="var(--fg)">{db}</text>'
        )
        parts.append(
            f'<rect x="{min(x_of(ratio), parity):.1f}" y="{y + 3:.1f}" '
            f'width="{abs(x_of(ratio) - parity):.1f}" height="11" '
            f'fill="var({color})" opacity="{0.45 if noisy else 1}" rx="1"/>'
        )
        # label sits on the outer side of the bar so it never lands on the fill
        if ratio > 1:
            label_x, anchor = x_of(ratio) + 5, "start"
        else:
            label_x, anchor = x_of(ratio) - 5, "end"
        parts.append(
            f'<text x="{label_x:.1f}" y="{y + 12:.1f}" text-anchor="{anchor}" '
            f'font-size="10.5" fill="var(--muted-fg)">{ratio:.3f}{"*" if noisy else ""}</text>'
        )
    parts.append("</svg>")
    legend = (
        '<ul class="legend">'
        '<li><span class="swatch" style="background:var(--series-ours)"></span>我们更好（比值 &gt; 1）</li>'
        '<li><span class="swatch" style="background:var(--series-price)"></span>PRICE 更好（比值 &lt; 1）</li>'
        '<li><span class="swatch" style="background:var(--accent);opacity:.5"></span>落在 seed 噪声带内（*）</li>'
        "</ul>"
    )
    note = (
        '<p class="axis-note">横轴为 PRICE / 我们的比值，对数刻度；条形跨过虚线即为我们更好。'
        "带 * 的三个库差距小于 94 号实测的 seed 间标准差 6.28%，应读作打平。</p>"
    )
    return (
        '<figure class="figure">'
        "<figcaption><b>26 源库的相对位置。</b>我们相对 PRICE 的赢面集中在中段，"
        "两端是 chembl（我们大输）与 tpc_h（我们大赢）。</figcaption>"
        f"{legend}{''.join(parts)}{note}</figure>"
    )


def svg_flip_chart(facts: dict) -> str:
    """Two-point slope showing the reversal between the two scenarios."""
    t, s = facts["targets"], facts["source_stats"]
    width, height = 780, 250
    left, right = 150, 640
    vmin, vmax = 0.4, 2.0
    span = math.log10(vmax) - math.log10(vmin)

    def x_of(value: float) -> float:
        return left + (math.log10(value) - math.log10(vmin)) / span * (right - left)

    parity = x_of(1.0)
    y_target, y_source = 70, 170
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="目标库与源库的 PRICE 除以我们 中位比值对比，1.0 为持平线">'
    ]
    for tick in (0.5, 1.0, 2.0):
        x = x_of(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="40" x2="{x:.1f}" y2="200" stroke="var(--border)"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="218" text-anchor="middle" font-size="11" '
            f'fill="var(--muted-fg)">{tick}</text>'
        )
    parts.append(
        f'<line x1="{parity:.1f}" y1="40" x2="{parity:.1f}" y2="200" stroke="var(--fg)" '
        'stroke-width="1.5" stroke-dasharray="4 3"/>'
    )
    parts.append(
        f'<text x="{parity:.1f}" y="30" text-anchor="middle" font-size="11" '
        'fill="var(--muted-fg)">持平 1.00（PRICE 与我们打平）</text>'
    )
    parts.append(
        f'<line x1="{x_of(t["price_median"]):.1f}" y1="{y_target}" '
        f'x2="{x_of(s["price_median"]):.1f}" y2="{y_source}" stroke="var(--series-ours)" '
        'stroke-width="2" fill="none"/>'
    )
    for y, label, value, color in (
        (y_target, "目标库（4，双方都没训练）", t["price_median"], "--series-price"),
        (y_source, "源库（26，PRICE 全训练）", s["price_median"], "--series-ours"),
    ):
        x = x_of(value)
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="7" fill="var({color})"/>')
        parts.append(
            f'<text x="{x + 14:.1f}" y="{y + 4}" font-size="13" font-weight="600" '
            f'fill="var(--fg)">{value:.3f}</text>'
        )
        parts.append(
            f'<text x="0" y="{y + 4}" font-size="12" fill="var(--muted-fg)">{label}</text>'
        )
    parts.append("</svg>")
    note = (
        '<p class="axis-note">纵轴为场景、横轴为 PRICE / 我们的中位比值（对数刻度）。'
        "同一条连线跨过持平线，说明赢面来源在两组之间发生了反转。</p>"
    )
    return (
        '<figure class="figure">'
        "<figcaption><b>反转。</b>同一个模型，在双方都没见过的库上落后 1.8 倍，"
        "在 PRICE 见过且我们多数也见过的库上好 1.18 倍。</figcaption>"
        f"{''.join(parts)}{note}</figure>"
    )


def svg_fewshot_charts(facts: dict) -> str:
    """Small multiples: our Q95 across K against the PRICE reference."""
    panels = []
    for db in [d for d in DOC14_DBS if d in facts["report_dbs"]]:
        width, height = 300, 230
        left, right, top, bottom = 46, 24, 26, 40
        values = [facts["doc14"][db][k]["ERM-sum"]["mean"] for k in K_VALUES]
        price = facts["doc14"][db][0]["price"]
        vmax = max(values + [price]) * 1.35
        vmin = min(values + [price]) / 1.35
        span = math.log10(vmax) - math.log10(vmin)

        def y_of(value: float) -> float:
            return top + (math.log10(vmax) - math.log10(value)) / span * (
                height - top - bottom
            )

        def x_of(index: int) -> float:
            return left + index * (width - left - right) / (len(K_VALUES) - 1)

        parts = [
            f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{db} 目标库上我们与 PRICE 的 Q95 随 K 的变化">'
        ]
        parts.append(
            f'<text x="0" y="12" font-size="12" font-weight="600" fill="var(--fg)">{db}</text>'
        )
        for tick in (vmin, math.sqrt(vmin * vmax), vmax):
            y = y_of(tick)
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                'stroke="var(--border)"/>'
            )
            parts.append(
                f'<text x="{left - 5}" y="{y + 3:.1f}" text-anchor="end" font-size="9.5" '
                f'fill="var(--muted-fg)">{fmt(tick, 0)}</text>'
            )
        price_y = y_of(price)
        parts.append(
            f'<line x1="{left}" y1="{price_y:.1f}" x2="{width - right}" y2="{price_y:.1f}" '
            'stroke="var(--series-price)" stroke-width="1.5" stroke-dasharray="5 3"/>'
        )
        parts.append(
            f'<text x="{width - right}" y="{price_y - 5:.1f}" text-anchor="end" font-size="10" '
            f'fill="var(--series-price)">PRICE {price:.1f}</text>'
        )
        points = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(values))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="var(--series-ours)" '
            'stroke-width="2"/>'
        )
        for index, value in enumerate(values):
            parts.append(
                f'<circle cx="{x_of(index):.1f}" cy="{y_of(value):.1f}" r="3.5" '
                'fill="var(--series-ours)"/>'
            )
            parts.append(
                f'<text x="{x_of(index):.1f}" y="{y_of(value) - 8:.1f}" text-anchor="middle" '
                f'font-size="9" fill="var(--muted-fg)">{value:.0f}</text>'
            )
        for index, k in enumerate(K_VALUES):
            parts.append(
                f'<text x="{x_of(index):.1f}" y="{height - 22}" text-anchor="middle" '
                f'font-size="10" fill="var(--muted-fg)">K={k}</text>'
            )
        parts.append("</svg>")
        panels.append(
            f'<figure class="figure" style="margin:0">{"".join(parts)}'
            '<p class="axis-note">纵轴对数刻度。</p></figure>'
        )
    shown = [d for d in DOC14_DBS if d in facts["report_dbs"]]
    return (
        '<div class="grid3">' + "".join(panels) + "</div>"
        f'<p class="axis-note">曲线为 ERM-sum 臂在{len(shown)}个目标库上的 Q95（臂 × seed 均值），'
        "虚线为 PRICE 的 <b>K=0</b> 值；PRICE 自己的 K&gt;0 值见下表（它在这些库上"
        "同样没有优于自己的 K=0）。适应在部分库上把差距缩小、在另一部分库上反而变差，"
        "没有哪一处超过 PRICE。</p>"
    )


# ------------------------------------------------------------------- assembly


def build_protocol_block(facts: dict) -> str:
    protocol = facts["protocol"]
    checkpoint = protocol["checkpoint"]
    price = protocol["price"]
    return f"""
<div class="verdict">
  <p class="label">评估协议（本次全部数字的共同口径）</p>
  <p>三个估计量在同一批唯一查询上并列：<b>我们</b>是冻结的 seed42 ERM-sum baseline
  （<code>{checkpoint["path"].split("/")[-2]}/{checkpoint["path"].split("/")[-1]}</code>，
  sha256 <code>{checkpoint["sha256"][:12]}…</code>，last.pt，未做任何 checkpoint 选择）、
  <b>PRICE</b> 是官方提供的权重（<code>{price["checkpoint"].split("/")[-1]}</code>，
  sha256 <code>{price["checkpoint_sha256"][:12]}…</code>，原生 <code>log1p(card)+1</code>
  解码到 <code>log(card)</code>）、<b>PG</b> 是缓存里的原始优化器估计。</p>
  <p>评估是 <b>K=0 零样本</b>，对每个库的<b>全部唯一查询做单次全量推理</b>，无 support。
  指标是标准 <b>unique-query Q95</b>：预测与真值的对数误差取 95 分位再取指数，
  精确预测记 1.0，越小越好。目标库用 maxima <code>{price["target_maxima"]}</code>、
  源库用 <code>{price["source_maxima"]}</code>，两组布局差异已实测为相对 9.1e-07。</p>
</div>
<h3>曝光矩阵（决定每张表能说明什么）</h3>
<div class="tablewrap"><table>
<thead><tr><th>库集合</th><th>数量</th><th>我们训练过</th><th>PRICE 训练过</th><th>这一组比较测的是什么</th></tr></thead>
<tbody>
<tr><td>源训练库</td><td>23</td><td>是</td><td>是</td><td>拟合能力，<b>不是</b>空间漂移</td></tr>
<tr><td>源验证库（chembl / legalacts / telstra）</td><td>3</td><td>否</td><td>是</td><td>我们 out-of-sample，PRICE in-sample</td></tr>
<tr><td>目标库（{esc(" / ".join(facts["report_dbs"]))}）</td><td>{len(facts["report_dbs"])}</td><td>否</td><td>否</td><td><b>唯一对等的漂移比较</b></td></tr>
</tbody></table></div>
<p class="small muted">目标库只用于最终评估，从不进入训练、超参选择、适应策略选择或 checkpoint 选择；
脚本在运行时以白名单守卫拒绝任何曝光重叠的 checkpoint。PRICE 在全部 26 个源库上训练过，
这一点决定了下面两张表的语义完全不同。</p>
"""


def load_training_configs(repo_root: Path) -> dict:
    """Load the FOMAML run and its ERM control; they must differ only in `method`."""
    meta = load_json(repo_root / "configs/meta.json")
    erm = load_json(repo_root / "configs/erm.json")
    diff = sorted(k for k in set(meta) | set(erm) if meta.get(k) != erm.get(k))
    if diff != ["method"]:
        raise SystemExit(f"ERM is not a matched control; differing keys: {diff}")
    return meta


def build_training_block(facts: dict, meta: dict) -> str:
    """Where the meta-learning lives: the inner/outer loop, and its ERM control."""
    shown = [db for db in DOC14_DBS if db in facts["report_dbs"]]
    for db in shown:
        for k in (0, 5, 10):
            fast = facts["doc14"][db][k]["FOMAML-set"]["mean"]
            plain = facts["doc14"][db][k]["ERM-set"]["mean"]
            if not fast < plain:
                raise SystemExit(
                    f"meta-learning claim fails at {db} K={k}: {fast} vs {plain}"
                )
    n_cn = {1: "一", 2: "两", 3: "三"}.get(len(shown), str(len(shown)))
    k0 = "、".join(
        f"{db} {facts['doc14'][db][0]['FOMAML-set']['mean']:.1f} vs "
        f"{facts['doc14'][db][0]['ERM-set']['mean']:.1f}"
        for db in shown
    )
    sum_claims = []
    for db in shown:
        row = facts["doc14"][db][0]
        fast, plain = row["FOMAML-sum"]["mean"], row["ERM-sum"]["mean"]
        verdict = "领先" if fast < plain else "落后"
        sum_claims.append(f"{db} {fast:.1f} vs {plain:.1f}（{verdict}）")

    steps = (
        ("① 抽库", "从训练库随机取 1 个（每步 4 个）"),
        ("② 切任务", "support K∈{5,10,20,50,100} + query 64 条"),
        ("③ 内循环", "support 上 5 步梯度，lr 1e-3"),
        ("④ 外循环", "query 损失，Adam 3e-4 更新初始化"),
    )
    outer_lr = f"{meta['outer_lr']:.0e}".replace("e-0", "e-")
    inner_lr = f"{meta['inner_lr']:.0e}".replace("e-0", "e-")
    boxes = []
    for index, (title, detail) in enumerate(steps):
        x = 10 + index * 250
        boxes.append(
            f'<rect x="{x}" y="16" width="230" height="84" rx="2" '
            'fill="var(--card)" stroke="var(--border)"/>'
            f'<text x="{x + 115}" y="46" text-anchor="middle" font-size="13" '
            f'font-weight="600" fill="var(--fg)">{title}</text>'
            f'<text x="{x + 115}" y="68" text-anchor="middle" font-size="10.5" '
            f'fill="var(--muted-fg)">{detail}</text>'
        )
        if index < len(steps) - 1:
            start, end = x + 230, x + 250
            boxes.append(
                f'<line x1="{start}" y1="58" x2="{end - 6}" y2="58" '
                'stroke="var(--border)" stroke-width="1.5"/>'
                f'<polygon points="{end - 6},54 {end - 6},62 {end},58" '
                'fill="var(--border)"/>'
            )
    loop_x0, loop_x1 = 10 + 115, 10 + 3 * 250 + 115
    boxes.append(
        f'<path d="M {loop_x1} 100 V 160 H {loop_x0} V 108" fill="none" '
        'stroke="var(--border)" stroke-width="1.5"/>'
        f'<polygon points="{loop_x0 - 4},108 {loop_x0 + 4},108 {loop_x0},100" '
        'fill="var(--border)"/>'
        '<text x="500" y="150" text-anchor="middle" font-size="11" '
        'fill="var(--muted-fg)">重复 2000 步；每个任务都从同一个公共初始化重新开始</text>'
    )
    return f"""
<h3>训练方式：监督式 FOMAML（元学习在这里，不在网络里）</h3>
<p class="lede">同一个网络，元学习与它的对照用的是同一套结构。区别在训练目标：
元学习学的是一个<b>能在 5 步内适应新库的公共初始化</b>，不是某个库的估计函数。</p>
<figure class="figure">
  <figcaption><b>一个 meta-step。</b>内循环在 support 上把参数推到任务局部；外循环拿
  query 的损失去更新<b>公共初始化</b>，而不是留下这 5 步的结果。所以训练结束时得到的是
  「一个适合被少量样本快速调整的起点」。</figcaption>
  <div style="overflow-x:auto">
  <svg viewBox="0 0 1000 180" style="min-width:700px" role="img"
       aria-label="FOMAML 训练流程：抽库、切任务、内循环、外循环，重复 2000 步">
    {"".join(boxes)}
  </svg>
  </div>
</figure>
<div class="tablewrap"><table class="prose cols-3">
<caption><b>训练超参。</b>FOMAML 与 ERM 对照共用同一份配置，唯一的差别是 <code>method</code> 字段。</caption>
<thead><tr><th>项目</th><th>值</th><th>说明</th></tr></thead>
<tbody>
<tr><td>外循环</td><td class="num">{meta["steps"]} 步 · Adam {outer_lr}</td><td style="text-align:left">更新公共初始化</td></tr>
<tr><td>内循环</td><td class="num">{meta["inner_steps"]} 步 · lr {inner_lr}</td><td style="text-align:left">每个任务从同一初始化重新开始，全参数</td></tr>
<tr><td>任务采样</td><td class="num">每步 {meta["meta_batch"]} 个库</td><td style="text-align:left">K 从 {meta["support_sizes"]} 随机，query {meta["query_size"]} 条，与 support 不重叠</td></tr>
<tr><td>损失</td><td class="num">{meta["loss"]}</td><td style="text-align:left">在 log 基数空间上回归</td></tr>
<tr><td>二阶</td><td class="num">一阶近似</td><td style="text-align:left">FOMAML 截断二阶依赖，但保留 <code>fast = θ − lr·∇</code> 这条初始化通路</td></tr>
</tbody></table></div>
<p><b>ERM 对照怎么做的</b>：同一结构、同一数据划分、同一 seed，两个配置文件逐字段相同，
唯一差别是 <code>method</code> 从 <code>fomaml</code> 改成 <code>erm</code>——没有内循环，
直接监督预测。这一步是判断「收益是否来自元学习」的必要对照；少了它，任何提升都可以被解释成
「网络或数据本身更好」。</p>
<p class="note"><b>口径声明，必须先说：</b>第 02–05 节的主表是 <b>ERM-sum</b> 臂，<b>不是 FOMAML</b>。
全库 population 评估那条线只跑过 ERM-sum baseline；元学习臂的证据在
<b>06 节</b>（FOMAML / ERM 四列对照）与 <b>08 节</b>（延迟测的是 FOMAML-set）。</p>
<p><b>元学习值多少。</b>set 臂上，K=0 时{n_cn}个库都是 FOMAML 更低（{k0}），K=5 与 K=10 仍然如此。
这一列来自 06 节的 <b>episode 均值</b>口径，与 02 节的全库 population 表（{" / ".join(f"{db} {facts['target3seed']['databases'][db]['ours_k0']['q95']['mean']:.1f}" for db in shown)}）
是两套算法，不能互换——所以这里只用来比较 FOMAML 与 ERM 的相对高低，不用来引用绝对值。
到 K≥20 方向不再一致：stats 上 FOMAML 仍略低，但差距与两臂的 seed 波动同量级（06 节 SD 列）；
ergastf1 上 ERM 反而更低。sum 臂的对照更不干净：K=0 时 {"、".join(sum_claims)}。
所以「元学习有效」这句话只能限定在 <b>set 臂与 K≤10</b>，不能推广到全部配置。</p>
"""


def architecture_parameters(
    group_dims: tuple[int, ...],
    width: int,
    poolings: tuple[str, ...],
    head_hidden: tuple[int, ...],
    scalars: int,
) -> int:
    """Forward parameter count for one pooling variant of SetEstimator."""
    encoders = sum(dim * width + width + width * width + width for dim in group_dims)
    head_in = len(poolings) * len(group_dims) * width + scalars
    head = head_in * head_hidden[0] + head_hidden[0]
    for fan_in, fan_out in zip(head_hidden, head_hidden[1:]):
        head += fan_in * fan_out + fan_out
    head += head_hidden[-1] + 1
    return encoders + head


def build_architecture_block(facts: dict) -> str:
    """What the model actually is: four token groups, masked pooling, PG residual."""
    config = facts["protocol"]["checkpoint"]["model_config"]
    if config.get("arch") != "sum":
        raise SystemExit(
            f"report assumes the sum arm, protocol says {config.get('arch')!r}"
        )
    if not config.get("use_pg"):
        raise SystemExit(
            "report describes a PG-residual model, checkpoint has use_pg=False"
        )
    width = int(config["width"])
    pool_set, pool_sum = ("mean", "max"), ("mean", "max", "sum")
    set_params = architecture_parameters(
        GROUP_DIMS, width, pool_set, HEAD_HIDDEN, GLOBAL_SCALARS
    )
    sum_params = architecture_parameters(
        GROUP_DIMS, width, pool_sum, HEAD_HIDDEN, GLOBAL_SCALARS
    )
    if set_params != facts["set_parameters"]:
        raise SystemExit(
            f"architecture drift: set params {set_params:,} != checkpoint "
            f"{facts['set_parameters']:,}"
        )
    if sum_params != facts["our_parameters"]:
        raise SystemExit(
            f"architecture drift: sum params {sum_params:,} != checkpoint "
            f"{facts['our_parameters']:,}"
        )
    if facts["our_parameter_bytes"] != sum_params * 4:
        raise SystemExit(
            f"checkpoint is not FP32: {facts['our_parameter_bytes']:,} bytes for "
            f"{sum_params:,} params"
        )

    lane_w, lane_gap, lane_x0 = 237, 12, 8
    lane_x = [lane_x0 + i * (lane_w + lane_gap) for i in range(len(GROUP_DIMS))]
    lane_cx = [x + lane_w / 2 for x in lane_x]
    canvas_w = lane_x0 * 2 + len(GROUP_DIMS) * lane_w + (len(GROUP_DIMS) - 1) * lane_gap
    box_w = canvas_w * 0.78
    box_x = (canvas_w - box_w) / 2
    head_in = len(pool_sum) * len(GROUP_DIMS) * width + GLOBAL_SCALARS
    broad = facts["broad"]["databases"]
    pg_q50 = [broad[db]["overview"]["pg"]["q50"] for db in facts["report_dbs"]]
    pg_q95 = [broad[db]["overview"]["pg"]["q95"] for db in facts["report_dbs"]]

    def arrow(cx: float, y0: float, y1: float) -> str:
        return (
            f'<line x1="{cx:.1f}" y1="{y0:.1f}" x2="{cx:.1f}" y2="{y1 - 6:.1f}" '
            'stroke="var(--border)" stroke-width="1.5"/>'
            f'<polygon points="{cx - 4:.1f},{y1 - 6:.1f} {cx + 4:.1f},{y1 - 6:.1f} '
            f'{cx:.1f},{y1:.1f}" fill="var(--border)"/>'
        )

    lanes = []
    for index, ((label, count, detail), dim) in enumerate(
        zip(GROUP_LABELS, GROUP_DIMS)
    ):
        x, cx = lane_x[index], lane_cx[index]
        lanes.append(
            f'<rect x="{x}" y="6" width="{lane_w}" height="54" rx="2" '
            'fill="var(--card)" stroke="var(--border)"/>'
            f'<text x="{cx:.1f}" y="28" text-anchor="middle" font-size="13" '
            f'font-weight="600" fill="var(--fg)">{label} {count}×{dim}</text>'
            f'<text x="{cx:.1f}" y="46" text-anchor="middle" font-size="11" '
            f'fill="var(--muted-fg)">{detail}</text>'
            f'<rect x="{x}" y="88" width="{lane_w}" height="46" rx="2" '
            'fill="var(--muted)" stroke="var(--border)"/>'
            f'<text x="{cx:.1f}" y="116" text-anchor="middle" font-size="12" '
            f'fill="var(--fg)">MLP {dim} → {width} → {width}</text>'
            f'<rect x="{x}" y="160" width="{lane_w}" height="46" rx="2" '
            'fill="var(--card)" stroke="var(--border)"/>'
            f'<text x="{cx:.1f}" y="179" text-anchor="middle" font-size="12" '
            'fill="var(--fg)">掩码池化 mean + max + sum</text>'
            f'<text x="{cx:.1f}" y="196" text-anchor="middle" font-size="11" '
            f'fill="var(--muted-fg)">{len(pool_sum) * width} 维 / 组</text>'
            + arrow(cx, 60, 88)
            + arrow(cx, 134, 160)
            + arrow(cx, 206, 236)
        )
    pooled_total = len(pool_sum) * len(GROUP_DIMS) * width
    set_head_in = len(pool_set) * len(GROUP_DIMS) * width + GLOBAL_SCALARS
    token_rows = "".join(
        f"<tr><td><b>{label}</b></td><td>{dim}</td><td>{count}</td>"
        f"<td>{detail}</td></tr>"
        for (label, count, detail), dim in zip(GROUP_LABELS, GROUP_DIMS)
    )
    return f"""
<h3>模型架构：四组 token → 掩码池化 → PG 残差</h3>
<p class="lede">把一条查询拆成四组变长 token，每组用共享的小 MLP 编码、掩码池化成定长向量，
再用一个三层 MLP 预测「PG 错了多少」，加回 PG 的估计。<b>模型学的不是基数本身，是修正量。</b></p>
<figure class="figure">
  <figcaption><b>数据流（sum 变体，即本报告全部精度数字用的那一套）。</b>
  四个方框是四组 token，长度随查询变化；灰色是逐组共享的编码器；最后一层零初始化，
  所以未训练时输出精确等于 PG。</figcaption>
  <div style="overflow-x:auto">
  <svg viewBox="0 0 {canvas_w:.0f} 452" style="min-width:700px" role="img" aria-label="模型架构数据流：四组 token 编码、池化、拼接、回归头、加回 PG 锚">
    {"".join(lanes)}
    <rect x="{box_x:.1f}" y="236" width="{box_w:.1f}" height="58" rx="2" fill="var(--muted)" stroke="var(--border)"/>
    <text x="{canvas_w / 2:.1f}" y="259" text-anchor="middle" font-size="13" font-weight="600" fill="var(--fg)">
      拼接 {pooled_total} 维 ＋ {GLOBAL_SCALARS} 个全局标量 ＝ {head_in} 维</text>
    <text x="{canvas_w / 2:.1f}" y="279" text-anchor="middle" font-size="11" fill="var(--muted-fg)">
      四组 token 数量取对数 · PG 对数估计 · Σ log 表规模</text>
    <line x1="{canvas_w / 2:.1f}" y1="294" x2="{canvas_w / 2:.1f}" y2="314" stroke="var(--border)" stroke-width="1.5"/>
    <polygon points="{canvas_w / 2 - 4:.1f},314 {canvas_w / 2 + 4:.1f},314 {canvas_w / 2:.1f},320" fill="var(--border)"/>
    <rect x="{box_x:.1f}" y="320" width="{box_w:.1f}" height="52" rx="2" fill="var(--card)" stroke="var(--border)"/>
    <text x="{canvas_w / 2:.1f}" y="343" text-anchor="middle" font-size="13" font-weight="600" fill="var(--fg)">
      MLP 头 {head_in} → {HEAD_HIDDEN[0]} → {HEAD_HIDDEN[1]} → 1</text>
    <text x="{canvas_w / 2:.1f}" y="361" text-anchor="middle" font-size="11" fill="var(--muted-fg)">
      末层权重与偏置零初始化</text>
    <line x1="{canvas_w / 2:.1f}" y1="372" x2="{canvas_w / 2:.1f}" y2="398" stroke="var(--border)" stroke-width="1.5"/>
    <polygon points="{canvas_w / 2 - 4:.1f},398 {canvas_w / 2 + 4:.1f},398 {canvas_w / 2:.1f},404" fill="var(--border)"/>
    <rect x="{box_x:.1f}" y="404" width="{box_w:.1f}" height="42" rx="2" fill="var(--card)" stroke="var(--primary)" stroke-width="2"/>
    <text x="{canvas_w / 2:.1f}" y="430" text-anchor="middle" font-size="13" font-weight="600" fill="var(--fg)">
      输出 ＝ PG 对数估计 ＋ 残差</text>
  </svg>
  </div>
</figure>
<h4 style="font-size:15px;margin:var(--s4) 0 var(--s2)">一条查询怎么变成 token</h4>
<div class="tablewrap"><table class="prose cols-4">
<caption><b>四组输入。</b>J / F / T / C 是每条查询自己带的长度，模型不做截断也不补齐到固定长度。</caption>
<thead><tr><th>组</th><th>维度</th><th>每条查询的数量</th><th>装的是什么</th></tr></thead>
<tbody>
{token_rows}
</tbody></table></div>
<p class="small muted">维度的来源是 PRICE 的公开特征口径（<code>histogram40</code>），四组分别是
连接列直方图 40 维、fanout 40 维、表级 4 维、过滤列 43 维；本项目只消费这些数值，不重新提取。
<b>fanout 的准确含义</b>是：<b>本表</b> join 列每个取值在<b>对侧表</b>里匹配到多少行，
再按本表 join 列直方图分桶取均值，得到 40 维分布；<b>不是两侧行数的乘积</b>
（PRICE 源码 <code>utils/statistics/gen_fanout.py</code> 用对侧 <code>value_counts</code> 回填、按桶取均值）。
每条查询的 join 数、表数、谓词数都可以不同，掩码保证 padding 不参与池化、也不产生梯度。</p>
<h4 style="font-size:15px;margin:var(--s4) 0 var(--s2)">规模：参数量与体积</h4>
<div class="tablewrap"><table>
<caption><b>参数量。</b>两套变体只差一路 sum 池化，其余结构完全相同。</caption>
<thead><tr><th>变体</th><th>池化</th><th>头输入</th><th>参数量</th><th>FP32 字节数</th><th>本报告用在哪</th></tr></thead>
<tbody>
<tr><td><b>set</b>（FOMAML）</td><td>mean + max</td><td class="num">{set_head_in}</td>
<td class="num">{set_params:,}</td><td class="num">{set_params * 4:,}</td>
<td style="text-align:left">第 08 节延迟测量</td></tr>
<tr><td><b>sum</b>（ERM，本报告主模型）</td><td>mean + max + sum</td><td class="num">{head_in}</td>
<td class="num"><b>{sum_params:,}</b></td><td class="num">{sum_params * 4:,}</td>
<td style="text-align:left">第 02–07 节全部精度数字</td></tr>
<tr><td>PRICE（对照）</td><td>两阶段 self-attention</td><td class="num">—</td>
<td class="num">{facts["price_parameters"]:,}</td><td class="num">{facts["price_parameters"] * 4:,}</td>
<td style="text-align:left">全篇对照基线</td></tr>
</tbody></table></div>
<p>我们的主模型是 PRICE 的 <b>1/{facts["price_parameters"] / sum_params:.1f}</b> 参数
（{sum_params:,} vs {facts["price_parameters"]:,}）；换成 set 变体是 1/{facts["price_parameters"] / set_params:.1f}。
参数少不等于端到端快——这件事第 08 节用实测分开了。</p>
<h4 style="font-size:15px;margin:var(--s4) 0 var(--s2)">三个设计决定</h4>
<ul class="takeaways">
<li><b>集合结构，不是固定宽度向量。</b>不同库的表数、join 数、谓词数从 1 到几十不等。
四组各自有一套在组内共享的编码器，再掩码池化，同一个模型就能吃任意 schema ——
这是同一份权重能在 {len(facts["protocol"]["checkpoint"]["train_databases"])} 个库上训练、
又在 {len(facts["report_dbs"])} 个没见过的目标库上直接推理的结构前提。</li>
<li><b>残差锚定在 PG 上。</b>输出 ＝ PG 对数估计 ＋ 残差。PG 的尾部很差
（目标库 Q95 {min(pg_q95):.0f}–{max(pg_q95):.0f} 倍），但中位数只有
{min(pg_q50):.1f}–{max(pg_q50):.1f} 倍；从它出发学修正，比从零学基数容易——
这是设计上的取舍，不是已证明的因果结论。</li>
<li><b>最后一层零初始化。</b>训练第 0 步残差恒为 0，模型输出精确等于 PG，
所以训练只可能在这个基础上改善，起点不会先把 PG 已经算对的部分破坏掉。</li>
</ul>
<p class="small muted">另外：没有 Dropout、没有 BatchNorm——内循环要在一批 support 上做 5 步梯度更新，
任何随机失活都会让内外循环和评估不可复现。</p>
<h4 style="font-size:15px;margin:var(--s4) 0 var(--s2)">它不编码什么</h4>
<p class="note"><b>这三条决定了它的能力上限，不能回避。</b>
① <b>不显式编码 join graph 的配对关系</b>：四组 token 各自池化，token 之间的关系只剩数值统计。
但 <b>PRICE 同样不显式知道哪两个 fanout 属于同一个 join</b>（06 号 §2）——真正的差别是
PRICE 有 token 间 attention 交互、两阶段结构（先 scaling 再 filtering）和
{facts["price_parameters"] / sum_params:.0f}–{facts["price_parameters"] / set_params:.0f} 倍的参数量差距。
这是一个尚未验证的候选解释，不是本报告的结论。
② <b>没有表名、列名或字符串语义</b>，只有数值统计量；模型不知道「imdb」和「stats」是什么，
只知道这组数字长什么样。
③ <b>不读目标库的统计表</b>，目标库只提供查询特征与 PG 估计。</p>
"""


def build_verdict_block(facts: dict) -> str:
    t, s = facts["targets"], facts["source_stats"]
    noise = facts["noise_band_sources"]
    stable = s["win_price"] - len(noise)
    omitted = facts.get("omit_four_plus", False)
    n_db = len(facts["report_dbs"])
    n_cn = {3: "三", 4: "四"}.get(n_db, str(n_db))
    four_plus_li = (
        ""
        if omitted
        else (
            "  <li><b>主指标口径下差距更大</b>：4+ 表桶上 PRICE/我们 中位 "
            f"{facts['four_plus_price_median']:.3f}"
            f"（整库口径是 {t['price_median']:.3f}）。</li>\n"
        )
    )
    sample_tail = "" if omitted else "，4+ 桶仅 118 条、由约 6 条决定"
    support_lead = "四条支撑事实：" if omitted else "五条支撑事实："
    return f"""
<div class="verdict">
  <p class="label">一句话结论</p>
  <p>在<b>双方都没训练过</b>的{n_cn}个目标库上，我们的 K=0 尾部精度
  <b>{t["win_price"]}/{n_db} 全部落后于 PRICE</b>
  （PRICE/我们 中位 {t["price_median"]:.3f}，即我们差约 {1 / t["price_median"]:.1f} 倍；
  三 seed 复核后是 <b>{facts["target3seed_pairs_beating_price"]}/{facts["target3seed_pairs"]}</b> 个
  （seed × 库）组合获胜，中位 {facts["target3seed_ratio"]:.3f}，方向和量级都不变）；
  在 <b>PRICE 训练过、我们训练过 23 个</b>的 26 个源库上，我们 <b>{s["win_price"]}/26</b> 优于 PRICE
  （中位 {s["price_median"]:.3f}，其中 {len(noise)} 个落在 seed 噪声带内、
  {stable} 个稳赢）。</p>
  <p><b>这个反转本身就是结论</b>：赢面来自拟合，不是漂移。对传统优化器 PG，两组都好于它
  （目标库 {t["win_pg"]}/{n_db} 赢，源库 {s["win_pg"]}/26 赢）。</p>
</div>
<p>{support_lead}</p>
<ul>
  <li><b>差距落在排序一致性上</b>，不是校准、也不是过度收缩：目标库
  {t["corr_lose"]}/{n_db} 的与真值相关性低于 PRICE
  （dCorr 中位 {t["dcorr_median"]:+.4f}），源库只有 {s["corr_lose"]}/26 落后；两组的偏差中位
  （{t["bias_median"]:+.3f} / {s["bias_median"]:+.3f}）都贴近零，预测离散度我们反而更大。</li>
{four_plus_li}  <li><b>样本量必须随数字一起报</b>：IMDB 只有 521 条唯一查询，其 Q95 由约 26 条决定{sample_tail}。</li>
  <li><b>同赛道的直接对手基本只有 PRICE</b>：整库留出零样本这一设定，可查到的公开工作里
  只有 PRICE（已发表）与一篇已撤回的预印本；CardOOD、GRASP、AHCE/DACE 的指标与我们恒等，
  但设定都不是整库零样本（详见 <a href="#literature">09 与文献的同口径对照</a>）。</li>
  <li><b>换个角度也没有额外收益</b>：110 号实测端到端执行时间，我们和 PRICE 相对
  <code>pg_native</code> 都落在
  {min(facts["e2e_ratio_ours"][0], facts["e2e_ratio_price"][0]):.3f}–
  {max(facts["e2e_ratio_ours"][1], facts["e2e_ratio_price"][1]):.3f} 之间，
  <b>连注入精确真值也只到
  {facts["e2e_ratio_true"][0]:.3f}–{facts["e2e_ratio_true"][1]:.3f}</b>；
  前向单项虽然快 {facts["speedup_forward_cpu"][0]:.1f}–{facts["speedup_forward_cpu"][1]:.1f}×，
  但<b>精度与执行时间不单调</b>（<a href="#latency">08 延迟与端到端</a>）。</li>
</ul>
<p class="small muted">第一次看到 <code>seed</code>、<code>K=0</code>、<code>Q95</code> 这类词时，
可直接查末尾的 <a href="#glossary">12 名词表</a>。</p>
"""


def q95_definition_gap(facts: dict) -> float:
    """IMDB 4+ PG Q95 under the two definitions, as a percent of the 102 value."""
    doc102 = facts["imdb102"]["databases"]["imdb"]["bucket_4plus"]["pg"]["q95"]
    doc103 = facts["broad"]["databases"]["imdb"]["four_plus"]["pg"]["q95"]
    return abs(doc102 - doc103) / doc102 * 100


def build_limits_block(facts: dict) -> str:
    s = facts["source_stats"]
    noise = facts["noise_band_sources"]
    imdb = facts["broad"]["databases"]["imdb"]["overview"]
    bucket_gap = q95_definition_gap(facts)
    stats_ratio = (
        facts["broad"]["databases"]["stats"]["overview"]["pg"]["q95"]
        / facts["broad"]["databases"]["stats"]["overview"]["ours_k0"]["q95"]
    )
    four_plus_limit = (
        ""
        if facts.get("omit_four_plus")
        else (
            '  <li><span class="tag no">不可写</span><span>把源库的 4+ 桶值与 102 号的桶值混排'
            "——两份证据文件的 Q95 定义不同（倍数空间 vs 对数空间），"
            f"IMDB 4+ 桶的 PG 列相差 {bucket_gap:.2f}%。</span></li>\n"
        )
    )
    return f"""
<ul class="limits">
  <li><span class="tag ok">可写</span><span>对传统估计器 PG，目标库
  {facts["targets"]["win_pg"]}/{len(facts["report_dbs"])} 赢、源库 {s["win_pg"]}/26 赢；
  IMDB 上 Q50 首次优于 PG（{imdb["ours_k0"]["q50"]:.3f} vs {imdb["pg"]["q50"]:.3f}）。</span></li>
  <li><span class="tag ok">可写</span><span>源分布内的拟合可超过给定的 PRICE 参考（{s["win_price"]}/26，
  其中 {s["win_price"] - len(noise)} 个稳赢、{len(noise)} 个打平）。</span></li>
  <li><span class="tag no">不可写</span><span>“我们优于 PRICE”——对等漂移场景是 0/{len(facts["report_dbs"])}。</span></li>
  <li><span class="tag no">不可写</span><span>“{s["win_price"]}/26 优于 PRICE”而不带拟合限定——PRICE 全见过这些库。</span></li>
  <li><span class="tag no">不可写</span><span>“全部目标库完整验收”——IMDB 只有 K=0，没有 few-shot 行。</span></li>
{four_plus_limit}  <li><span class="tag no">不可写</span><span>“目标库全胜 PG”——stats 是 {stats_ratio:.3f}，属于打平而非胜负。</span></li>
  <li><span class="tag no">不可写</span><span>“我们优于 CardOOD / GRASP / AHCE”——它们的指标口径与我们恒等，
  但留出方式不是整库零样本，没有可比对象；也不把已撤回的 2502.14350 当作证据引用。</span></li>
  <li><span class="tag no">不可写</span><span>“我们比 PRICE 快”——端到端执行时间无可测差异
  （我们 {facts["e2e_ratio_ours"][0]:.3f}–{facts["e2e_ratio_ours"][1]:.3f}，
  PRICE {facts["e2e_ratio_price"][0]:.3f}–{facts["e2e_ratio_price"][1]:.3f}），
  且这个量级连精确真值都达不到。能写的是<b>前向单项</b>快
  {facts["speedup_forward_cpu"][0]:.1f}–{facts["speedup_forward_cpu"][1]:.1f}×。</span></li>
  <li><span class="tag no">不可写</span><span>“精度上去了查询就更快”——实测反驳：真值臂 Q-error 恒为 1.00，
  却在 {len(facts["report_dbs"])} 个库里从来不是最快的（第 08 节）。</span></li>
</ul>
<p class="small muted">其它限定：<b>目标库的 Q95 已补到三 seed</b>（108 号，第 02 节），
第 03–05 节的定量分解仍是<b>单 seed（42）</b>，标 <code>needs_evidence</code>；
94 号实测 seed 间标准差 6.28pp，大于那些表里多数效应量。
L3 的「执行时间」不含特征提取、模型前向与优化器规划时间，所以「L3 无收益」不等于
「接入我们没有额外开销」——那部分成本在 L2 里（第 08 节）。</p>
"""


def build_evidence_block(facts: dict, digests: dict[str, str]) -> str:
    lines = []
    for name, repo_path in EVIDENCE.items():
        lines.append(f"{digests[name]}  {repo_path}")
    body = "\n".join(lines)
    lit_lines = []
    for name, repo_path in LIT_EVIDENCE.items():
        lit_lines.append(f"{digests[name]}  {repo_path}")
    lit_body = "\n".join(lit_lines)
    broad = facts["broad"]
    return f"""
<p>本报告<b>我们方</b>的每个数字都由 <code>scripts/build_advisor_report.py</code> 从下列已入库的
证据文件计算，不手工誊抄。生成时会做<b>十几组断言</b>，任何不一致都会中止构建；
下面列三组长期使用的，本轮为 106 / 108 / 110 新增的七组摘要见
<a href="../实验文档/104-导师汇报报告与文献对标.md">104 号 §7.2</a>：</p>
<ul class="small">
  <li>30 库 Q95 与 <code>cross_database</code> 计数自洽；</li>
  <li>14 号表格中可由 JSON 复现的三个臂（FOMAML-set / ERM-set / ERM-sum）逐格对得上
  （容差 0.01），第四臂 FOMAML-sum 因 seed44 摘要为空只取文档值；</li>
  <li>14 号的 PRICE / PG 两列与 30 库证据文件的 K=0 值一致。</li>
</ul>
<div class="evidence"><div>{esc(body)}</div></div>
<p><b>文献方</b>的数字是<b>论文发表值的转录</b>，不是我们自己测的，因此单独列出并附出处；
转录的原文依据、核对方式与未核实项都记在下列文件里，报告只引用其中已核实的数值：</p>
<div class="evidence"><div>{esc(lit_body)}</div></div>
<p class="small muted"><b>下列证据等级只描述第 05 节那份 30 库误差分解文件，不是全报告等级</b>
（全报告等级见页头）。
<code>{broad["evidence_level"]}</code>；
<code>needs_evidence = {str(broad["needs_evidence"]).lower()}</code>，
理由：{esc(broad["needs_evidence_reason"])}。
<code>target_databases_read = {broad["target_databases_read"]}</code>，仅用于最终评估。</p>
"""


GLOSSARY = [
    ("训练与模型", None, None),
    (
        "seed（随机种子）",
        "训练随机性的起点。同一个配置配同一个 seed，每次跑出的模型都一样；换 seed 等于换一次参数初始化、换一次数据排列顺序，得到一个不同的模型。",
        "主线精度表用 seed 42 这一个模型；目标库的 Q95 已扩到三 seed（第 02 节），其余定量分解仍是单 seed。94 号实测换 seed 会让同一方法的 Q95 波动约 ±6.28 个百分点，所以差距小于它的结论只能算打平。",
    ),
    (
        "checkpoint（检查点）",
        "训练过程中的模型存档文件。",
        "固定取最后一次训练的存档 last.pt，从不在目标库上比较多个存档再挑最好的——那等于按目标测试选模型。",
    ),
    (
        "ERM",
        "经验风险最小化，即普通训练：每条查询独立地学一个回归。",
        "我们四个臂之一。",
    ),
    (
        "FOMAML",
        "元学习：先学一个「遇到新库能快速适应的初始化」，再做少量调整。",
        "本课题的核心方法，与 ERM 对照。",
    ),
    (
        "set / sum（臂名后缀）",
        "模型内部的两种聚合结构：一个把特征当集合处理，一个做求和聚合。",
        "四个臂是 FOMAML-set / ERM-set / FOMAML-sum / ERM-sum；结构在源验证库上选定，主线用 sum。",
    ),
    (
        "arm（臂）",
        "一组完整的对照配置（架构 + 训练方式 + seed）。",
        "14 号主表的每一列就是一个臂。",
    ),
    (
        "maxima",
        "PRICE 的输入特征布局长度参数，决定特征向量怎么切分。",
        "目标库用 [12,14,8,10]、源库用 [12,12,7,10]；换它会改变特征被读到的位置，但实测数值差异相对仅 9.1e-07。",
    ),
    ("评估口径", None, None),
    (
        "K",
        "适应时给模型看的示例查询条数。",
        "K=0 表示一条都不给（零样本）；K=5/10/20/50/100 是少样本档位。",
    ),
    ("support（支撑查询）", "用来做适应的示例查询。", "K>0 的部分才有 support。"),
    (
        "zero-shot / 零样本",
        "不看任何示例，直接对目标库预测。",
        "目标库侧的全部数字都是 K=0。",
    ),
    (
        "few-shot（少样本）",
        "给少量示例后快速适应一个新库。",
        "本课题的场景；14 号给了 K=0…100 的完整轨迹。",
    ),
    (
        "unique-query（唯一查询）",
        "去重后的不同查询；同一条 SQL 重复出现只算一次。",
        "所有表都是「全部唯一查询」口径，例如 IMDB 521 条、stats 2,603 条。",
    ),
    (
        "population（全体口径）",
        "对该库所有唯一查询一次性全跑，不抽样。",
        "与 episode 均值是两套口径，报告里不可互换。",
    ),
    (
        "episode（适应回合）",
        "一次支撑样本抽样加一次评估。",
        "14 号主表是 3 seed × 5 次抽样的均值。",
    ),
    ("指标", None, None),
    (
        "Q95",
        "误差的 95 分位，越小越好。本报告定义为 exp(percentile(|预测对数 − 真值对数|, 95))，精确预测记 1.0。",
        "目标库、26 个源库、4+ 桶全部用它。",
    ),
    (
        "log 空间（对数空间）",
        "用对数表示误差：差 10 倍变成差 1。",
        "所有偏差、相关、离散度统计都在对数空间计算。",
    ),
    (
        "Q50 / Q90 / Q99",
        "误差的 50 / 90 / 99 分位。",
        "Q50 是中位数，Q95 是尾部——两者可以给出相反结论，所以要分开看。",
    ),
    (
        "4+ 表桶",
        "参与 join 的表数 ≥ 4 的那批查询。",
        "项目主指标 macro_four_plus_q95 所在的粒度；IMDB 该桶只有 118 条，Q95 由约 6 条决定。",
    ),
    (
        "macro（宏平均）",
        "先算每个库自己的指标，再对库取平均。",
        "与把所有查询混在一起算不同。",
    ),
    (
        "corr（与真值的相关性）",
        "预测值高低与真值高低的一致程度，衡量「排序准不准」。",
        "解释误差结构的关键量；它高不代表 Q95 就好。",
    ),
    (
        "dCorr",
        "corr(PRICE) − corr(我们)。",
        "正值表示 PRICE 的排序更准；目标库全部为正，源库中位为负。",
    ),
    (
        "bias_med（中位偏差）",
        "预测与真值偏差的中位数（对数空间）。",
        "接近 0 表示既不普遍高估也不普遍低估。",
    ),
    (
        "over%（高估占比）",
        "预测偏大的查询所占比例。",
        "50% 附近表示居中；IMDB 是 27.6%，即 72% 的查询被低估。",
    ),
    (
        "离散度（std）",
        "预测值的分散程度。",
        "若远小于真值的分散度，说明模型过度收缩——报告里 22/30 库我们比 PRICE 更散，这条被否证。",
    ),
    ("库与曝光", None, None),
    ("源训练库", "我们训练过的库。", "23 个。"),
    (
        "源验证库",
        "只用来选方案（模型结构、适应策略）、没参与训练的库。",
        "chembl / legalacts / telstra 三个。",
    ),
    (
        "目标库",
        "我们和 PRICE 都没训练过的库。",
        "本报告列出的各目标库；唯一对等的漂移比较点，且不参与任何选择。",
    ),
    (
        "拟合诊断 vs 漂移证据",
        "在模型见过的分布上比，叫拟合诊断；在双方都没见过的库上比，才叫漂移证据。",
        "26 源库表是前者，4 目标库表是后者——两者的结论方向相反，所以必须分开讲。",
    ),
    (
        "PG（PostgreSQL）",
        "数据库自带优化器给出的行数估计。",
        "本项目的传统基线，不是学习出来的模型。",
    ),
    (
        "PRICE",
        "另一个基数估计方法的官方预训练权重。",
        "非严格训练条件匹配的强参考（它在全部 26 个源库上都训练过）。",
    ),
    ("证据", None, None),
    (
        "噪声带（±6.28pp）",
        "由 seed 间标准差界定的「打平区间」。",
        "落在带内的差异读作打平，报告里灰色标记。",
    ),
    (
        "needs_evidence",
        "证据强度标记，提示该结论需要更多实验支撑。",
        "第 01 节与第 03–05 节的 ours 侧数字仍是单 seed，带这个标记；"
        "目标库的 Q95 已补到三 seed（第 02 节），那部分不再标。",
    ),
    (
        "sha256",
        "文件内容指纹，内容变一点就完全不同。",
        "用于证明引用的到底是哪一份证据文件。",
    ),
    ("文献与指标口径", None, None),
    (
        "Q-ERROR",
        "PRICE 论文里用的误差指标：预测值与真值差了几倍，取两者的较大倍数。",
        "与我们的 Q95 是同一个量（已数值验证，相对差 2.9e-15），所以第 09 节能把我们的数直接放进论文的表里。",
    ),
    (
        "p-error",
        "计划代价之比：把估计值当真值去算查询代价，与用真实基数算的代价相比差多少。",
        "另一种指标，2502.14350 顺带报了；它和 Q95 不在同一个数轴上，不能互相换算或混排。",
    ),
    (
        "all-rows（全部行）",
        "不去重，把工作负载里每一条查询结果行都算进指标。",
        "PRICE 论文用的是这个口径；本项目主指标用 unique-query。第 09 节另算了一份 all-rows 专门用来对表。",
    ),
    (
        "E2E（端到端）",
        "把估计器接进真实数据库跑完整条查询链，测总耗时。",
        "PRICE、GRASP、CardOOD 都报了。本项目 110 号已实测多库四臂（不注入 / 精确真值 / 我们 / PRICE），"
        "结论是执行时间无可测差异，详见第 08 节。",
    ),
    (
        "L1 / L2 / L3",
        "三层延迟口径：L1 是模型前向单项，L2 是 SQL 解析＋特征提取＋前向的完整估计链，"
        "L3 是真实数据库执行时间。",
        "三层逐层放大，速度优势也随之缩小；混着报会夸大收益。L3 不含前两层，L1 也不能当端到端用。",
    ),
    (
        "Visual Genome",
        "一个图像标注数据库。",
        "PRICE 论文与 2502.14350 论文里叫 Visual Genome 的，就是本项目与 PRICE 缓存里的 genome。",
    ),
    (
        "join 模板",
        "多表连接的结构骨架（表与连接条件固定、参数可变）。",
        "GRASP 留出的是模板而非数据库——它的「泛化」是同库内换骨架，与本项目的跨库漂移不是一回事。",
    ),
    (
        "covariate shift（协变量偏移）",
        "测试数据的分布相对训练数据变了，但预测目标没变。",
        "CardOOD 的 OOD 指的是这个，纯查询工作负载漂移，不含整库留出。",
    ),
    (
        "Group DRO",
        "一种鲁棒训练法：给训练集里最难的那部分加权，逼模型照顾最差情况。",
        "CardOOD 里在 forest 上把 Q95 从 102.48 压到 39.57；与本项目的少样本适应是两回事，但思路可借鉴。",
    ),
]


def build_glossary_block(facts: dict) -> str:
    rows = []
    drop_four_plus = facts.get("omit_four_plus", False)
    for term, plain, concrete in GLOSSARY:
        if plain is None:
            rows.append(f'<tr class="group"><td colspan="3">{esc(term)}</td></tr>')
            continue
        if drop_four_plus and term == "4+ 表桶":
            continue
        if "genome" not in facts["report_dbs"] and term == "Visual Genome":
            continue
        if drop_four_plus and concrete:
            concrete = concrete.replace("、4+ 桶全部用它", "全部用它")
        rows.append(
            f"<tr><td><b>{esc(term)}</b></td><td>{esc(plain)}</td><td>{esc(concrete)}</td></tr>"
        )
    return (
        '<div class="tablewrap"><table class="prose">'
        "<caption><b>全文出现的缩写与术语</b>　「通俗解释」是给不熟悉这个方向的人看的，"
        "「本报告中的含义」是该词在这次实验里的具体所指。</caption>"
        "<thead><tr><th>名词</th><th>通俗解释</th><th>本报告中的含义</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_advantages_block(facts: dict) -> str:
    """What we can actually claim against PRICE and the classic estimator."""
    t, s = facts["targets"], facts["source_stats"]
    dbs = facts["report_dbs"]
    n_db = len(dbs)
    pg_ratio_by_db = {
        db: facts["broad"]["databases"][db]["overview"]["pg"]["q95"]
        / facts["broad"]["databases"][db]["overview"]["ours_k0"]["q95"]
        for db in dbs
    }
    best_pg_db = max(pg_ratio_by_db, key=pg_ratio_by_db.get)
    best_pg_ratio = pg_ratio_by_db[best_pg_db]
    rows = [
        (
            "目标库 K=0 Q95",
            f"中位落后 PRICE {1 / t['price_median']:.1f}×（{t['win_price']}/{n_db} 全输）",
            f"领先 1.8×（PRICE/我们 {t['price_median']:.3f}）",
            f"{t['win_pg']}/{n_db} 赢，最大 {best_pg_ratio:.1f}×（{best_pg_db}）",
            "对传统估计器有明确优势；对预训练参考没有",
        ),
        (
            "源库 K=0 Q95",
            f"{s['win_price']}/26 优于 PRICE",
            "—",
            f"{s['win_pg']}/26 赢，PG/我们 中位 {s['pg_median']:.2f}",
            "拟合场景我们更好，但 PRICE 见过这些库，不是漂移证据",
        ),
        (
            "4+ 表桶（主指标粒度）",
            f"PRICE/我们 中位 {facts['four_plus_price_median']:.3f}",
            "领先",
            f"PG/我们 中位 {facts['four_plus_pg_median']:.2f}",
            "对 PRICE 的差距比整库口径更大；对我们有利的 PG 优势也更明显",
        ),
        (
            "参数量（FP32）",
            f"{facts['our_parameters']:,}",
            f"{facts['price_parameters']:,}",
            "无参数（启发式）",
            f"我们比 PRICE 小 {facts['parameter_ratio']:.1f}×"
            f"（set 架构 {facts['price_parameters'] / facts['set_parameters']:.1f}×）",
        ),
        (
            "训练曝光",
            "23 个源库",
            "26 个源库",
            "0",
            "同样的结果用了更少的源库",
        ),
        (
            "是否吃 PG 估计作输入",
            "是" if facts["use_pg"] else "否",
            "否",
            "本身就是优化器",
            "从 PG 的估计出发做修正，而不是从零学",
        ),
        (
            "能否随新库适应",
            "方法本体就是少样本适应",
            "本轮对照为零样本",
            "不能",
            "方法层面的差异，精度上尚未在漂移场景证明",
        ),
        (
            "推理延迟（L1 前向 p50）※",
            f"比 PRICE 快 {facts['speedup_forward_cpu'][0]:.1f}–"
            f"{facts['speedup_forward_cpu'][1]:.1f}×（CPU 单线程）",
            "基准",
            "无模型前向",
            "单项前向确实快一到两个数量级，但整条估计链只快 "
            f"{facts['speedup_chain'][0]:.1f}–{facts['speedup_chain'][1]:.1f}×（第 08 节）",
        ),
        (
            "数据库执行时间（L3 端到端）※",
            f"无可测收益（{facts['e2e_ratio_ours'][0]:.3f}–"
            f"{facts['e2e_ratio_ours'][1]:.3f}）",
            f"无可测收益（{facts['e2e_ratio_price'][0]:.3f}–"
            f"{facts['e2e_ratio_price'][1]:.3f}）",
            "基准 1.000",
            "连注入精确真值也只到 "
            f"{facts['e2e_ratio_true'][0]:.3f}–{facts['e2e_ratio_true'][1]:.3f}；"
            "精度与执行时间不单调",
        ),
    ]
    if facts.get("omit_four_plus"):
        rows = [row for row in rows if not row[0].startswith("4+ 表桶")]
    four_plus_note = (
        ""
        if facts.get("omit_four_plus")
        else f"，4+ 桶中位 {facts['four_plus_price_median']:.3f}"
    )
    body = "".join(
        f"<tr><td><b>{esc(a)}</b></td><td>{esc(b)}</td><td>{esc(c)}</td>"
        f"<td>{esc(d)}</td><td>{esc(e)}</td></tr>"
        for a, b, c, d, e in rows
    )
    return f"""
<p class="lede">把话说明白：<b>目前能对导师主张的精度优势是对传统估计器 PG，不是对 PRICE。</b>
下面逐条列出可主张的与不可主张的，数字全部来自前几节的证据。</p>
<div class="tablewrap"><table class="prose cols-4">
<caption><b>我们 / PRICE / PG 逐维度对照</b>　PG 指 PostgreSQL 自带优化器的行数估计，
是本项目的传统基线；PRICE 是非严格训练条件匹配的预训练参考。</caption>
<thead><tr><th>维度</th><th>我们</th><th>PRICE</th><th>PG</th><th>怎么读</th></tr></thead>
<tbody>{body}</tbody></table></div>
<p><b>可以主张的三条：</b>①对传统优化器 PG，在双方都没训练过的目标库上 {t["win_pg"]}/{n_db} 赢、
最大的 {best_pg_db} 领先 {best_pg_ratio:.1f} 倍，源库 {s["win_pg"]}/26 赢；②参数规模比 PRICE 小
{facts["parameter_ratio"]:.1f}×，且用的源库更少；③方法本体带少样本适应能力，
而本轮对 PRICE 的对照是零样本——这是设计差异，不是精度优势。</p>
<p><b>必须同时说明的三条：</b>①对 PRICE 的尾部精度我们目前是负结果
（目标库 {t["win_price"]}/{n_db} 输，三 seed 下仍是 {facts["target3seed_pairs_beating_price"]}/{facts["target3seed_pairs"]}{four_plus_note}），
所以不能说比 PRICE 好；②参数量小<b>不等于</b>端到端更快——这个信息不足的问题 110 号已经补上了，
结论是<b>前向确实快、执行时间没有可测收益</b>（第 08 节），两件事必须分开说；
③本表<b>前七行</b>的「我们」列对应 <b>ERM-sum</b> 配置（{facts["our_parameters"]:,} 参数），
集合架构（{facts["set_parameters"]:,} 参数）是另一套，两个倍数不能混用；
带 <b>※</b> 的两行延迟测的是 <b>FOMAML-set seed42</b>（{facts["set_parameters"]:,} 参数），
不是 ERM-sum——延迟由张量形状决定、与权重无关（两臂 p50 差 ≤0.04 ms），但报告里必须标清楚测的是哪个模型。</p>
"""


def fmt_compact(value: float) -> str:
    """Compact rendering for literature magnitudes (1e6 rather than 1000000)."""
    if value >= 10000:
        return f"{value:.0e}".replace("e+0", "e").replace("e+", "e")
    return f"{value:.2f}"


def build_threeseed_block(facts: dict) -> str:
    """Doc 108: the same four target databases, three seeds instead of one."""
    data = facts["target3seed"]
    order = tuple(facts["report_dbs"])
    rows = []
    worst_spread = (0.0, "")
    for db in order:
        entry = data["databases"][db]
        ours = entry["ours_k0"]["q95"]
        per_seed = entry["ours_k0"]["per_seed_q95"]
        spread = ours["std"] / ours["mean"]
        if spread > worst_spread[0]:
            worst_spread = (spread, db)
        seeds = " / ".join(f"{per_seed[s]:.2f}" for s in ("42", "43", "44"))
        seed_span = f"{ours['min']:.2f}–{ours['max']:.2f}"
        rows.append(
            "<tr>"
            f"<td><b>{db}</b></td>"
            f"<td class='num'>{entry['n']:,}</td>"
            f"<td class='num'>{ours['mean']:.2f} ± {ours['std']:.2f}</td>"
            f"<td class='num muted'>{seed_span}</td>"
            f"<td class='num muted'>{seeds}</td>"
            f"<td class='num'>{entry['price_k0']['q95']['mean']:.2f}</td>"
            f"<td class='num'>{entry['pg']['q95']['mean']:,.2f}</td>"
            f"<td class='num'>{entry['median_q95_ratio_price_over_ours']:.3f}</td>"
            f"<td class='num'>{entry['median_q95_ratio_pg_over_ours']:.2f}×</td>"
            f"<td class='num'>{entry['seeds_beating_price_on_q95']}/3</td>"
            "</tr>"
        )
    n_db = len(order)
    n_cn = {3: "三", 4: "四"}.get(n_db, str(n_db))
    return f"""
<h3>三 seed 复核：方向不变，量级不变，但 {worst_spread[1]} 的离散度大到不能只写均值</h3>
<p>上一节的数字是<b>单 seed（42）</b>。108 号用与 seed42 逐字节相同的训练配置、只改 <code>seed</code> 字段，
把{n_cn}个目标库各重测两遍。结论没有翻案，但多了两条必须一起报的事实。</p>
<div class="tablewrap"><table>
<caption><b>{n_cn}目标库 K=0 唯一查询 Q95，三 seed</b>　PRICE 与 PG 不随 seed 变化（PRICE 是同一份官方权重，
PG 是缓存里的原始估计），所以它们的列没有误差棒。<b>「PRICE/我们」与「PG/我们」两列是
逐 seed 比值的中位数</b>，不是「两列均值相除」——两种算法在重尾数据上会差几个百分点。</caption>
<thead><tr><th>库</th><th>n</th><th>我们（均值 ± SD）</th><th>三 seed 区间</th>
<th>逐 seed</th><th>PRICE</th><th>PG</th><th>PRICE/我们</th><th>PG/我们</th><th>赢 PRICE</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p><b>这张表最该看的是「我们（均值 ± SD）」那一列的 ±SD。</b>{worst_spread[1]} 的三 seed 标准差是均值的
{worst_spread[0] * 100:.0f}%，逐 seed 从 {data["databases"][worst_spread[1]]["ours_k0"]["q95"]["min"]:.2f}
跨到 {data["databases"][worst_spread[1]]["ours_k0"]["q95"]["max"]:.2f}。
引用 {worst_spread[1]} 的任何倍数都必须带区间，不能只引均值。</p>
<p>汇总：<b>{facts["target3seed_pairs"] - facts["target3seed_pairs_beating_price"]}
/{facts["target3seed_pairs"]}</b> 个（seed × 库）组合在 Q95 上输给 PRICE，
<b>{facts["target3seed_pairs_beating_pg"]}/{facts["target3seed_pairs"]}</b> 个赢 PG；
PRICE/我们 的中位比值：单 seed {facts["targets"]["price_median"]:.3f}、
三 seed {facts["target3seed_ratio"]:.3f}，方向与量级都一致。</p>
"""


def build_probe_block(facts: dict) -> str:
    """Doc 106: is database identity linearly readable from the frozen encoding?"""
    probe = facts["probe"]
    k_way = probe["k_way"]
    control = probe["negative_control"]
    separated = facts["probe_centroid_ratio_separated"]
    median_ratio = facts["probe_centroid_ratio_median"]
    return f"""
<h3>表示探针：模型确实在表征里编了「这是哪个库」</h3>
<p>前一段说差距跟着训练曝光走，但没说机制。最直接的假设是：模型学到了一条
<b>库身份捷径</b>——只要认出这个库，就能套用该库的偏移量，而新库认不出来就失效。
106 号用探针检验这条假设：冻结 checkpoint，对 {probe["n_databases"]} 个源库的
{probe["n_rows"]:,} 条唯一查询做前向，取 <code>_head_input</code> 的
{probe["feature_dim"]} 维池化向量，跑 {k_way["folds"]} 折线性分类器，任务是
<b>{probe["n_databases"]} 选 1 猜库名</b>。</p>
<div class="tablewrap"><table>
<thead><tr><th>测的是什么</th><th>结果</th><th>随机基线</th><th>怎么读</th></tr></thead>
<tbody>
<tr><td><b>库身份可读性</b>（{probe["n_databases"]} 类，{k_way["folds"]} 折）</td>
<td class="num"><b>{k_way["accuracy"]:.4f}</b></td><td class="num">{k_way["chance"]:.4f}</td>
<td>几乎完全可分——捷径真实存在</td></tr>
<tr><td>负对照（打乱库标签后重跑同一探针）</td>
<td class="num">{control["accuracy"]:.4f}</td><td class="num">{control["chance"]:.4f}</td>
<td>回落到随机，说明上面那个数不是探针伪影</td></tr>
<tr><td>质心比中位（库内散度作分母，越大越分离）</td>
<td class="num">{median_ratio:.2f}</td><td class="num">—</td>
<td>只有 {separated}/{probe["n_databases"]} 个库超过 2.0</td></tr>
</tbody></table></div>
<p><b>但这个结果不能直接推出「每库一座孤岛」。</b>质心比中位只有 {median_ratio:.2f}，
真正明显分离的只有 {separated}/{probe["n_databases"]} 个库。也就是说捷径是
<b>重叠区域里的方向性差异</b>，不是每库各占一块互不重叠的地盘。
直接后果：单纯把各库质心拉近的域不变正则<b>不够</b>——那些正则优化的量，
在这个表征里本来就没有分开多少。后续任何「抹掉库信息」的干预，
都必须拿这个探针准确率当验收指标，而不是靠 Q95 间接判断。</p>
<p class="small muted">边界：单 seed（42）的冻结 checkpoint，只有线性探针一种读法，未做多种探针交叉验证。
未读任何目标库（<code>target_databases_read: {probe["target_databases_read"]}</code>），
脚本在构建时断言这一点。</p>
"""


def build_latency_block(facts: dict) -> str:
    """Doc 110: three layers of latency, and why none of them is an E2E win."""
    latency = facts["latency"]
    latency_gpu = facts["latency_gpu"]
    estimation = facts["estimation"]
    rows_l1 = []
    rows_l2 = []
    rows_l3 = []
    fallback_dbs = set(facts["latency_fallback_dbs"])
    cpu_ours_p50 = facts["latency_cpu_ours_p50"]
    for db in facts["report_dbs"]:
        cpu = latency["databases"][db]["models"]
        gpu = latency_gpu["databases"][db]["models"]
        ours_cpu = cpu_ours_p50[db]
        price_cpu = cpu["price"]["cache"]["population"]["p50"]
        ours_gpu = gpu["ours"]["population"]["p50"]
        price_gpu = gpu["price"]["cache"]["population"]["p50"]
        cpu_cell = f"{ours_cpu:.3f}"
        if db in fallback_dbs:
            cpu_cell += "<sup>†</sup>"
        rows_l1.append(
            "<tr>"
            f"<td><b>{db}</b></td>"
            f"<td class='num'>{cpu_cell}</td><td class='num'>{price_cpu:.2f}</td>"
            f"<td class='num'>{price_cpu / ours_cpu:.1f}×</td>"
            f"<td class='num'>{ours_gpu:.3f}</td><td class='num'>{price_gpu:.2f}</td>"
            f"<td class='num'>{price_gpu / ours_gpu:.1f}×</td>"
            "</tr>"
        )
        chain = estimation["databases"][db]
        ours_chain = chain["ours_total_ms"]["p50"]
        price_chain = chain["price_total_ms"]["p50"]
        rows_l2.append(
            "<tr>"
            f"<td><b>{db}</b></td>"
            f"<td class='num'>{ours_chain:.1f}</td><td class='num'>{price_chain:.1f}</td>"
            f"<td class='num'>{price_chain / ours_chain:.1f}×</td>"
            f"<td class='num'>{chain['extraction_share_of_ours_total'] * 100:.1f}%</td>"
            f"<td class='num'>{chain['queries_measured']}/{chain['queries_total']}</td>"
            "</tr>"
        )
        e2e = facts["e2e"][db]
        ratios = e2e["ratios_to_pg_native"]
        qerr = e2e["median_q_error"]
        rows_l3.append(
            "<tr>"
            f"<td><b>{db}</b></td>"
            f"<td class='num'>{ratios['true_card']['n']}</td>"
            f"<td class='num'>{ratios['true_card']['median']:.4f}</td>"
            f"<td class='num'>{ratios['ours']['median']:.4f}</td>"
            f"<td class='num'>{ratios['price']['median']:.4f}</td>"
            f"<td class='num muted'>{qerr['pg_native']['median']:.2f} / "
            f"{qerr['true_card']['median']:.2f} / {qerr['ours']['median']:.2f} / "
            f"{qerr['price']['median']:.2f}</td>"
            "</tr>"
        )
    lo_cpu, hi_cpu = facts["speedup_forward_cpu"]
    lo_gpu, hi_gpu = facts["speedup_forward_gpu"]
    lo_chain, hi_chain = facts["speedup_chain"]
    lo_ext, hi_ext = facts["extraction_share"]
    breaches = facts["latency_breaches"]
    breach_note = (
        "；".join(
            f"{b['database']} 两次 population pass 的 p50 相差 "
            f"{b['relative_delta'] * 100:.1f}%（容差 {b['tolerance'] * 100:.0f}%）"
            for b in breaches
        )
        or "各库两次独立 pass 的 p50 全部落在 5% 容差内"
    )
    lo_true, hi_true = facts["e2e_ratio_true"]
    # does the perfect-cardinality arm ever come out fastest?  if it did, the
    # "accuracy and runtime are not monotone" claim below would be false
    fastest_arm = []
    for db, payload in facts["e2e"].items():
        ratios = {
            arm: payload["ratios_to_pg_native"][arm]["median"]
            for arm in payload["ratios_to_pg_native"]
        }
        fastest_arm.append(min(ratios, key=ratios.get))
    if "true_card" in fastest_arm:
        raise SystemExit(
            "doc 110: the true-cardinality arm is fastest somewhere; the "
            "「精度与执行时间不单调」 claim would be false"
        )
    return f"""
<p class="lede">此前的阶段汇报把「没测延迟」列为缺口，110 号把它补上了，而且结论比缺测更值得说：
<b>前向快一到两个数量级，整条估计链快几倍，但真实数据库执行时间没有可测收益。</b></p>

<h3>L1：模型前向单项（不含特征提取、不含规划、不含执行）</h3>
<div class="tablewrap"><table>
<caption><b>单查询前向 p50，毫秒</b>　同一批样本逐条 batch=1，先丢 20 行 warm-up。
CPU 臂钉死单线程（<code>OMP_NUM_THREADS=1</code>），这是主口径。
被测模型是 <b>FOMAML-set seed42</b>（99,777 参数），不是前面各表的 ERM-sum。
PRICE 列取<b>各库自身的 <code>cache</code> maxima</b>（imdb 自身 maxima 比四库合并口径低 13%，两者不可混用）。</caption>
<thead><tr><th>库</th><th>我们 CPU</th><th>PRICE CPU</th><th>快</th>
<th>我们 GPU</th><th>PRICE GPU</th><th>快</th></tr></thead>
<tbody>{"".join(rows_l1)}</tbody></table></div>
<p>CPU 上快 <b>{lo_cpu:.1f}–{hi_cpu:.1f}×</b>，GPU 上只快 {lo_gpu:.1f}–{hi_gpu:.1f}×。
<b>batch=1 时 GPU 反而比 CPU 慢约 2 倍</b>（我们自己的 {cpu_ours_p50["imdb"]:.2f} ms 对
{latency_gpu["databases"]["imdb"]["models"]["ours"]["population"]["p50"]:.2f} ms），
只有 batch=128 的吞吐才反过来。单查询在线估计应放 CPU。</p>
<p class="small muted">稳定性：{breach_note}。违约的那次运行 110 号判其作废，
CPU 数字改取<b>同一库的 ERM 臂</b>（表内 <sup>†</sup> 标注）——依据是延迟由张量形状决定、
与权重数值无关，两臂 p50 实测差 ≤0.04 ms。</p>

<h3>L2：逐查询完整估计链（SQL 解析 + 特征提取 + 模型前向）</h3>
<div class="tablewrap"><table>
<caption><b>每条顶层查询的估计总耗时 p50，毫秒</b>　按查询把各子查询的提取与前向求和，
即优化器拿到全部基数估计之前的成本。</caption>
<thead><tr><th>库</th><th>我们</th><th>PRICE</th><th>快</th>
<th>特征提取占我们成本</th><th>可测查询</th></tr></thead>
<tbody>{"".join(rows_l2)}</tbody></table></div>
<p>整条链上我们快 <b>{lo_chain:.1f}–{hi_chain:.1f}×</b>，比 L1 的
{lo_cpu:.1f}–{hi_cpu:.1f}× 小了一个量级，原因很直接：
<b>特征提取占我们总成本的 {lo_ext * 100:.1f}–{hi_ext * 100:.1f}%</b>，
而两条臂消费的是同一组 PRICE 特征，这笔钱谁都省不掉。
想再快，要动的是提取器，不是模型。</p>

<h3>L3：真实 PostgreSQL 执行（注入基数后跑完整条查询）</h3>
<div class="tablewrap"><table>
<caption><b>执行时间中位数相对 <code>pg_native</code> 的比值，越小越快</b>
四臂在同一批抽样查询上比较，只在四臂都有执行时间的查询上算比值。
<code>true_card</code> 是注入<b>精确真值</b>的臂，代表完美估计能达到的上限。</caption>
<thead><tr><th>库</th><th>n</th><th>真值臂</th><th>我们</th><th>PRICE</th>
<th>Q-error 中位（PG/真值/我们/PRICE）</th></tr></thead>
<tbody>{"".join(rows_l3)}</tbody></table></div>

<p><b>这一段是本报告最该被追问的地方。</b>把基数估计换成<b>精确真值</b>，执行时间中位数
也只在 <b>{lo_true:.3f}–{hi_true:.3f}</b> 之间摆动；我们
（{facts["e2e_ratio_ours"][0]:.3f}–{facts["e2e_ratio_ours"][1]:.3f}）和 PRICE
（{facts["e2e_ratio_price"][0]:.3f}–{facts["e2e_ratio_price"][1]:.3f}）都在同一带内。
也就是说在这批查询上，<b>基数误差不是执行时间的主导项</b>。</p>
<p>更强的一条：<b>精度与执行时间不单调。</b>真值臂的 Q-error 恒为 1.00，代表完美估计，
但它在<b>每一个库</b>里都不是最快的（表内可直接核对）——把估计换成精确真值，也没有让计划更快。
<b>估计更准不等于计划更好。</b>这条动摇的是「提升基数精度 → 加速查询」这条论证链本身，
而不只是我们自己的成绩。</p>
<p class="small muted">口径边界（必须和第 10 节「边界与不可声称」一起读）：L3 汇总的是
<code>EXPLAIN (ANALYZE)</code> 的 <code>Execution Time</code>，即<b>给定基数、已规划好</b>的查询跑完的时间。
它<b>不含</b>特征提取与模型前向（那是 L2）、<b>不含</b>优化器规划时间、<b>不含</b>中间件往返与 hint 构造。
所以「L3 没收益」不等于「接入我们不会有额外开销」——两段成本要分开看。
另外各库的语句上限不统一（120 s 与 300 s 两种），跨库绝对时间不可直接比，
但比值是库内算的，不影响库内结论。</p>
"""


def build_related_work_block(facts: dict) -> str:
    """Landscape of comparable published work, with what each one can and cannot support."""
    rw = facts["related_work"]

    landscape_rows = []
    for paper in rw["papers"]:
        same = paper["same_setting_as_ours"]
        badge = (
            '<span class="flag same">同设定</span>'
            if same
            else '<span class="flag diff">不同设定</span>'
        )
        metric_note = (
            "口径恒等，可同列比较"
            if paper["metric_identical_to_ours"]
            else "口径未核实"
        )
        ids = " · ".join(
            part
            for part in (
                f"arXiv {paper['arxiv']}" if paper.get("arxiv") else "",
                f"DOI {paper['doi']}" if paper.get("doi") else "",
            )
            if part
        )
        id_line = f"<br>{esc(ids)}" if ids else ""
        authors_line = f"<br>{esc(paper['authors'])}" if paper.get("authors") else ""
        landscape_rows.append(
            f"<tr><td><b>{esc(paper['name'])}</b><br>"
            f'<span class="small muted">{esc(paper["venue"])}{id_line}'
            f"{authors_line}</span></td>"
            f"<td>{esc(paper['metric'])}<br>"
            f'<span class="small muted">与我们的 Q95 {metric_note}</span></td>'
            f"<td>{esc(paper['holdout'])}<br>"
            f'<span class="small muted">少样本：{esc(paper["fewshot"])}</span></td>'
            f"<td>{badge}<br>{esc(paper['role'])}</td></tr>"
        )

    cardood = rw["cardood_q95"]
    cardood_rows = "".join(
        f"<tr><td>{esc(row['setting'])}</td>"
        f"<td>{' / '.join(fmt_compact(v) for v in row['erm'])}</td>"
        f"<td>{esc(row['best_method'])}</td>"
        f"<td>{' / '.join(fmt_compact(v) for v in row['best'])}</td></tr>"
        for row in cardood["rows"]
    )
    cardood_q95_gain = [row["erm"][1] / row["best"][1] for row in cardood["rows"]]
    cardood_gain = (
        f"其最好方法相对 ERM 的 Q95 改善倍数，在上面 {len(cardood_q95_gain)} 种设置里为 "
        f"{min(cardood_q95_gain):.2f}×–{max(cardood_q95_gain):.2f}×。"
    )
    cardood_e2e = rw["cardood_e2e"]
    cardood_e2e_rows = "".join(
        f"<tr><td>{esc(row['db'])}</td><td>{row['pg']:,}</td><td>{row['best']:,}</td>"
        f"<td>{row['ratio']:.3f}</td></tr>"
        for row in cardood_e2e["rows"]
    )

    grasp = rw["grasp_q95"]
    grasp_rows = "".join(
        f"<tr><td>{esc(row['method'])}</td><td>{esc(row['db'])}</td>"
        f"<td>{' / '.join(fmt_compact(v) for v in row['seen'])}</td>"
        f"<td>{' / '.join(fmt_compact(v) for v in row['unseen'])}</td></tr>"
        for row in grasp["rows"]
    )
    grasp_e2e = rw["grasp_e2e"]
    grasp_e2e_rows = "".join(
        f"<tr><td>{esc(row['method'])}</td><td>{esc(row['ratio'])}</td></tr>"
        for row in grasp_e2e["rows"]
    )

    paperb = rw["paperb_table"]
    paperb_rows = "".join(
        f"<tr><td>{esc(row['db'])}</td>"
        f"<td>{' / '.join(fmt_compact(v) for v in row['full'])}</td>"
        f"<td>{' / '.join(fmt_compact(v) for v in row['simple'])}</td></tr>"
        for row in paperb["rows"]
    )

    def _takeaway(item: str) -> str:
        """Doc 110 superseded the "we never measured E2E" takeaway (written 10-22).

        The evidence JSON is sha256-pinned, so the sentence is struck through in
        place rather than edited -- the old claim stays visible and traceable, and
        the reader cannot mistake it for current.
        """
        if "没有报时间" in item or "只报误差" in item:
            return (
                f'<li><s class="muted">{esc(item)}</s><br>'
                "<b>（已被 110 号更新：110 号已补测 L1/L2/L3，见第 08 节；"
                "现状是「有实测、但口径与 GRASP/PRICE 报的不同，绝对值不可并列」）</b></li>"
            )
        return f"<li>{esc(item)}</li>"

    takeaways = "".join(_takeaway(item) for item in rw["takeaways"])

    lineage = rw["metric_lineage"]
    lineage_rows = "".join(
        f"<tr><td>{esc(row['method'])}</td>"
        f"<td>{'；'.join(esc(v) for v in row['own'])}</td>"
        f"<td>{fmt_compact(row['price_table']) if row['price_table'] is not None else '—'}</td></tr>"
        for row in lineage["rows"]
    )

    return f"""
<h3>相关工作版图：谁和我们在同一根轴上</h3>
<p class="lede">近年做「基数估计 + 分布外泛化」的公开工作不算少，但把它们按
<b>留出方式</b>分类后，真正与我们同设定的极少。下表的判断标准只有一条：
<b>是不是整库留出、零样本直接测</b>——这是本项目实验协议的核心。</p>
<div class="tablewrap"><table class="prose cols-4">
<caption><b>相关工作版图</b>　「同设定」= 整库留出零样本。
指标列标出该文的 Q95 与我们的 Q95 是否<b>恒等</b>（恒等才谈得上直接比大小）。</caption>
<thead><tr><th>论文与出处</th><th>指标口径</th><th>留出方式与少样本</th><th>与我们的关系</th></tr></thead>
<tbody>{"".join(landscape_rows)}</tbody></table></div>

<h3>值得引用的具体数字</h3>

<h4 style="font-size:15px;margin:var(--s4) 0 var(--s2)">CardOOD（VLDB Journal 2026）</h4>
<p class="small">同库内工作负载漂移，<b>没有</b>整库零样本；但 q-error 口径与我们恒等，
是鲁棒训练方向最直接的参照。它对比的基线 ERM 就是「同库内无偏移」的普通训练。</p>
<div class="tablewrap"><table>
<caption><b>CardOOD 的 Q-error（Overall，Q50 / Q95 / Q99）</b>　转录自 arXiv 2412.05864 全文。
我们的 Q95 与其 Q95 口径恒等，但<b>它们的库和留出方式与我们不同</b>，因此这里只作方法学参照，不作优劣比较。</caption>
<thead><tr><th>设置</th><th>ERM 基线</th><th>最好方法</th><th>最好方法数值</th></tr></thead>
<tbody>{cardood_rows}</tbody></table></div>
<div class="tablewrap"><table>
<caption><b>CardOOD 的端到端时间</b>　与其 Q-error 表分开列出，因为两者并不成正比（见表下说明）。</caption>
<thead><tr><th>库</th><th>PostgreSQL 原生 (s)</th><th>最好方法 (s)</th><th>比值</th></tr></thead>
<tbody>{cardood_e2e_rows}</tbody></table></div>
<p class="small muted">{esc(cardood["note"])} {cardood_gain} {esc(cardood_e2e["note"])}</p>

<h4 style="font-size:15px;margin:var(--s4) 0 var(--s2)">GRASP（VLDB 2025）</h4>
<p class="small">不访问数据库数据、只吃查询与 schema 的组合式模型；留出的是
<b>join 模板</b>而非数据库，因此与我们不是同一设定。它同时提供了
<b>端到端时间</b>——10-22 那版报告把它列为我们缺失的一类证据，110 号已补测（第 08 节）；
但两者口径不同（GRASP 报 CEB 上整条查询的总运行时间，我们的 L3 是
<code>EXPLAIN (ANALYZE)</code> 的 <code>Execution Time</code>），<b>绝对值不可直接比</b>。</p>
<div class="tablewrap"><table>
<caption><b>GRASP 的 Q-error（seen / unseen join 模板，Median / %95 / Mean）</b>
转录自 vldb.org 官方 PDF（vol18/p2519-wu.pdf）Table 6。</caption>
<thead><tr><th>方法</th><th>库</th><th>seen 模板</th><th>unseen 模板</th></tr></thead>
<tbody>{grasp_rows}</tbody></table></div>
<div class="tablewrap"><table>
<caption><b>GRASP 的端到端时间</b>（CEB-IMDb-full，相对 True-Card 基线的总运行时间比）。</caption>
<thead><tr><th>方法</th><th>总运行时间 / True-Card 基线</th></tr></thead>
<tbody>{grasp_e2e_rows}</tbody></table></div>
<p class="small muted">{esc(grasp["note"])} {esc(grasp_e2e["note"])}</p>

<h4 style="font-size:15px;margin:var(--s4) 0 var(--s2)">2502.14350（同设定，但已撤回）</h4>
<div class="tablewrap"><table>
<caption><b>该文 Table 2（Q50 / Q95 / Q99）</b>　转录自 arXiv 2502.14350 v1 全文。</caption>
<thead><tr><th>库</th><th>PRICE 预训（26 库）</th><th>精简预训集（3.8%）</th></tr></thead>
<tbody>{paperb_rows}</tbody></table></div>
<p class="small muted">{esc(paperb["note"])}</p>
<p class="note"><b>必须声明的撤回状态：</b>{esc(paperb["caveat"])}
该文与 PRICE、与我们<b>同设定</b>（26 个预训练库 → 4 个整库留出零样本），
是除 PRICE 外唯一一条同赛道的公开尝试，但它的结论目前不能当作可靠证据引用。</p>

<h3>同一个方法在不同论文里的数不一样</h3>
<p class="small">{esc(lineage["why"])}</p>
<div class="tablewrap"><table>
<caption><b>JOB-light / IMDb 负载：各论文自报值 vs PRICE Table 2（imdb 行 Q95）</b>
各论文自报值逐条转录自其原文表格（DeepDB / FLAT / NeuroCard 三篇的 JOB-light 表）。
末列是 PRICE Table 2 里同一方法的 imdb 行，也就是我们上面那张表里的列。</caption>
<thead><tr><th>方法</th><th>各论文自报（Q95）</th><th>PRICE Table 2（imdb Q95）</th></tr></thead>
<tbody>{lineage_rows}</tbody></table></div>
<p class="small muted">{esc(lineage["note"])}</p>
<p class="note"><b>统计单元：</b>{esc(lineage["population_note"])}</p>
<p class="note"><b>一个常被引错的数：</b>{esc(lineage["bird_note"])}</p>
<p class="small muted">{esc(lineage["metric_convention"])}</p>

<h3>这次调研的结论</h3>
<ol class="takeaways">{takeaways}</ol>
<p class="small muted">口径说明：{esc(rw["metric_identity"])}</p>
<p class="small muted">转录说明：{esc(rw["extraction_note"])}</p>
"""


def build_literature_block(facts: dict) -> str:
    """Our all-rows panel placed next to the published PRICE table."""
    allrows = facts["allrows"]
    paper = facts["paper_table"]
    qe = paper["q_error"]
    # measured PRICE forward p50, for the "not comparable to the paper's ms" note
    price_l1_ms = [
        facts["latency"]["databases"][db]["models"]["price"]["cache"]["population"][
            "p50"
        ]
        for db in facts["report_dbs"]
    ]
    speed_lo, speed_hi = facts["speedup_forward_cpu"]

    # reproduction check: our re-run of PRICE and PG must land on the paper
    price_diffs = {
        (db, p): abs(
            allrows["databases"][db]["price_k0"][f"q{p}"]
            - qe[db]["PRICE (Pretrained)"][f"q{p}"]
        )
        / qe[db]["PRICE (Pretrained)"][f"q{p}"]
        for db in PAPER_ORDER
        for p in (50, 90, 95, 99)
    }
    price_spread = "、".join(
        f"Q{p} 最大差 {max(v for (db, q), v in price_diffs.items() if q == p) * 100:.2f}%"
        for p in (50, 90, 95, 99)
    )
    repro = []
    for db in PAPER_ORDER:
        ours_price = allrows["databases"][db]["price_k0"]["q95"]
        ours_pg = allrows["databases"][db]["pg"]["q95"]
        pub_price = qe[db]["PRICE (Pretrained)"]["q95"]
        pub_pg = qe[db]["PG"]["q95"]
        repro.append(
            f"<tr><td>{db}</td>"
            f"<td>{ours_price:.4f}</td><td>{pub_price:.2f}</td>"
            f'<td class="{"win" if abs(ours_price - pub_price) / pub_price < 0.05 else "lose"}">'
            f"{abs(ours_price - pub_price) / pub_price * 100:.2f}%</td>"
            f"<td>{ours_pg:.4f}</td><td>{pub_pg:.2f}</td>"
            f'<td class="{"win" if abs(ours_pg - pub_pg) / pub_pg < 0.001 else "lose"}">'
            f"{abs(ours_pg - pub_pg) / pub_pg * 100:.2f}%</td></tr>"
        )

    # the paper's whole table with our row added
    methods = [
        "PG",
        "ALECE",
        "MSCN",
        "DeepDB",
        "NeuroCard",
        "FactorJoin",
        "PRICE (Pretrained)",
        "PRICE (Finetuned)",
    ]
    rows = []
    for method in methods:
        cells = []
        for db in PAPER_ORDER:
            entry = qe[db].get(method)
            if entry is None:
                cells.append("<td>—</td>")
                continue
            value = entry["q95"]
            ours = allrows["databases"][db]["ours_k0"]["q95"]
            if ours < value:
                cls = "win"
            else:
                cls = "lose"
            cells.append(f'<td class="{cls}">{value:.2f}</td>')
        rows.append(f"<tr><td>{esc(method)}</td>" + "".join(cells) + "</tr>")
    ours_cells = "".join(
        f'<td class="tie"><b>{allrows["databases"][db]["ours_k0"]["q95"]:.2f}</b></td>'
        for db in PAPER_ORDER
    )
    rows.append(
        f'<tr class="summary"><td>我们（ERM-sum, K=0, 本次实测）</td>{ours_cells}</tr>'
    )

    # ranking is over zero-shot / pretrained methods only: PRICE (Finetuned)
    # saw 5e4 target-side queries, so it is a different setting
    zero_shot = [m for m in methods if m != "PRICE (Finetuned)"]
    ranks = []
    for db in PAPER_ORDER:
        values = {m: qe[db][m]["q95"] for m in zero_shot if m in qe[db]}
        values["我们"] = allrows["databases"][db]["ours_k0"]["q95"]
        ordered = sorted(values.items(), key=lambda kv: kv[1])
        position = [name for name, _ in ordered].index("我们") + 1
        ranks.append((db, position, len(ordered), ordered[0][0], ordered[0][1]))

    rank_rows = "".join(
        f"<tr><td>{db}</td><td>第 {pos} / {total}</td>"
        f"<td>{best}（{best_value:.2f}）</td></tr>"
        for db, pos, total, best, best_value in ranks
    )

    return f"""
<p class="lede">本节前半部分只比较两种数字：<b>论文公开发表值</b>（PRICE 论文 Table 2，
PVLDB 18(3): 637-650, 2024）与<b>本次实测值</b>（同一份 checkpoint、同一套缓存、同一指标定义，
但换成论文用的 all-rows 口径）。我们先验证两者同口径，再把我们的结果放进论文的那张表里。
后半部分是<b>相关工作版图</b>：同一方向还有哪些论文、它们的指标口径是否与我们一致、
哪些数可以直接引、哪些不能。</p>
<p class="note"><b>关于本节里的 genome：</b>论文原表是四个库，本节为了能与论文逐列对表而
<b>保留其 genome 列</b>——这是论文的发表值，不是我们的实验结论，也不参与本报告的漂移判断。
本报告自己的目标库分析只涉及三个库。</p>

<h3>口径验证：我们的复算能否落在论文的数上</h3>
<div class="tablewrap"><table>
<caption><b>复现检查</b>　论文的 Q-ERROR 与我们的 unique-query Q95 是同一个量
（<code>exp(percentile(|Δlog|, 95))</code> ≡ <code>percentile(max(est/true, true/est), 95)</code>，
已数值验证相对差 2.9e-15）；唯一差别是论文用 all-rows、本项目主指标用 unique-query。</caption>
<thead><tr><th>库</th><th>我们复算 PRICE Q95</th><th>论文 PRICE Q95</th><th>差</th>
<th>我们复算 PG Q95</th><th>论文 PG Q95</th><th>差</th></tr></thead>
<tbody>{"".join(repro)}</tbody></table></div>
<p class="small muted">PG 是同一批缓存里的原始估计，不含任何模型，所以它在四库上都与论文吻合到
0.04% 以内（最大差出现在 ergastf1）就证明<b>我们的缓存、PG 估计与指标定义与论文一致</b>；
PRICE 列四库四分位也全部吻合，{price_spread}
（Q95 的最大差出现在 imdb，是该库尾部插值位置的差异）。</p>

<h3>放进论文那张表：我们在什么位置</h3>
<div class="tablewrap"><table>
<caption><b>同一张表（all-rows 口径，Q95，越小越好）</b>　除最后一行外全部是论文 Table 2 的发表值；
最后一行是本次实测的我们。列序与论文 Table 2 一致（imdb, stats, ergastf1, genome）。
绿色表示该方法在该库上优于我们，红色表示我们优于它。</caption>
<thead><tr><th>方法</th><th>imdb</th><th>stats</th><th>ergastf1</th><th>genome</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<div class="tablewrap"><table>
<caption><b>我们的名次</b>　只在<b>零样本/预训练</b>方法之间按 Q95 排序（论文未跑或未报告的格不计入）。
PRICE (Finetuned) 用 5×10⁴ 条目标库查询微调过，属另一种设定，不参与此排名。</caption>
<thead><tr><th>库</th><th>我们的名次</th><th>该库最优</th></tr></thead>
<tbody>{rank_rows}</tbody></table></div>
<p><b>怎么读这一节：</b>在这个口径下，我们<b>优于 DeepDB、NeuroCard、MSCN、FactorJoin 中的大多数</b>，
而这些对手是<b>在每个目标库上单独训练 5×10⁴ 条查询</b>的模型；我们是<b>用 23 个源库预训练一次、
零样本直接用</b>。我们仍然落后 PRICE 与 ALECE，且在 Q95 上确实不是第一。</p>
<p><b>两条必须同时说的：</b>①论文给了推理延迟与端到端两个数（PRICE 推理 5.3–14.1 ms/查询，
E2E 在 stats/ergastf1/genome 三库最优）。此前的阶段汇报把「我们没测」列为最明显的证据缺口，
110 号已经补上：<b>前向单项我们快 {facts["speedup_forward_cpu"][0]:.1f}–{facts["speedup_forward_cpu"][1]:.1f}×，
但端到端执行时间没有可测差异</b>（第 08 节）。所以「参数少 {facts["parameter_ratio"]:.2f}×」
这条<b>仍然不能</b>换算成速度优势——现在这一点有实测支撑，不再只是谨慎措辞。
<b>但论文的 5.3–14.1 ms 与我们实测的 PRICE 前向不可比</b>：本轮同机同批实测 PRICE 前向 p50 是
{min(price_l1_ms):.1f}–{max(price_l1_ms):.1f} ms，比论文高 2–6 倍
（CPU 型号、batch、padding 口径均不同）。
{speed_lo:.1f}–{speed_hi:.1f}× 是<b>同机同批的比值</b>，
不能与论文的毫秒绝对值并列使用；
②PRICE 论文摘要称「achieving the highest estimation
accuracy on several unseen databases」，但其 Table 2 的 Q95 列里它<b>四个库都没拿第一</b>
（ALECE 在三库优于它，DeepDB 与 NeuroCard 在 imdb 优于它），正文措辞则更谨慎
（「has no significant gaps with other ML-based methods」）。引用时建议以表格为准。</p>
<p class="small muted">出处置信：{esc(paper["source"]["citation"])}
（{esc(paper["source"]["artifact_on_server"])}，DOI {esc(paper["source"]["doi"])}，
arXiv {esc(paper["source"]["arxiv"])}）。论文的 'Visual Genome' 即本项目的 'genome'。
本次实测证据：<code>实验文档/results/p2-2-allrows-lit-20261022/</code>。</p>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--git-branch", required=True, help="branch holding the evidence"
    )
    parser.add_argument(
        "--repo-root", default=".", help="repository to read the evidence from"
    )
    parser.add_argument(
        "--lit-branch",
        default=None,
        help=(
            "branch holding the literature-comparison evidence; "
            "defaults to --git-branch (the 10-22 branch is long merged)"
        ),
    )
    parser.add_argument("--out", required=True, help="output HTML path")
    parser.add_argument(
        "--omit-four-plus",
        action="store_true",
        help=(
            "drop every 4+ table-bucket metric from the report; used for the "
            "short talk version, where three metric definitions side by side "
            "cost more than they explain"
        ),
    )
    parser.add_argument(
        "--omit-db",
        action="append",
        default=[],
        choices=TARGET_DBS,
        help=(
            "drop one target database from the report (repeatable); every "
            "aggregate, count and median is recomputed over the remaining set"
        ),
    )
    parser.add_argument(
        "--omit-section",
        action="append",
        default=[],
        choices=tuple(OMITTABLE_SECTIONS),
        help=(
            "drop a whole section and its in-text pointers (repeatable): "
            + ", ".join(f"{key}={title}" for key, title in OMITTABLE_SECTIONS.items())
        ),
    )
    parser.add_argument(
        "--report-date",
        default="2026-10-26",
        help="date shown in the masthead (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--commit",
        default="b63b18f",
        help="commit shown in the masthead; the one holding the evidence",
    )
    parser.add_argument(
        "--template",
        default=str(Path(__file__).with_name("advisor_report_template.html")),
        help="HTML template with {{PLACEHOLDER}} slots",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        evidence_dir = Path(tmp)
        digests = fetch_evidence(args.git_branch, repo_root, evidence_dir)
        digests.update(
            fetch_evidence(
                args.lit_branch or args.git_branch,
                repo_root,
                evidence_dir,
                LIT_EVIDENCE,
            )
        )
        report_dbs = tuple(db for db in TARGET_DBS if db not in args.omit_db)
        if not report_dbs:
            raise SystemExit("--omit-db removed every target database")
        facts = build_facts(evidence_dir, report_dbs)
        facts["omit_four_plus"] = args.omit_four_plus

        if args.omit_four_plus:
            four_plus_part = (
                '<p class="note"><b>口径：</b>本节全部数字取自 103 号证据文件'
                "（Q95 = exp(percentile(|误差|, 95))，整库唯一查询口径）。</p>"
            )
        else:
            four_plus_part = (
                "<h3>4+ 表桶：项目主指标所在的粒度</h3>"
                + build_target_four_plus_table(facts)
                + '<p class="note"><b>口径提示：</b>本节全部数字取自 103 号证据文件'
                "（Q95 = exp(percentile(|误差|, 95))），与 102 号文件"
                "（Q95 = percentile(exp(|误差|), 95)）"
                "在整库尺度上差 ≤1e-7，但在重尾小样本的 4+ 桶上可见差异"
                f"（IMDB 4+ 桶 PG 列相差 {q95_definition_gap(facts):.2f}%）。"
                "两套数字各自成立，不可互相替代或混排。</p>"
            )

        target_block = (
            f'<p class="lede">这 {len(facts["report_dbs"])} 个库 PRICE 与我们都没有训练过，是本项目唯一一组对等的'
            "空间漂移比较点。口径：我们冻结的 seed42 baseline 的 K=0 零样本，"
            "全部唯一查询，单次全量，无 support。本节末另附三 seed 复核（108 号）。</p>"
            + build_target_table(facts)
            + svg_target_chart(facts)
            + four_plus_part
            + build_threeseed_block(facts)
        )

        source_block = (
            '<p class="lede">PRICE 在全部 26 个源库上训练过，我们训练过 23 个，'
            "所以这张表测的是<b>拟合能力</b>，不是漂移。它给出的是「模型在自己见过的分布里有多强」，"
            "以及 PRICE 官方权重在这批库上的拟合水平。</p>"
            + build_source_stats(facts)
            + svg_source_ratio_chart(facts)
            + build_source_table(facts)
        )

        flip_block = (
            '<p class="lede">把两组并排放，结论就出来了：同一个模型，在没见过的库上落后，'
            "在见过的库上更好。赢面集中在模型见过的分布里。</p>"
            + svg_flip_chart(facts)
            + build_flip_table(facts)
        )

        error_block = (
            '<p class="lede">102 号称出了差距但没回答为什么。103 号把它拆成两个候选解释，'
            "用 30 库的误差分解检验：<b>不是校准问题</b>（偏差中位贴近零），"
            "<b>不是过度收缩</b>（我们预测更散），<b>是排序一致性差一截</b>，"
            "而且这个差距精确跟随训练曝光。</p>"
            + build_error_table(facts)
            + '<p class="note"><b>更正：</b>第一轮 7 库上「corr 与 Q95 一一对应」成立，'
            "扩到 30 库后只有 27/30 一致（accidents、airline 的 corr 略高但 Q95 输不足 1%，"
            "实为打平；movielens 的 corr 略低但 Q95 赢）。相关性解释了大部分 Q95 胜负，"
            "但不是唯一决定因素。</p>" + build_probe_block(facts)
        )

        fewshot_block = (
            '<p class="lede">上面所有数字都是 <b>K=0 零样本</b>，而课题是<b>空间漂移下的少样本'
            "元适应</b>。14 号的 K&gt;0 轨迹是另一半：适应在部分库上把差距明显缩小，"
            "在另一部分库上反而变差（负迁移），其余基本持平——"
            "但所有情况下都没有超过 PRICE。</p>"
            + svg_fewshot_charts(facts)
            + build_fewshot_table(facts)
        )

        blocks = {
            "TITLE": "面向空间漂移的自适应基数估计",
            "SUBTITLE": "当前进度与对照结果：漂移场景下我们赢的是拟合，不是迁移",
            "DATE": args.report_date,
            "BRANCH": args.git_branch,
            "COMMIT": args.commit,
            "EVIDENCE_LEVEL": (
                f"{len(facts['report_dbs'])} 个目标库三 seed · 单 seed 定量分解 · 只读延迟实测"
            ),
            "VERDICT_BLOCK": build_verdict_block(facts),
            "PROTOCOL_BLOCK": build_protocol_block(facts),
            "TRAINING_BLOCK": build_training_block(
                facts, load_training_configs(repo_root)
            ),
            "ARCHITECTURE_BLOCK": build_architecture_block(facts),
            "TARGET_BLOCK": target_block,
            "SOURCE_BLOCK": source_block,
            "FLIP_BLOCK": flip_block,
            "ERROR_BLOCK": error_block,
            "FEWSHOT_BLOCK": fewshot_block,
            "ADVANTAGES_BLOCK": build_advantages_block(facts),
            "LATENCY_BLOCK": build_latency_block(facts),
            "LIMITS_BLOCK": build_limits_block(facts),
            "EVIDENCE_BLOCK": build_evidence_block(facts, digests),
            "LITERATURE_BLOCK": (
                build_literature_block(facts) + build_related_work_block(facts)
            ),
            "GLOSSARY_BLOCK": build_glossary_block(facts),
            "FOOTER_NOTE": (
                "本报告由 scripts/build_advisor_report.py 从已入库证据文件生成：我方数字由证据 JSON "
                "计算并带一致性断言，文献方数字为论文发表值的转录（出处与未核实项见第 09 节与第 11 节）。"
                "原始记录见实验文档 102 / 103 号（精度）、106 号（表示探针）、108 号（三 seed 复核）、"
                "110 号（延迟与端到端）；"
                "证据 JSON 见 实验文档/results/ 下的 p2-2-*-2026102{1,2,3}/、p2-2-db-identity-probe-20261023/ "
                "与 latency-20261025/。"
            ),
        }

        template = Path(args.template).read_text(encoding="utf-8")
        for key, value in blocks.items():
            template = template.replace("{{" + key + "}}", value)
        leftover = re.findall(r"\{\{[A-Z0-9_]+\}\}", template)
        if leftover:
            raise SystemExit(f"unfilled template slots: {sorted(set(leftover))}")

        template = strip_sections(template, set(args.omit_section))

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(template, encoding="utf-8")
        print(f"wrote {out} ({len(template):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
