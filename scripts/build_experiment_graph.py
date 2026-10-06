from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_FILES = 250
MAX_WORDS = 2_000_000
MAX_VIZ_NODES = 5000
CONFIDENCES = {"EXTRACTED", "INFERRED", "AMBIGUOUS"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_id(relative: str) -> str:
    return "artifact_" + hashlib.sha256(relative.encode()).hexdigest()[:24]


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def safe_source(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe source path: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source outside root: {relative}")
    return path


def validate_semantic(root: Path, fragment: dict, documents: list[Path]) -> None:
    hashes = fragment.get("source_hashes", {})
    expected = {str(p.relative_to(root)) for p in documents}
    if set(hashes) != expected:
        raise ValueError(
            f"Semantic coverage mismatch: {sorted(expected ^ set(hashes))}"
        )
    for relative, sha in hashes.items():
        path = safe_source(root, relative)
        if not path.is_file() or digest(path) != sha:
            raise ValueError(
                f"Semantic fragment stale: {relative}; re-extract Markdown"
            )
    for record in fragment.get("nodes", []) + fragment.get("edges", []):
        relative = record.get("source_file", "")
        safe_source(root, relative)
        if record.get("external_reference"):
            if (
                record.get("id") != artifact_id(relative)
                or record.get("entity_type") != "Artifact"
                or record.get("sha256") is not None
                or record.get("status")
                not in (None, "unknown", "external_reference", "referenced_not_read")
            ):
                raise ValueError(
                    f"External reference must be an unverified placeholder: {relative}"
                )
            continue
        if relative not in hashes or record.get("sha256") != hashes[relative]:
            raise ValueError(f"Semantic provenance missing: {relative}")
        if not record.get("source_location"):
            raise ValueError(f"Semantic source location missing: {relative}")


def edge_key(edge: dict) -> str:
    identity = [
        edge.get(k)
        for k in (
            "source",
            "target",
            "relation",
            "source_file",
            "source_location",
            "sha256",
        )
    ]
    return (
        "edge_"
        + hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
    )


def validate_edge(edge: dict, node_ids: set[str]) -> None:
    if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
        raise ValueError(f"Dangling edge: {edge}")
    if not edge.get("relation"):
        raise ValueError("Edge relation missing")
    confidence, score = edge.get("confidence"), edge.get("confidence_score")
    if confidence not in CONFIDENCES or not isinstance(score, (float, int)):
        raise ValueError("Edge confidence invalid")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Edge confidence score outside range")
    if confidence == "EXTRACTED" and score != 1.0:
        raise ValueError("EXTRACTED edge must have confidence_score 1.0")


def merge_fragments(fragments: list[dict]) -> dict:
    nodes, edges = {}, {}
    for fragment in fragments:
        for node in fragment.get("nodes", []):
            if not re.fullmatch(r"[a-z0-9_]+", node.get("id", "")):
                raise ValueError("Node ID must be safe lowercase ASCII")
            previous = nodes.get(node["id"])
            if previous and previous.get("source_file") != node.get("source_file"):
                raise ValueError(f"Node ID collision: {node['id']}")
            if (
                previous
                and not previous.get("external_reference")
                and not node.get("external_reference")
                and previous != node
            ):
                raise ValueError(f"Conflicting node evidence: {node['id']}")
            if not previous or (
                previous.get("external_reference")
                and not node.get("external_reference")
            ):
                nodes = {**nodes, node["id"]: dict(node)}
    for fragment in fragments:
        for edge in fragment.get("edges", []):
            validate_edge(edge, set(nodes))
            key = edge_key(edge)
            if key in edges and edges[key] != dict(edge, id=key):
                raise ValueError(f"Conflicting evidence: {key}")
            edges = {**edges, key: dict(edge, id=key)}
    return {
        "nodes": [nodes[k] for k in sorted(nodes)],
        "edges": [edges[k] for k in sorted(edges)],
        "hyperedges": [h for f in fragments for h in f.get("hyperedges", [])],
        "input_tokens": sum(f.get("input_tokens", 0) for f in fragments),
        "output_tokens": sum(f.get("output_tokens", 0) for f in fragments),
    }


def project_relations(extraction: dict) -> dict:
    pair_counts = Counter((e["source"], e["target"]) for e in extraction["edges"])
    extra_nodes, edges = [], []
    for edge in extraction["edges"]:
        if pair_counts[(edge["source"], edge["target"])] == 1:
            edges.append(dict(edge))
            continue
        relation_id = "relation_" + edge["id"]
        extra_nodes.append(
            {
                "id": relation_id,
                "label": edge["relation"],
                "file_type": "rationale",
                "entity_type": "EvidenceRelation",
                "source_file": edge.get("source_file"),
                "source_location": edge.get("source_location"),
                "sha256": edge.get("sha256"),
                "confidence": edge["confidence"],
                "confidence_score": edge["confidence_score"],
            }
        )
        edges.extend([dict(edge, target=relation_id), dict(edge, source=relation_id)])
    return {
        "nodes": [dict(n) for n in extraction["nodes"]] + extra_nodes,
        "edges": edges,
        "hyperedges": extraction.get("hyperedges", []),
        "directed": True,
    }


def file_fragment(root: Path, files: list[Path]) -> dict:
    return {
        "nodes": [
            {
                "id": artifact_id(str(p.relative_to(root))),
                "label": str(p.relative_to(root)),
                "file_type": "code" if p.suffix == ".py" else "document",
                "entity_type": "Artifact",
                "source_file": str(p.relative_to(root)),
                "source_location": None,
                "sha256": digest(p),
                "status": None,
                "date": None,
                "branch": None,
                "commit": None,
                "validation_state": "recorded_not_reexecuted",
            }
            for p in files
        ],
        "edges": [],
    }


def attach_sources(fragment: dict) -> dict:
    nodes = {n["id"] for n in fragment["nodes"]}
    edges = [
        {
            "source": artifact_id(n["source_file"]),
            "target": n["id"],
            "relation": "documents",
            "confidence": "EXTRACTED",
            "confidence_score": 1.0,
            "weight": 1.0,
            "source_file": n["source_file"],
            "source_location": n.get("source_location"),
            "sha256": n.get("sha256"),
        }
        for n in fragment["nodes"]
        if n.get("source_file")
        and n["id"] != artifact_id(n["source_file"])
        and artifact_id(n["source_file"]) in nodes
    ]
    return dict(fragment, edges=fragment["edges"] + edges)


def ast_fragment(root: Path, paths: list[Path], out: Path) -> dict:
    from graphify.extract import extract

    original = (
        extract(paths, cache_root=out.parent) if paths else {"nodes": [], "edges": []}
    )

    def resolve_source(value: str) -> Path:
        path = Path(value)
        result = path if path.is_absolute() else root / path
        if not result.resolve().is_relative_to(root.resolve()):
            raise ValueError("AST source outside root")
        if not result.is_file():
            raise ValueError("AST source must be a file")
        return result

    mapping = {
        n["id"]: "code_"
        + hashlib.sha256(
            json.dumps(
                [
                    str(resolve_source(n["source_file"]).relative_to(root))
                    if n.get("source_file")
                    else None,
                    n.get("label"),
                    n.get("source_location"),
                ]
            ).encode()
        ).hexdigest()[:32]
        for n in original["nodes"]
    }
    if len(mapping) != len(original["nodes"]):
        raise ValueError("AST node IDs collide; extraction must be file-scoped")

    nodes = [
        dict(
            n,
            id=mapping[n["id"]],
            source_file=str(resolve_source(n["source_file"]).relative_to(root))
            if n.get("source_file")
            else None,
            sha256=digest(resolve_source(n["source_file"]))
            if n.get("source_file")
            else None,
            external_reference=not bool(n.get("source_file")),
            validation_state="static_not_executed"
            if n.get("source_file")
            else "external_unverified",
        )
        for n in original["nodes"]
    ]
    edges = [
        dict(
            e,
            source=mapping[e["source"]],
            target=mapping[e["target"]],
            source_file=str(resolve_source(e["source_file"]).relative_to(root)),
            sha256=digest(resolve_source(e["source_file"])),
            confidence_score=e.get(
                "confidence_score", 1.0 if e.get("confidence") == "EXTRACTED" else 0.75
            ),
        )
        for e in original["edges"]
        if e["source"] in mapping and e["target"] in mapping
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "hyperedges": [],
        "external_edges_omitted": len(original["edges"]) - len(edges),
    }


def normalize_structured(fragment: dict) -> dict:
    nodes = [dict(n, **n.get("metadata", {})) for n in fragment["nodes"]]
    aliases = [
        dict(n, id=artifact_id(n["source_file"]))
        for n in nodes
        if n.get("type") == "artifact"
    ]
    links = [
        {
            "source": artifact_id(n["source_file"]),
            "target": n["id"],
            "relation": "indexes_json",
            "confidence": "EXTRACTED",
            "confidence_score": 1.0,
            "weight": 1.0,
            "source_file": n["source_file"],
            "source_location": "",
            "sha256": n["sha256"],
        }
        for n in nodes
        if n.get("type") == "artifact"
    ]
    return dict(
        fragment,
        nodes=nodes + aliases,
        edges=fragment["edges"] + links,
        input_tokens=0,
        output_tokens=0,
    )


def source_snapshot(root: Path, files: list[Path]) -> dict:
    return {str(p.relative_to(root)): digest(p) for p in files}


def git_snapshot(root: Path) -> dict:
    def read(args: list[str]) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "branch": read(["branch", "--show-current"]),
        "head": read(["rev-parse", "HEAD"]),
        "dirty": bool(read(["status", "--porcelain"])),
        "scope": "capture_only_not_experiment_provenance",
    }


def load_semantic(root: Path, files: list[Path], semantic: Path | None) -> dict:
    if not files:
        return {"nodes": [], "edges": [], "source_hashes": {}}
    if semantic is None:
        raise ValueError(
            "Markdown requires --semantic with content-hash-bound extraction"
        )
    result = json.loads(semantic.read_text(encoding="utf-8"))
    validate_semantic(root, result, files)
    return result


def label_communities(graph: Any, communities: dict) -> dict:
    labels = {}
    for cid, members in communities.items():
        kinds = Counter(
            graph.nodes[n].get(
                "entity_type", graph.nodes[n].get("file_type", "Unknown")
            )
            for n in members
        )
        sources = Counter(
            (graph.nodes[n].get("source_file") or "external").split("/")[0]
            for n in members
        )
        labels[cid] = (
            f"{sources.most_common(1)[0][0]} / {kinds.most_common(1)[0][0]} {cid}"
        )
    return labels


def enhance_html(content: str, extraction: dict) -> str:
    evidence = {n["id"]: n for n in extraction["nodes"]}
    encoded = (
        json.dumps(evidence, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    script = """<script>
const EVIDENCE = __EVIDENCE__;
const originalShowInfo = showInfo;
showInfo = function(id) {
  originalShowInfo(id);
  const data = EVIDENCE[id] || {};
  const panel = document.getElementById('info-content');
  for (const key of ['source_location', 'source_pointer', 'sha256', 'date', 'branch',
                     'commit', 'status', 'validation_state', 'scope', 'value']) {
    if (data[key] === undefined) continue;
    const field = document.createElement('div');
    field.className = 'field';
    field.style.overflowWrap = 'anywhere';
    field.textContent = key + ': ' + JSON.stringify(data[key]);
    panel.appendChild(field);
  }
};
network.setOptions({layout:{randomSeed:42}});
document.addEventListener('click', function(event) {
  const link = event.target.closest('.neighbor-link[data-node-id]');
  if (link) focusNode(link.getAttribute('data-node-id'));
});
</script>""".replace("__EVIDENCE__", encoded)
    style = "<style>#info-panel{overflow-y:auto;max-height:65vh;}#info-content .field{overflow-wrap:anywhere;}@media(max-width:600px){body{flex-direction:column;}#graph{min-height:52vh;}#sidebar{width:100%;height:48vh;}#legend-wrap{min-height:50px;}}</style>"
    content = content.replace("</body>", style + script + "</body>")
    viewport = '<meta name="viewport" content="width=device-width, initial-scale=1">'
    if viewport not in content:
        if "<head>" not in content:
            raise ValueError("graphify HTML has no head element; update enhance_html")
        content = content.replace("<head>", "<head>" + viewport, 1)
    broken = 'onclick="focusNode(${JSON.stringify(nid)})"'
    fixed = 'data-node-id="${esc(nid)}"'
    if broken in content:
        content = content.replace(broken, fixed)
    elif fixed not in content:
        raise ValueError("graphify neighbor-link template changed; update enhance_html")
    return content


def export_outputs(
    extraction: dict, detection: dict, out: Path, root: Path, capture: dict
) -> None:
    from graphify.analyze import god_nodes, suggest_questions, surprising_connections
    from graphify.build import build_from_json
    from graphify.cluster import cluster, score_all
    from graphify.export import to_html, to_json
    from graphify.report import generate

    graph = build_from_json(project_relations(extraction), directed=True)
    if graph.number_of_nodes() > MAX_VIZ_NODES:
        raise ValueError(
            f"Graph exceeds {MAX_VIZ_NODES} visual nodes; narrow the corpus"
        )
    communities = cluster(graph)
    cohesion = score_all(graph, communities)
    labels = label_communities(graph, communities)
    questions = suggest_questions(graph, communities, labels)
    analysis = {
        "communities": {str(k): v for k, v in communities.items()},
        "cohesion": cohesion,
        "labels": labels,
        "gods": god_nodes(graph),
        "surprises": surprising_connections(graph, communities),
        "questions": questions,
    }
    report = generate(
        graph,
        communities,
        cohesion,
        labels,
        analysis["gods"],
        analysis["surprises"],
        detection,
        {"input": extraction["input_tokens"], "output": extraction["output_tokens"]},
        str(root),
        suggested_questions=questions,
    )
    report += audit_appendix(extraction, capture)
    with tempfile.TemporaryDirectory(dir=out) as folder:
        staging = Path(folder)
        if not to_json(graph, communities, str(staging / "graph.json")):
            raise ValueError("graphify JSON export refused")
        categories = {0: [], 1: [], 2: []}
        for nid, data in graph.nodes(data=True):
            category = (
                0
                if data.get("file_type") == "code"
                else 1
                if str(data.get("source_file", "")).endswith(".json")
                else 2
            )
            categories[category].append(nid)
        to_html(
            graph,
            categories,
            str(staging / "graph.html"),
            community_labels={0: "代码结构", 1: "JSON 实验证据", 2: "文档结论与限制"},
        )
        html_path = staging / "graph.html"
        content = enhance_html(
            html_path.read_text(encoding="utf-8"), project_relations(extraction)
        )
        for old, new in zip(
            ("#4E79A7", "#F28E2B", "#E15759"), ("#3987e5", "#d95926", "#199e70")
        ):
            content = content.replace(old, new)
        html_path.write_text(content, encoding="utf-8")
        report = report.replace(
            f"- Token cost: {extraction['input_tokens']:,} input · {extraction['output_tokens']:,} output",
            "- Token cost: input/output split unavailable; extraction is NOT zero-cost. See cost.json.",
        )
        report = re.sub(
            r"## Community Hubs \(Navigation\).*?(?=## )",
            "",
            report,
            flags=re.DOTALL,
        )
        (staging / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
        for name in ("graph.json", "graph.html", "GRAPH_REPORT.md"):
            os.replace(staging / name, out / name)
    atomic_json(out / "analysis.json", analysis)


def audit_appendix(extraction: dict, capture: dict) -> str:
    stats = Counter(
        n.get("entity_type", n.get("file_type", "unknown")) for n in extraction["nodes"]
    )
    return (
        "\n## Evidence Audit\n\n"
        "- All evidence edges: experiment_graph.json. graph.json is a directed display projection.\n"
        "- EXTRACTED means explicitly recorded, not independently verified or re-executed.\n"
        "- INFERRED artifact grouping does not establish that a run produced a result.\n"
        "- Historical claims, planned work and unverified WIP retain their source scope.\n"
        "- Missing experiment date/branch/commit/status remain null; capture Git metadata is not experiment provenance.\n"
        "- Source hashes are checked before and after extraction; stale semantic fragments fail closed.\n"
        "- Excludes raw data, checkpoints, mapping JSONL, logs, histories, manifests and stale snapshots.\n"
        "- This graph cannot certify leakage, causality, GPU state or test success.\n"
        f"- Capture: {json.dumps(capture, ensure_ascii=False)}\n"
        f"- Authority node kinds: {json.dumps(dict(stats), ensure_ascii=False)}\n"
        "- Token accounting: cost.json; unavailable input/output splits are not zero cost.\n\n"
        "## Rebuild\n\n"
        "Run the same command with --root, --out and --semantic. Re-extract Markdown if source hashes change; "
        "rebuild rather than graphify --update to remove obsolete evidence. No watchers, hooks or MCP installed.\n"
    )


def build(root: Path, out: Path, semantic: Path | None) -> dict:
    try:
        from scripts.experiment_evidence import extract_json, select_files
    except ModuleNotFoundError:
        from experiment_evidence import extract_json, select_files

    root, out = root.resolve(), out.resolve()
    workspace = Path(__file__).resolve().parents[1]
    if out == workspace or not out.is_relative_to(workspace):
        raise ValueError("Output must be a subdirectory of this tool worktree")
    if out == root:
        raise ValueError("Output cannot be the source root")
    selected = select_files(root)
    files = sorted({p for group in selected.values() for p in group})
    words = sum(len(p.read_text(encoding="utf-8").split()) for p in files)
    if len(files) > MAX_FILES or words > MAX_WORDS:
        groups = Counter(str(p.relative_to(root)).split("/")[0] for p in files)
        raise ValueError(
            f"Corpus exceeds limit: {len(files)} files, {words} words; narrow scope: {groups.most_common(5)}"
        )
    before = source_snapshot(root, files)
    capture = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "git": git_snapshot(root),
    }
    out.mkdir(parents=True, exist_ok=True)
    sem = load_semantic(root, selected["document"], semantic)
    ast = ast_fragment(root, selected["code"], out)
    structured = normalize_structured(extract_json(root, selected["json"]))
    known_ids = {n["id"] for n in structured["nodes"]}
    file_nodes = file_fragment(root, files)
    file_nodes = dict(
        file_nodes, nodes=[n for n in file_nodes["nodes"] if n["id"] not in known_ids]
    )
    extraction = merge_fragments(
        [attach_sources(merge_fragments([structured, file_nodes, ast, sem]))]
    )
    if source_snapshot(root, files) != before:
        raise ValueError("Source changed during build; retry from a consistent capture")
    detection = {
        "total_files": len(files),
        "total_words": words,
        "warning": "Selected evidence corpus; whitespace word counts undercount Chinese text.",
        "files": {key: [str(p) for p in paths] for key, paths in selected.items()},
    }
    export_outputs(extraction, detection, out, root, capture)
    atomic_json(
        out / "experiment_graph.json",
        dict(
            extraction,
            schema=1,
            capture=capture,
            evidence_policy="recorded_not_reexecuted",
        ),
    )
    atomic_json(
        out / "manifest.json",
        {"sources": before, "capture": capture, "selected": detection["files"]},
    )
    atomic_json(
        out / "cost.json",
        {
            "note": (
                "Semantic Markdown extraction ran as in-session agents. Input/output "
                "token splits were not reported and are recorded as null, not zero. "
                "deterministic_tokens is a UTF-8/4 estimate for JSON parsing only and "
                "is not a model tokenizer measurement."
            ),
            "model_input_tokens": None,
            "model_output_tokens": None,
            "deterministic_tokens_estimate": extraction.get("input_tokens", 0),
            "files": len(files),
            "words_whitespace_estimate": words,
        },
    )
    logging.getLogger(__name__).info(
        "Built %s files, %s nodes, %s evidence edges",
        len(files),
        len(extraction["nodes"]),
        len(extraction["edges"]),
    )
    return extraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a hash-bound experimental evidence index"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--semantic", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        build(args.root, args.out, args.semantic)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Evidence build failed: {error}\n")


if __name__ == "__main__":
    main()
