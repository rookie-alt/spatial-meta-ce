import hashlib
import json
import math
from pathlib import Path

import numpy as np

TRAIN = (
    "accidents",
    "airline",
    "baseball",
    "basketball",
    "carcinogenesis",
    "ccs",
    "consumer",
    "credit",
    "employee",
    "financial",
    "fnhk",
    "grants",
    "hepatitis",
    "hockey",
    "movielens",
    "sakila",
    "sap",
    "seznam",
    "ssb",
    "talkingdata",
    "tournament",
    "tpc_h",
    "tubepricing",
)
VALIDATION = ("chembl", "legalacts", "telstra")
TARGETS = ("imdb", "stats", "genome", "ergastf1")
ARMS = ("A_w64_smooth", "B_w64_mse", "C_w128_smooth", "D_w128_mse")
CACHE_FILES = ("manifest.json", "cards.npy", "offsets.npy")
FORMAL = "实验文档/results/precision-screen-20261006/"
S1 = "实验文档/results/s1-source-diagnosis-20261006-verified/"
EXPECTED_DIGEST = "ad433d964fe562efacd19c8a03005bfe7df530d96bed0ca3535684854f6f4de8"
FROZEN = {
    "configs/data.json": "a9057d65c7cb220bd21e7cd67dacbc15c0c20dcc87f61d9b3e6470c0ea094d90",
    "configs/precision_screen.json": "cd401ea8b361c6e8b003ddae24c5a743b9768b04efe52db1503f7346f5ad5112",
    "实验文档/16-S2源训练曝光与残差监督分布诊断方案.md": "ea67591a9ee971182c5cb40377a0a994fada3a110600da3fa6828e409e8bbb4f",
    FORMAL
    + "summary.json": "3fd40103ba5ce19c93c3fa7861a5012c902a4237da5ba7502b5add7b04eae78a",
    FORMAL
    + "status.json": "eb535f744e7006fdbf9c6c031314bbda2936d1a12b6d4d801d791431822fcc99",
    FORMAL
    + "preflight.json": "3b037edd777a7e39259033b020ffff8015cbb4cd6bd169ab8453b16bc5d8bfe8",
    FORMAL
    + "verification.json": "d65973e918e1c40321f25850bbaa5cba5699dfb36bab02d5fe4ea36f632bea91",
    S1
    + "summary.json": "d4adf1d1a2908d8093122b1020c8172233ccf19e6d63eb4b52e99da1d06e2199",
    S1
    + "verification.json": "6e28ca3ba3f1a33674835f8a59ee3e79fe4503c4d97c9fa31a67ee03d8f4c7ed",
    S1
    + "strata.csv": "52b29a6542e72dbaa0a7d60d3fe761337efa512638c2d18bff79559cf7c873d6",
}


class ReplayMismatch(ValueError):
    pass


def safe(root: Path, path: Path) -> Path:
    if any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)):
        raise ValueError("Symlink paths are forbidden")
    if ".." in path.parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("S2 path escapes frozen root")
    return path


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_contract(root: Path) -> dict:
    for name, sha in FROZEN.items():
        if file_hash(safe(root, root / name)) != sha:
            raise ValueError(f"Frozen evidence bytes changed: {name}")
    data = json.loads((root / "configs/data.json").read_text())
    if data["splits"] != {
        "train": list(TRAIN),
        "validation": list(VALIDATION),
        "test": list(TARGETS),
    }:
        raise ValueError("Frozen database order changed")
    formal = json.loads((root / (FORMAL + "summary.json")).read_text())
    preflight = json.loads((root / (FORMAL + "preflight.json")).read_text())
    status = json.loads((root / (FORMAL + "status.json")).read_text())
    if (
        status["status"] != "completed"
        or status["smoke"]
        or preflight["numpy"] != "1.26.1"
    ):
        raise ValueError("Formal execution not verified")
    if [r["arm"] for r in formal["runs"]] != list(ARMS):
        raise ValueError("Frozen arms changed")
    shared = {
        k: v
        for k, v in formal["runs"][0]["config"].items()
        if k not in ("loss", "model")
    }
    expected_counts = {
        "optimizer_steps": 4000,
        "backward_calls": 16000,
        "query_presentations": 1621915,
        "support_presentations": 0,
    }
    for run in formal["runs"]:
        if (
            {k: v for k, v in run["config"].items() if k not in ("loss", "model")}
            != shared
            or run["counts"] != expected_counts
            or run["sample_plan_digest"] != EXPECTED_DIGEST
            or run["sample_plan_episodes"] != 16000
            or run["optimizer_steps"] != 4000
        ):
            raise ValueError("Four arms do not share frozen sampling")
        if run["manifest_hashes"] != {
            db: preflight["sources"][db]["manifest_hash"]
            for db in (*TRAIN, *VALIDATION)
        }:
            raise ValueError("Formal source identity changed")
    if any(
        shared[k] != v
        for k, v in {
            "method": "erm",
            "seed": 42,
            "steps": 4000,
            "meta_batch": 4,
            "accumulation": 1,
            "query_size": 64,
            "support_sizes": [5, 10, 20, 50, 100],
        }.items()
    ):
        raise ValueError("Frozen budget changed")
    return {
        "train": list(TRAIN),
        "validation": list(VALIDATION),
        "config": formal["runs"][0]["config"],
        "sources": preflight["sources"],
        "input_sha256": dict(FROZEN),
    }


def snapshot(root: Path, contract: dict) -> dict:
    expected = {
        **contract["input_sha256"],
        **{
            f"artifacts/cache/pretrain/{db}/{name}": contract["sources"][db][
                "file_sha256"
            ][name]
            for db in (*TRAIN, *VALIDATION)
            for name in CACHE_FILES
        },
    }
    for relative, sha in expected.items():
        if file_hash(safe(root, root / relative)) != sha:
            raise ValueError(f"Frozen input bytes changed: {relative}")
    return expected


def read_cache(root: Path, db: str, frozen: dict) -> dict:
    if db not in (*TRAIN, *VALIDATION):
        raise ValueError("S2 only permits frozen source databases")
    directory = safe(root, root / "artifacts/cache/pretrain" / db)
    for name in CACHE_FILES:
        if file_hash(safe(root, directory / name)) != frozen["file_sha256"][name]:
            raise ValueError(f"Frozen cache bytes changed: {db}/{name}")
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("content_hash") != frozen["manifest_hash"]
        or digest({k: v for k, v in manifest.items() if k != "content_hash"})
        != frozen["manifest_hash"]
        or manifest.get("schema") != 1
        or manifest.get("transform") != "price40-group-logfanout-v1"
        or manifest.get("provenance", {}).get("excluded")
    ):
        raise ValueError("Invalid frozen manifest identity")
    rows = manifest["rows"]
    cards = np.load(directory / "cards.npy", allow_pickle=False)
    offsets = np.load(directory / "offsets.npy", allow_pickle=False)
    if (
        not rows
        or len(rows) != frozen["rows"]
        or cards.shape != (len(rows), 2)
        or offsets.shape != (len(rows) + 1, 4)
    ):
        raise ValueError("Cache shape mismatch")
    if (
        cards.dtype.kind not in "fiu"
        or not np.isfinite(cards).all()
        or (cards < 0).any()
    ):
        raise ValueError("Invalid cardinalities")
    if offsets.dtype.kind not in "iu" or (offsets < 0).any() or (offsets[0] != 0).any():
        raise ValueError("Invalid offsets")
    if (offsets[1:] < offsets[:-1]).any():
        raise ValueError("Nonmonotonic offsets")
    tables = offsets[1:, 2].astype(np.int64) - offsets[:-1, 2].astype(np.int64)
    if (tables < 1).any():
        raise ValueError("Invalid table counts")
    keys = [row["query_hash"] for row in rows]
    if any(not isinstance(h, str) or not h for h in keys) or any(
        not row["id"].startswith(f"{db}/pretrain/") for row in rows
    ):
        raise ValueError("Invalid query identity")
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    if not np.array_equal(cards, cards[first[inverse]]) or not np.array_equal(
        tables, tables[first[inverse]]
    ):
        raise ValueError("Duplicate identity has conflicting true, PG or tables")
    indices = np.sort(first)
    if len(indices) != manifest["unique_queries"]:
        raise ValueError("Unique denominator mismatch")
    selected = cards[indices].astype(np.float64)
    return {
        "true": selected[:, 0],
        "pg": selected[:, 1],
        "indices": indices.astype(np.int64),
        "query_hash": [keys[i] for i in indices],
        "n_tables": tables[indices],
        "raw_rows": len(rows),
    }


def advance(
    state: dict, selected: np.ndarray, k: int, query_size: int, batch: int
) -> dict:
    n = len(state["support"])
    si = np.bincount(selected[:k], minlength=n)
    qi = np.bincount(selected[k:], minlength=n)
    return {
        "support": state["support"] + si,
        "query": state["query"] + qi,
        "mass": state["mass"] + (si + qi) / (batch * (k + query_size)),
        "episodes": state["episodes"] + 1,
        "k_counts": {**state["k_counts"], k: state["k_counts"][k] + 1},
    }


def replay(
    caches: dict, train: list, config: dict, prefixes: tuple = (1000, 2000, 4000)
) -> dict:
    batch, query_size = config["meta_batch"], config["query_size"]
    if (
        config["accumulation"] != 1
        or batch < 1
        or query_size < 1
        or not train
        or len(set(train)) != len(train)
    ):
        raise ValueError("Invalid replay sampling configuration")
    for db in train:
        indices = caches[db]["indices"]
        if (
            len(indices) < max(config["support_sizes"]) + query_size
            or (np.diff(indices) <= 0).any()
        ):
            raise ValueError("Insufficient first-occurrence pool")
    rng = np.random.default_rng(config["seed"] + 1)
    previous = hashlib.sha256(b"").hexdigest()
    states = {
        db: {
            "support": np.zeros(len(caches[db]["indices"]), dtype=np.int64),
            "query": np.zeros(len(caches[db]["indices"]), dtype=np.int64),
            "mass": np.zeros(len(caches[db]["indices"]), dtype=np.float64),
            "episodes": 0,
            "k_counts": {k: 0 for k in config["support_sizes"]},
        }
        for db in train
    }
    checkpoints, presentations = {}, 0
    for number in range(1, config["steps"] * batch + 1):
        db = str(rng.choice(train))
        k = int(rng.choice(config["support_sizes"]))
        raw = rng.choice(caches[db]["indices"], size=k + query_size, replace=False)
        chosen = np.searchsorted(caches[db]["indices"], raw)
        hashes = caches[db]["query_hash"]
        episode = {
            "db": db,
            "support": [hashes[i] for i in chosen[:k]],
            "query": [hashes[i] for i in chosen[k:]],
        }
        previous = digest({"previous": previous, "episode": episode})
        states = {**states, db: advance(states[db], chosen, k, query_size, batch)}
        presentations += k + query_size
        if number % batch == 0 and number // batch in prefixes:
            checkpoints = {
                **checkpoints,
                number // batch: {
                    "databases": dict(states),
                    "digest": previous,
                    "episodes": number,
                    "presentations": presentations,
                },
            }
    return {"digest": previous, "prefixes": checkpoints}


def certify(result: dict) -> None:
    final = result["prefixes"][4000]
    if (
        result["digest"] != EXPECTED_DIGEST
        or final["episodes"] != 16000
        or final["presentations"] != 1621915
    ):
        raise ReplayMismatch("S2 public sampling digest or counts did not match")
    if not math.isclose(
        math.fsum(float(v["mass"].sum()) for v in final["databases"].values()),
        4000,
        rel_tol=0,
        abs_tol=1e-8,
    ):
        raise ReplayMismatch("S2 nominal coefficient conservation failed")
