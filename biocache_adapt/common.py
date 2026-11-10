"""BioCache 适配器的零重依赖基础设施。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional


def object_nbytes(value: Any, seen: Optional[set[int]] = None) -> int:
    """递归计算实际字节；tensor 按底层 storage 计，避免共享 storage 重复计数。"""
    if value is None:
        return 0
    seen = seen if seen is not None else set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if hasattr(value, "untyped_storage"):
        try:
            storage = value.untyped_storage()
            if id(storage) in seen:
                return 0
            seen.add(id(storage))
            return int(storage.nbytes())
        except Exception:
            pass
    if hasattr(value, "nbytes") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return int(value.nbytes)
        except Exception:
            pass
    if isinstance(value, Mapping):
        return sys.getsizeof(value) + sum(
            object_nbytes(k, seen) + object_nbytes(v, seen) for k, v in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return sys.getsizeof(value) + sum(object_nbytes(item, seen) for item in value)
    if is_dataclass(value):
        return sys.getsizeof(value) + sum(
            object_nbytes(getattr(value, field.name), seen) for field in fields(value)
        )
    if hasattr(value, "__dict__"):
        return sys.getsizeof(value) + object_nbytes(vars(value), seen)
    return sys.getsizeof(value)


def describe_tree(value: Any, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()

    def visit(obj: Any, path: str) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, "shape") and (hasattr(obj, "element_size") or hasattr(obj, "nbytes")):
            rows.append({"path": path, "bytes": object_nbytes(obj),
                         "shape": list(getattr(obj, "shape", ())),
                         "dtype": str(getattr(obj, "dtype", "unknown")),
                         "device": str(getattr(obj, "device", "cpu"))})
        elif isinstance(obj, Mapping):
            for key, item in obj.items():
                visit(item, f"{path}.{key}")
        elif isinstance(obj, (list, tuple)):
            for index, item in enumerate(obj):
                visit(item, f"{path}[{index}]")
        elif is_dataclass(obj):
            for field in fields(obj):
                visit(getattr(obj, field.name), f"{path}.{field.name}")
        elif hasattr(obj, "__dict__"):
            visit(vars(obj), path)
        else:
            rows.append({"path": path, "bytes": object_nbytes(obj), "type": type(obj).__name__})

    visit(value, prefix)
    return rows


@dataclass(frozen=True)
class TimestampedFrame:
    timestamp: float
    frame: Any


class PhaseTimer:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextmanager
    def measure(self, phase: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.totals[phase] = self.totals.get(phase, 0.0) + time.perf_counter() - start
            self.counts[phase] = self.counts.get(phase, 0) + 1

    def report(self) -> dict[str, dict[str, float | int]]:
        return {key: {"seconds": value, "calls": self.counts[key]} for key, value in self.totals.items()}


BUDGET_TIERS = {
    "12.5%": {"ratio": 0.125, "description": "aggressive"},
    "25%": {"ratio": 0.25, "description": "balanced"},
    "50%": {"ratio": 0.50, "description": "quality"},
}


def iter_benchmark(path: str, benchmark: str) -> Iterator[dict[str, Any]]:
    supported = {"ovo", "streamingbench", "twinpast", "vnbench", "egolifeqa"}
    name = benchmark.lower().replace("-", "").replace("_", "")
    if name not in supported:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records: Iterable[Any] = (
            json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    else:
        raw = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            records = next((raw[k] for k in ("data", "samples", "questions", "annotations")
                            if isinstance(raw.get(k), list)), [raw])
        else:
            raise ValueError("benchmark JSON root must be object or array")
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"record {index} is not an object")
        pick = lambda keys: next((row.get(k) for k in keys if row.get(k) is not None), None)
        yield {
            "benchmark": benchmark,
            "sample_id": str(pick(("id", "uid", "question_id")) or index),
            "video": pick(("video", "video_path", "video_id", "clip")),
            "timestamp": pick(("timestamp", "time", "question_time", "ts")),
            "question": pick(("question", "query", "q", "text")),
            "answer": pick(("answer", "label", "gt")),
            "options": pick(("options", "choices", "candidates")),
            "raw": row,
        }


def write_jsonl(path: str, records: Iterable[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, default=str) + "\n")


def write_manifest(run_dir: str, config: Mapping[str, Any], repo_root: str) -> str:
    def git(*args: str) -> Optional[str]:
        try:
            return subprocess.run(["git", "-C", repo_root, *args], capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((Path(repo_root) / "biocache_adapt").glob("*.py"))}
    manifest = {
        "schema_version": "biocache-adapt/v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv, "python": sys.version, "config": dict(config),
        "git": {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "dirty": bool(git("status", "--porcelain"))},
        "source_sha256": hashes,
    }
    os.makedirs(run_dir, exist_ok=True)
    output = os.path.join(run_dir, "manifest.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output
