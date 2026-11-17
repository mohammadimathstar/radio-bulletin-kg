"""General I/O helpers shared by the pipeline modules (clustering, linking, profiles).

Deliberately small and dependency-free: everything here must be importable from
any module, which is why it lives in ``utils/``.  Loading of the IE/OCR/cluster
files themselves stays in the packages that own those formats
(``entity_resolution.io``, ``linking.clusters``).
"""
import json
from pathlib import Path


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file; a truncated last line (crash mid-write) is ignored."""
    p, out = Path(path), []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def append_jsonl(fh, obj: dict) -> None:
    """Write one JSON line and flush so the file can be read while the script runs."""
    fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    fh.flush()


def write_jsonl(records, path: str) -> None:
    """Write records as JSONL (one JSON object per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def write_json(data, path: str) -> None:
    """Write data as pretty JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)