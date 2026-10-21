"""Certificate serialization helpers."""
import json
from pathlib import Path

SCHEMA = "bl4_uniform_radial_v2"


def write_certificate(path, certificate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8")


def read_certificate(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA:
        raise ValueError("unsupported certificate schema")
    return data

