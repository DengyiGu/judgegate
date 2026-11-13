"""Record environment versions and fingerprints of source and local experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version

from .data import OUTPUT, ROOT, file_hash


def main():
    sources = [ROOT / "pyproject.toml", *sorted((ROOT / "src").rglob("*.py"))]
    sources += sorted((ROOT / "experiments").glob("*.py"))
    sources += [ROOT / "experiments/config.json"]
    artifacts = sorted((OUTPUT / "runs").rglob("*.trec"))
    artifacts += sorted((OUTPUT / "judgments").glob("*.jsonl"))
    artifacts += sorted((OUTPUT / "simulation").glob("*.json"))
    record = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "versions": {
            name: version(name)
            for name in (
                "numpy",
                "torch",
                "transformers",
                "sentence-transformers",
                "scikit-learn",
                "scipy",
                "huggingface-hub",
            )
        },
        "source_hash_convention": "UTF-8 bytes with CRLF normalized to LF; artifacts use raw bytes",
        "sha256": {
            **{
                path.relative_to(ROOT).as_posix(): hashlib.sha256(
                    path.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                for path in sources
            },
            **{path.relative_to(ROOT).as_posix(): file_hash(path) for path in artifacts},
        },
    }
    target = ROOT / "docs/reports/provenance.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("recorded", len(record["sha256"]), "fingerprints")


if __name__ == "__main__":
    main()
