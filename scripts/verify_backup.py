"""Check exported bytes and structured data against the backup manifest."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "BACKUP_MANIFEST.json").read_text())
errors = []
for entry in manifest["files"]:
    path = ROOT / entry["path"]
    if not path.is_file():
        errors.append("missing: " + entry["path"])
        continue
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["export_sha256"]:
        errors.append("hash mismatch: " + entry["path"])
    try:
        if path.suffix == ".json":
            json.loads(data)
        elif path.suffix == ".jsonl":
            for line in data.decode().splitlines():
                if line.strip():
                    json.loads(line)
    except (ValueError, UnicodeDecodeError) as exc:
        errors.append("parse error: " + entry["path"] + " " + str(exc))
if errors:
    raise SystemExit("\n".join(errors))
print(f"Verified {len(manifest['files'])} files: hashes and JSON/JSONL syntax.")
