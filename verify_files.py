"""Verify the published file manifest: python verify_files.py."""
from pathlib import Path
import hashlib
import json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
for name,expected in manifest.items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=expected:
        raise SystemExit(f'Hash mismatch: {name}')
print(f'Verified {len(manifest)} files')
