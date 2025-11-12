import hashlib
import json
import os
import pathlib
import sys

DIST = pathlib.Path(os.environ.get("MODELS_DIST", "dist"))
CONTRACT = os.environ.get("MODELS_CONTRACT_VERSION", "v1.4.1")

# Перечисление моделей, которые реально грузит рантайм (NCNN)
MODELS = {
    "zerodcepp_fp16": [
        "models/zerodcepp_fp16.param",
        "models/zerodcepp_fp16.bin",
    ],
    "restormer_fp16": [
        "models/restormer_fp16.param",
        "models/restormer_fp16.bin",
    ],
}


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not (DIST / "models").exists():
        print(f"[ERROR] missing {DIST/'models'}", file=sys.stderr)
        sys.exit(2)

    entries = []
    for mid, rels in MODELS.items():
        files = []
        for rel in rels:
            fp = DIST / rel
            if not fp.exists():
                print(f"[ERROR] missing model file: {fp}", file=sys.stderr)
                sys.exit(2)
            files.append({"path": rel, "sha256": sha256(fp)})
        entries.append({"id": mid, "backend": "ncnn", "files": files})

    sums = DIST / "SHA256SUMS.txt"
    jq_sum = sha256(sums) if sums.exists() else "0" * 64
    entries.append({"id": "metadata", "backend": "METADATA", "jq_sha256sum": jq_sum})

    lock = {"api_contract_version": CONTRACT, "entries": entries}
    out = DIST / "models.lock.json"
    out.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
