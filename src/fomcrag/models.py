"""Model registry and a resumable downloader.

Models load from ./models/<name> when present, else from the Hugging Face Hub by id. Where huggingface.co is
unreachable (it is blocked on some networks), fetch the same files from ModelScope:

    python -m fomcrag.models --source modelscope            # embedder + reranker (~1.6 GB)
    python -m fomcrag.models --source modelscope --llm      # + local generator (~8 GB)

Only safetensors weights and tokenizer/config files are fetched: nothing that executes code on load.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
EMBEDDER = "BAAI/bge-base-en-v1.5"
RERANKER = "BAAI/bge-reranker-base"
GENERATOR = "Qwen/Qwen3-4B-Instruct-2507"
SAFE_SUFFIXES = (".json", ".txt", ".safetensors", ".model")
SOURCES = {
    "hf": ("https://huggingface.co/api/models/{repo}/tree/main?recursive=true", "https://huggingface.co/{repo}/resolve/main/{path}"),
    "modelscope": ("https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true",
                   "https://www.modelscope.cn/models/{repo}/resolve/master/{path}"),
}


def resolve(repo: str) -> str:
    local = MODELS_DIR / repo.split("/")[-1]
    return str(local) if (local / "config.json").exists() else repo


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fomc-rag"}), timeout=60) as r:
        return r.read()


def list_files(repo: str, source: str) -> list[tuple[str, int]]:
    data = json.loads(_get(SOURCES[source][0].format(repo=repo)))
    if source == "modelscope":
        return [(f["Path"], f["Size"]) for f in data["Data"]["Files"] if f["Type"] == "blob"]
    return [(f["path"], f["size"]) for f in data if f["type"] == "file"]


def download(url: str, dest: Path, size: int, retries: int = 30) -> None:
    """Resume with HTTP Range until the file has exactly the size the hub lists.

    Mirrors drop long connections without raising, so a finished read() is not proof of a complete file.
    """
    part = dest.with_name(dest.name + ".part")
    if dest.exists():
        if dest.stat().st_size == size:
            return
        dest.replace(part)  # truncated by an earlier run: resume it
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fomc-rag", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=120) as r, part.open("ab" if r.status == 206 else "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except OSError as e:
            print(f"    retry {attempt + 1}: {e}")
            time.sleep(min(2 ** attempt, 30))
    if not part.exists() or part.stat().st_size != size:
        raise RuntimeError(f"{dest}: incomplete after {retries} attempts")
    part.replace(dest)


def fetch(repo: str, source: str) -> Path:
    out = MODELS_DIR / repo.split("/")[-1]
    for path, size in list_files(repo, source):
        if path.endswith(SAFE_SUFFIXES) and not path.startswith(("onnx/", "openvino/")) and path != "configuration.json":
            download(SOURCES[source][1].format(repo=repo, path=path), out / path, size)
            print(f"  {repo}/{path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="hf")
    ap.add_argument("--llm", action="store_true", help="also fetch the local generator model")
    args = ap.parse_args()
    for repo in [EMBEDDER, RERANKER] + ([GENERATOR] if args.llm else []):
        print(repo, "->", fetch(repo, args.source))


if __name__ == "__main__":
    main()
