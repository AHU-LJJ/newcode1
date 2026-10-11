"""Download the pinned official BGE-M3 checkpoint into the project workspace."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=ROOT / "models/bge-m3")
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    info = HfApi().model_info("BAAI/bge-m3", revision=args.revision)
    files = {item.rfilename for item in info.siblings}
    weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    if weights not in files:
        raise RuntimeError("Expected official single-file BGE-M3 checkpoint not found")
    patterns = [weights, "config.json", "config_sentence_transformers.json", "modules.json",
                "sentence_bert_config.json", "sentencepiece.bpe.model", "special_tokens_map.json",
                "tokenizer.json", "tokenizer_config.json", "1_Pooling/config.json"]
    print(f"Downloading BAAI/bge-m3 revision {info.sha} to {args.path}", flush=True)
    snapshot_download("BAAI/bge-m3", revision=info.sha, local_dir=args.path,
                      cache_dir=ROOT / ".cache/huggingface", allow_patterns=patterns, max_workers=3)
    metadata = args.path / "source_revision.json"
    temporary = metadata.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(repo_id="BAAI/bge-m3", revision=info.sha), indent=2) + "\n")
    temporary.replace(metadata)
    print("Model and tokenizer ready", flush=True)


if __name__ == "__main__":
    main()
