#!/usr/bin/env python3
"""Download and convert BLINK checkpoints to this project's format.

BLINK (Wu et al. 2020, facebookresearch/BLINK) uses BERT-large and stores
weights under different key names. This script:
  1. Downloads biencoder_wiki_large.bin and crossencoder_wiki_large.bin.
  2. Remaps state dict keys to match our BiEncoder / CrossEncoder classes.
  3. Saves {"model": state_dict, "epoch": 0} .pth files into models/.

Key mapping
-----------
Biencoder:
  context_encoder.bert_model.*  →  mention_encoder.bert.*
  cand_encoder.bert_model.*     →  entity_encoder.bert.*

Crossencoder:
  encoder.bert_model.*          →  encoder.bert.*

After running, update conf/eval.yaml:
  backends:
    biencoder:
      model_name: bert-large-uncased
      checkpoint: /path/to/models/biencoder_blink.pth
    crossencoder:
      model_name: bert-large-uncased
      biencoder_checkpoint: /path/to/models/biencoder_blink.pth
      checkpoint: /path/to/models/crossencoder_blink.pth
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import torch

_URLS = {
    "biencoder":    "https://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.bin",
    "crossencoder": "https://dl.fbaipublicfiles.com/BLINK/crossencoder_wiki_large.bin",
}

_KEY_MAPS = {
    "biencoder": [
        ("context_encoder.bert_model.", "mention_encoder.bert."),
        ("cand_encoder.bert_model.",    "entity_encoder.bert."),
    ],
    "crossencoder": [
        ("encoder.bert_model.", "encoder.bert."),
    ],
}


def _remap(state_dict: dict, key_map: list[tuple[str, str]]) -> dict:
    out: dict = {}
    skipped: list[str] = []
    for k, v in state_dict.items():
        mapped = False
        for src, dst in key_map:
            if k.startswith(src):
                out[dst + k[len(src):]] = v
                mapped = True
                break
        if not mapped:
            skipped.append(k)
    if skipped:
        print(f"  Skipped {len(skipped)} unrecognised keys (not part of BERT body): "
              f"{skipped[:3]}{'...' if len(skipped) > 3 else ''}", file=sys.stderr)
    return out


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(block, block_size, total):
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            print(f"\r  {pct}%", end="", flush=True, file=sys.stderr)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(file=sys.stderr)


def convert(models_dir: Path, raw_dir: Path, skip_download: bool) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, url in _URLS.items():
        raw_path = raw_dir / Path(url).name

        if not raw_path.exists():
            if skip_download:
                print(f"[{name}] Skipping — place {Path(url).name} in {raw_dir} and re-run.",
                      file=sys.stderr)
                continue
            _download(url, raw_path)

        print(f"[{name}] Converting ...", file=sys.stderr)
        raw = torch.load(raw_path, map_location="cpu")

        # BLINK checkpoints may be wrapped in {"state_dict": ...} or be flat
        if isinstance(raw, dict) and "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw

        # Strip DataParallel "module." prefix if present
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        converted = _remap(state_dict, _KEY_MAPS[name])

        if not converted:
            print(f"[{name}] ERROR: no keys were mapped — BLINK key format may have changed.",
                  file=sys.stderr)
            print(f"  Sample raw keys: {list(state_dict.keys())[:5]}", file=sys.stderr)
            sys.exit(1)

        out_path = models_dir / f"{name}_blink.pth"
        torch.save({"model": converted, "epoch": 0}, out_path)
        print(f"[{name}] Saved → {out_path}  ({len(converted)} tensors)", file=sys.stderr)

    print("\nUpdate conf/eval.yaml to use these checkpoints with model_name: bert-large-uncased",
          file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", default="models", type=Path,
                        help="Directory to write converted .pth files (default: models/)")
    parser.add_argument("--raw-dir", default=None, type=Path,
                        help="Directory to cache raw BLINK downloads "
                             "(default: <models-dir>/_blink_raw)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading; use already-present raw files")
    args = parser.parse_args()

    raw_dir = args.raw_dir or (args.models_dir / "_blink_raw")
    convert(args.models_dir, raw_dir, args.skip_download)
