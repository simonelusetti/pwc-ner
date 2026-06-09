"""Pipeline comparison entry point.  Called by forge: ``forge -M eval --config-name eval run``."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

import forge

from pipeline.backends.autoregressive import AutoregressiveBackend
from pipeline.backends.biencoder import BiEncoderBackend
from pipeline.backends.crossencoder import CrossEncoderBackend
from pipeline.run import load_examples, run_backend


def _build_backend(name: str, cfg: DictConfig, device: str):
    if name == "biencoder":
        checkpoint = OmegaConf.select(cfg, "checkpoint")
        if not checkpoint:
            raise ValueError("biencoder backend requires backends.biencoder.checkpoint=<path>")
        return BiEncoderBackend(
            checkpoint_path=checkpoint,
            model_name=str(cfg.get("model_name", "bert-base-uncased")),
            device=device,
            batch_size=cfg["batch_size"],
            max_mention_len=int(cfg.get("max_mention_len", 128)),
            max_entity_len=int(cfg.get("max_entity_len", 128)),
            retrieval_k=int(cfg.get("retrieval_k", 64)),
            tqdm_disabled=not sys.stderr.isatty(),
        )

    if name == "crossencoder":
        bi_ckpt    = OmegaConf.select(cfg, "biencoder_checkpoint")
        cross_ckpt = OmegaConf.select(cfg, "checkpoint")
        if not bi_ckpt or not cross_ckpt:
            raise ValueError(
                "crossencoder backend requires both biencoder_checkpoint and checkpoint"
            )
        return CrossEncoderBackend(
            biencoder_checkpoint=bi_ckpt,
            crossencoder_checkpoint=cross_ckpt,
            model_name=str(cfg.get("model_name", "bert-base-uncased")),
            device=device,
            batch_size=int(cfg.get("batch_size", 32)),
            max_mention_len=int(cfg.get("max_mention_len", 128)),
            max_entity_len=int(cfg.get("max_entity_len", 128)),
            max_cross_len=int(cfg.get("max_cross_len", 256)),
            retrieval_k=int(cfg.get("retrieval_k", 64)),
            tqdm_disabled=not sys.stderr.isatty(),
        )

    if name == "autoregressive":
        checkpoint = OmegaConf.select(cfg, "checkpoint")
        if not checkpoint:
            raise ValueError("autoregressive backend requires backends.autoregressive.checkpoint=<path>")
        return AutoregressiveBackend(
            checkpoint_path=checkpoint,
            model_name=str(cfg.get("model_name", "facebook/bart-base")),
            device=device,
            batch_size=int(cfg.get("batch_size", 8)),
            max_input_len=int(cfg.get("max_input_len", 128)),
            max_title_len=int(cfg.get("max_title_len", 32)),
            num_beams=int(cfg.get("num_beams", 5)),
            tqdm_disabled=not sys.stderr.isatty(),
        )

    raise ValueError(f"Unknown backend: {name!r}")


def main(cfg: DictConfig) -> None:
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available — falling back to CPU.", file=sys.stderr)

    run = forge.start_run(cfg)

    data_dir = Path(cfg.data_dir)
    dataset  = str(cfg.get("dataset", "zeshel"))
    splits   = list(cfg.get("splits", ["testa", "testb"]))
    subset   = cfg.get("subset", None)

    examples = load_examples(data_dir, dataset=dataset, splits=splits, subset=subset)
    print(f"Loaded {len(examples)} examples (dataset={dataset!r}, splits={splits})", file=sys.stderr)

    all_metrics: dict[str, dict] = {}

    for backend_name, backend_cfg in cfg.backends.items():
        if not backend_cfg.get("enabled", True):
            print(f"Skipping {backend_name} (disabled)", file=sys.stderr)
            continue

        print(f"Running backend: {backend_name}", file=sys.stderr)
        result = run_backend(examples, _build_backend(backend_name, backend_cfg, device))
        all_metrics[backend_name] = result.metrics

        m = result.metrics
        print(
            f"  {backend_name}: recall@64={m['recall_at_64']:.4f}  "
            f"norm_acc={m['normalised_accuracy']:.4f}  "
            f"unnorm_acc={m['unnormalised_accuracy']:.4f}",
            file=sys.stderr,
        )

        Path(f"predictions_{backend_name}.json").write_text(
            json.dumps({
                "backend": backend_name,
                "metrics": m,
                "predictions": [
                    {
                        "mention_id":    r.example.mention.mention_id,
                        "gold":          r.example.mention.gold_entity_id,
                        "prediction":    r.prediction,
                        "retrieved_top10": r.retrieved[:10],
                        "category":      r.example.mention.category,
                    }
                    for r in result.results
                ],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    run.finish(metrics={
        f"{k}_recall64": v["recall_at_64"] for k, v in all_metrics.items()
    })

    header = f"\n{'backend':<16}  {'recall@1':>8}  {'recall@10':>9}  {'recall@64':>9}  {'recall@100':>10}  {'norm_acc':>8}  {'unnorm_acc':>10}"
    print(header, file=sys.stderr)
    for name, m in all_metrics.items():
        print(
            f"{name:<16}  {m['recall_at_1']:>8.4f}  {m['recall_at_10']:>9.4f}  "
            f"{m['recall_at_64']:>9.4f}  {m['recall_at_100']:>10.4f}  "
            f"{m['normalised_accuracy']:>8.4f}  {m['unnormalised_accuracy']:>10.4f}",
            file=sys.stderr,
        )
