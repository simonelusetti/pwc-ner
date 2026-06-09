from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from prettytable import PrettyTable

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import forge

from zeshel.biencoder import BiEncoder, in_batch_loss
from zeshel.autoregressive import AutoregressiveEL
from zeshel.data import (
    build_biencoder_loaders,
    build_entity_loader,
    build_seq2seq_loader,
    load_zeshel,
    load_tackbp,
)
from zeshel.encoder import build_tokenizer
from zeshel.index import EntityIndex
from zeshel.metrics import compute_retrieval_metrics


log = logging.getLogger("forge")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(device: str, batch: dict) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _load_data(cfg: DictConfig):
    data_dir = Path(cfg.data.data_dir)
    dataset  = str(cfg.data.get("dataset", "zeshel"))
    splits   = list(cfg.data.get("splits", ["train", "valid"]))
    subset       = cfg.data.get("subset", None)
    full_entities = bool(cfg.data.get("full_entities", True))

    if dataset == "zeshel":
        return load_zeshel(data_dir, splits, subset=subset, full_entities=full_entities)
    if dataset == "tackbp":
        return load_tackbp(data_dir, splits, subset=subset)
    raise ValueError(f"Unknown dataset: {dataset!r}")


# ---------------------------------------------------------------------------
# Bi-encoder training
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_entity_index(
    model: BiEncoder,
    entities,
    tokenizer,
    max_entity_len: int,
    batch_size: int,
    device: str,
    tqdm_disabled: bool,
) -> EntityIndex:
    all_embs: list[torch.Tensor] = []
    all_ids: list[str] = []

    ent_list = list(entities.values())
    dl = build_entity_loader(ent_list, tokenizer, max_entity_len, batch_size, device=device)

    for batch in tqdm(dl, desc="Building entity index", disable=tqdm_disabled, file=sys.stderr):
        emb = model.encode_entities(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        all_embs.append(emb.cpu())
        all_ids.extend(batch["entity_ids"])

    index = EntityIndex()
    index.build(all_ids, torch.cat(all_embs, dim=0))
    return index


def evaluate_biencoder(
    model: BiEncoder,
    eval_dl,
    entities,
    tokenizer,
    *,
    device: str,
    max_entity_len: int,
    batch_size: int,
    tqdm_disabled: bool,
) -> dict[str, float]:
    model.eval()
    index = _build_entity_index(
        model, entities, tokenizer, max_entity_len, batch_size, device, tqdm_disabled
    )

    gold_ids: list[str] = []
    candidates: list[list[str]] = []

    for batch in tqdm(eval_dl, desc="Eval retrieval", leave=False, disable=tqdm_disabled, file=sys.stderr):
        mention_emb = model.encode_mentions(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        top_ids, _ = index.search(mention_emb, k=100)
        gold_ids.extend(batch["gold_entity_ids"])
        candidates.extend(top_ids)

    m = compute_retrieval_metrics(gold_ids, candidates)
    return {
        "recall_at_1":   m.recall_at_1,
        "recall_at_10":  m.recall_at_10,
        "recall_at_64":  m.recall_at_64,
        "recall_at_100": m.recall_at_100,
    }


def train_biencoder(cfg: DictConfig, run: forge.ExperimentRun) -> None:
    device = cfg.runtime.device
    tqdm_disabled = not sys.stderr.isatty() or bool(cfg.runtime.get("tqdm_disabled", False))

    entities, mentions_by_split = _load_data(cfg)
    train_split = str(cfg.data.get("train_split", "train"))
    eval_split  = str(cfg.data.get("eval_split",  "valid"))
    mentions_train = mentions_by_split.get(train_split, [])
    mentions_eval  = mentions_by_split.get(eval_split,  [])

    model_name = str(cfg.model.get("backbone", "bert-base-uncased"))
    tokenizer  = build_tokenizer(model_name)
    model = BiEncoder(model_name).to(device)

    train_dl, eval_dl = build_biencoder_loaders(
        mentions_train, mentions_eval, entities, tokenizer,
        cfg.data, cfg.runtime.data, device=device,
    )
    max_entity_len = int(cfg.data.get("max_entity_len", 128))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.model.optim.lr),
        weight_decay=float(cfg.model.optim.weight_decay),
        betas=tuple(cfg.model.optim.betas),
    )
    epochs = int(cfg.train.epochs)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_recall = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}", disable=tqdm_disabled, file=sys.stderr):
            optimizer.zero_grad(set_to_none=True)

            mention_emb = model.encode_mentions(
                batch["mention_ids"].to(device),
                batch["mention_mask"].to(device),
            )
            entity_emb = model.encode_entities(
                batch["entity_ids"].to(device),
                batch["entity_mask"].to(device),
            )
            logits = mention_emb @ entity_emb.T   # [B, B] in-batch negatives
            loss   = in_batch_loss(logits)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.detach().item()
            n_batches  += 1

        train_loss = total_loss / max(1, n_batches)
        eval_metrics = evaluate_biencoder(
            model, eval_dl, entities, tokenizer,
            device=device, max_entity_len=max_entity_len,
            batch_size=int(cfg.runtime.data.batch_size),
            tqdm_disabled=tqdm_disabled,
        )

        t = PrettyTable(["metric", "value"])
        t.add_row(["train_loss", f"{train_loss:.4f}"])
        for k, v in eval_metrics.items():
            t.add_row([k, f"{v:.4f}"])
        log.info("Epoch %d/%d\n%s", epoch, epochs, t)

        run.push_log({"train_loss": train_loss, "epoch": epoch, **eval_metrics}, step=epoch)

        ckpt = ckpt_dir / f"model_epoch{epoch}.pth"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, ckpt)

        recall_64 = eval_metrics.get("recall_at_64", 0.0)
        if recall_64 >= best_recall:
            best_recall = recall_64
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "model_best.pth")
            log.info("New best recall@64=%.4f — saved model_best.pth", best_recall)

    run.finish(metrics={"best_recall_at_64": best_recall})
    log.info("Training complete.  best_recall@64=%.4f", best_recall)

    deploy_dst = Path(OmegaConf.select(cfg, "deploy_path", default="/home/simonelusetti/pwc/models/biencoder_best.pth"))
    deploy_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_dir / "model_best.pth", deploy_dst)
    log.info("Deployed best checkpoint → %s", deploy_dst)


# ---------------------------------------------------------------------------
# Autoregressive training
# ---------------------------------------------------------------------------

def train_autoregressive(cfg: DictConfig, run: forge.ExperimentRun) -> None:
    device = cfg.runtime.device
    tqdm_disabled = not sys.stderr.isatty() or bool(cfg.runtime.get("tqdm_disabled", False))

    entities, mentions_by_split = _load_data(cfg)
    train_split = str(cfg.data.get("train_split", "train"))
    eval_split  = str(cfg.data.get("eval_split",  "valid"))
    mentions_train = mentions_by_split.get(train_split, [])
    mentions_eval  = mentions_by_split.get(eval_split,  [])

    model_name = str(cfg.model.get("backbone", "facebook/bart-base"))
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)

    model = AutoregressiveEL(model_name).to(device)

    train_dl = build_seq2seq_loader(mentions_train, entities, tokenizer, cfg.data, cfg.runtime.data, device=device)
    eval_dl  = build_seq2seq_loader(mentions_eval,  entities, tokenizer, cfg.data, cfg.runtime.data, device=device, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.model.optim.lr),
        weight_decay=float(cfg.model.optim.weight_decay),
        betas=tuple(cfg.model.optim.betas),
    )
    epochs    = int(cfg.train.epochs)
    ckpt_dir  = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0.0
        n_batches   = 0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}", disable=tqdm_disabled, file=sys.stderr):
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train += loss.detach().item()
            n_batches   += 1

        train_loss = total_train / max(1, n_batches)

        model.eval()
        total_eval = 0.0
        n_eval = 0
        with torch.no_grad():
            for batch in tqdm(eval_dl, desc="Eval", leave=False, disable=tqdm_disabled, file=sys.stderr):
                loss = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                total_eval += loss.item()
                n_eval += 1
        eval_loss = total_eval / max(1, n_eval)

        log.info("Epoch %d/%d  train_loss=%.4f  eval_loss=%.4f", epoch, epochs, train_loss, eval_loss)
        run.push_log({"train_loss": train_loss, "eval_loss": eval_loss, "epoch": epoch}, step=epoch)

        ckpt = ckpt_dir / f"model_epoch{epoch}.pth"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "model_best.pth")
            log.info("New best eval_loss=%.4f — saved model_best.pth", best_loss)

    run.finish(metrics={"best_eval_loss": best_loss})

    deploy_dst = Path(OmegaConf.select(cfg, "deploy_path", default="/home/simonelusetti/pwc/models/autoregressive_best.pth"))
    deploy_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_dir / "model_best.pth", deploy_dst)
    log.info("Deployed best checkpoint → %s", deploy_dst)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg: DictConfig) -> None:
    if cfg.runtime.device == "cuda" and not torch.cuda.is_available():
        OmegaConf.update(cfg, "runtime.device", "cpu")
        print("CUDA not available — falling back to CPU.", file=sys.stderr)

    run = forge.start_run(cfg)

    arch = str(cfg.model.get("arch", "biencoder"))
    if arch == "biencoder":
        train_biencoder(cfg, run)
    elif arch == "autoregressive":
        train_autoregressive(cfg, run)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}. Choose 'biencoder' or 'autoregressive'.")
