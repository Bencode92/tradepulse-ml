#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradePulse – FinBERT Fine‑Tuning Utility (clean version)
=======================================================

•  Charge un corpus (CSV/JSON) de textes financiers déjà étiquetés
  en **positive / neutral / negative**.
•  Découpe automatiquement en train / validation (80 / 20 stratifié).
•  Tokenise, fine‑tune et enregistre un FinBERT (ou autre modèle) déjà
  présent sur HuggingFace Hub.
•  Produit un *training_report.json* + logs TensorBoard dans <output_dir>.

Exemple :
----------
$ python finetune.py \\
    --dataset datasets/news_20250705.csv \\
    --output_dir models/finbert-v1 \\
    --model_name yiyanghkust/finbert-tone

Ou avec auto-sélection :
$ python finetune.py --output_dir models/finbert-auto
  # Auto-détecte le dernier dataset
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Auto-sélection helper
try:
    from utils import get_date_from_filename, latest_dataset

    AUTOSEL = True
except ImportError:
    AUTOSEL = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s — %(levelname)s — %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
    handlers=[logging.FileHandler("finetune.log"), logging.StreamHandler()],
)
logger = logging.getLogger("tradepulse-finetune")


# ---------------------------------------------------------------------------
# Fine‑tuner class
# ---------------------------------------------------------------------------
class Finetuner:
    LABEL_MAP: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
    ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

    def __init__(self, model_name: str, max_length: int):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=self.ID2LABEL,
            label2id=self.LABEL_MAP,
        )
        logger.info("✅ Model & tokenizer loaded : %s", model_name)

    # -------------------------------------------------------------------
    # Data helpers
    # -------------------------------------------------------------------
    def _load_raw(self, path: Path) -> List[Dict[str, str]]:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path).to_dict("records")
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text("utf-8"))
        raise ValueError(f"Unsupported file type : {path}")

    def _standardise(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for row in rows:
            text = row.get("text") or (
                f"{row.get('title', '')} {row.get('content', '')}".strip()
            )
            label = (
                row.get("label")
                or row.get("sentiment")
                or row.get("impact")
                or ""
            ).lower()
            if not text or label not in self.LABEL_MAP:
                continue
            out.append({"text": text, "label": self.LABEL_MAP[label]})
        return out

    def load_dataset(self, path: Path) -> DatasetDict:
        """Load & tokenise dataset, return HF DatasetDict."""
        raw = self._load_raw(path)
        data = self._standardise(raw)
        if not data:
            raise RuntimeError("No usable samples detected in dataset !")
        logger.info("📊 %d samples after cleaning", len(data))

        train, val = train_test_split(
            data,
            test_size=0.2,
            stratify=[d["label"] for d in data],
            random_state=42,
        )

        def tok(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        train_ds = Dataset.from_list(train).map(
            tok, batched=True, remove_columns=["text"]
        )
        val_ds = Dataset.from_list(val).map(
            tok, batched=True, remove_columns=["text"]
        )
        return DatasetDict(train=train_ds, validation=val_ds)

    # -------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------
    @staticmethod
    def _metrics(pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = pred
        preds = np.argmax(logits, axis=1)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------
    def train(self, ds: DatasetDict, args: argparse.Namespace):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"finbert-{ts}"

        targs = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_bs,
            per_device_eval_batch_size=args.eval_bs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup,
            evaluation_strategy=args.eval_strategy,
            save_strategy=args.save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=args.logging_steps,
            seed=args.seed,
            push_to_hub=args.push,
            hub_model_id=args.hub_id if args.push else None,
            report_to="tensorboard",
        )

        trainer = Trainer(
            model=self.model,
            args=targs,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self._metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        logger.info("🔥 Start training for %d epochs", args.epochs)
        trainer.train()
        trainer.save_model()

        eval_res = trainer.evaluate()
        logger.info(
            "✅ Training complete — F1: %.4f | Acc: %.4f",
            eval_res["eval_f1"],
            eval_res["eval_accuracy"],
        )

        # save a report
        report = {
            "model": self.model_name,
            "epochs": args.epochs,
            "metrics": eval_res,
            "timestamp": datetime.now().isoformat(),
        }
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir, "training_report.json"), "w") as fh:
            json.dump(report, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TradePulse FinBERT fine‑tuning utility"
    )
    p.add_argument(
        "--dataset",
        type=Path,
        help="Path to CSV/JSON dataset (auto-détection si omis)",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Where to save the model & logs",
    )
    p.add_argument("--model_name", default="yiyanghkust/finbert-tone")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--train_bs", type=int, default=16, help="Train batch size")
    p.add_argument("--eval_bs", type=int, default=32, help="Eval batch size")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--save_strategy", default="epoch")
    p.add_argument("--eval_strategy", default="epoch")
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push", action="store_true", help="Push model to HF Hub")
    p.add_argument(
        "--hub_id", type=str, default=None, help="HF repo id (org/model)"
    )
    return p


# ---------------------------------------------------------------------------
# Entrée principale
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    # Auto-sélection dataset
    if args.dataset is None and AUTOSEL:
        args.dataset = latest_dataset()
        if args.dataset:
            logger.info("🕵️  Auto-sélection dataset : %s", args.dataset)
        else:
            logger.error("❌ Aucun dataset trouvé")
            logger.info(
                "💡 Ajoutez des fichiers au format news_YYYYMMDD.csv "
                "dans datasets/"
            )
            return
    elif args.dataset is None:
        logger.error(
            "❌ Aucun dataset spécifié et auto-sélection non disponible"
        )
        logger.info(
            "💡 Utilisez: python scripts/finetune.py "
            "--dataset datasets/votre_fichier.csv --output_dir models/test"
        )
        return

    # Auto-génération nom de modèle basé sur la date du dataset
    if AUTOSEL and args.dataset:
        date_str = get_date_from_filename(str(args.dataset))
        if date_str and str(args.output_dir).endswith("-auto"):
            # Si output_dir se termine par -auto, on le remplace par la date
            new_output = str(args.output_dir).replace("-auto", f"-{date_str}")
            args.output_dir = Path(new_output)
            logger.info("📂 Nom de modèle auto-généré : %s", args.output_dir)

    tuner = Finetuner(model_name=args.model_name, max_length=args.max_length)
    ds = tuner.load_dataset(args.dataset)
    tuner.train(ds, args)


if __name__ == "__main__":
    main()
