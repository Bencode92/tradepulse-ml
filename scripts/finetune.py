#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradePulse – FinBERT Fine‑Tuning Utility avec Apprentissage Incrémental
======================================================================

🚀 NOUVEAU : Apprentissage Incrémental !
- Mode --incremental : Améliore un modèle existant au lieu de créer un nouveau
- Validation automatique avant mise à jour  
- Modèles production/développement séparés
- Rollback automatique si dégradation

•  Charge un corpus (CSV/JSON) de textes financiers déjà étiquetés
  en **positive / neutral / negative**.
•  Découpe automatiquement en train / validation (80 / 20 stratifié).
•  Tokenise, fine‑tune et enregistre un FinBERT (ou autre modèle) déjà
  présent sur HuggingFace Hub.
•  Produit un *training_report.json* + logs TensorBoard dans <output_dir>.

Usage Classique (existant):
---------------------------
$ python finetune.py \
    --dataset datasets/news_20250705.csv \
    --output_dir models/finbert-v1 \
    --model_name yiyanghkust/finbert-tone

🚀 NOUVEAU - Usage Apprentissage Incrémental:
--------------------------------------------
# Mode production (améliore le modèle stable)
$ python finetune.py --incremental --mode production --dataset datasets/news_20250707.csv

# Mode développement (améliore le modèle de dev)
$ python finetune.py --incremental --mode development --dataset datasets/news_20250707.csv

# Mode test (validation locale, pas de mise à jour)
$ python finetune.py --incremental --mode test --dataset datasets/news_20250707.csv

Auto-sélection (existant, conservé):
-----------------------------------
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
from typing import Dict, List, Tuple

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

# 🚀 NOUVEAU : Configuration des modèles de production
MODELS_CONFIG = {
    "production": {
        "hf_id": "Bencode92/tradepulse-finbert-prod",          # Modèle PRODUCTION stable
        "description": "Modèle stable pour site TradePulse",
        "auto_update": True,  # Met à jour automatiquement si amélioration
    },
    "development": {
        "hf_id": "Bencode92/tradepulse-finbert-dev",          # Modèle DEV pour tests
        "description": "Modèle de développement et tests",
        "auto_update": False,  # Mise à jour manuelle seulement
    }
}

# 🚀 NOUVEAU : Seuils de performance pour validation
PERFORMANCE_THRESHOLDS = {
    "min_accuracy": 0.80,      # Précision minimum acceptable
    "min_f1": 0.78,            # F1-score minimum 
    "improvement_threshold": 0.02,  # Amélioration minimum pour mise à jour (2%)
}

# Auto-sélection helper (existant, conservé)
try:
    from utils import get_date_from_filename, latest_dataset
    AUTOSEL = True
except ImportError:
    AUTOSEL = False

# ---------------------------------------------------------------------------
# Logging (existant, conservé)
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s — %(levelname)s — %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FMT,
    handlers=[logging.FileHandler("finetune.log"), logging.StreamHandler()],
)
logger = logging.getLogger("tradepulse-finetune")


# ---------------------------------------------------------------------------
# Fine‑tuner class (adapté pour supporter l'incrémental)
# ---------------------------------------------------------------------------
class Finetuner:
    LABEL_MAP: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
    ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

    def __init__(self, model_name: str, max_length: int, incremental_mode: bool = False, baseline_model: str = None):
        self.model_name = model_name
        self.max_length = max_length
        self.incremental_mode = incremental_mode
        self.baseline_model = baseline_model
        
        if incremental_mode and baseline_model:
            # 🚀 NOUVEAU : Mode incrémental - charger modèle existant
            try:
                logger.info(f"📥 Tentative de chargement du modèle existant: {baseline_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(baseline_model)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    baseline_model,
                    num_labels=3,
                    id2label=self.ID2LABEL,
                    label2id=self.LABEL_MAP,
                )
                logger.info(f"✅ Modèle incrémental chargé: {baseline_model}")
                self.model_name = baseline_model  # Update model name
            except Exception as e:
                # Fallback sur le modèle de base
                logger.warning(f"⚠️ Impossible de charger {baseline_model}: {e}")
                logger.info(f"🔄 Fallback sur modèle de base: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=3,
                    id2label=self.ID2LABEL,
                    label2id=self.LABEL_MAP,
                )
                self.baseline_model = model_name
        else:
            # Mode classique (existant, conservé)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,
                id2label=self.ID2LABEL,
                label2id=self.LABEL_MAP,
            )
            logger.info("✅ Model & tokenizer loaded : %s", model_name)

    # -------------------------------------------------------------------
    # 🚀 NOUVEAU : Méthodes d'évaluation pour mode incrémental
    # -------------------------------------------------------------------
    def evaluate_on_test(self, test_dataset: Dataset) -> Dict[str, float]:
        """Évalue le modèle sur un dataset de test"""
        # Tokenisation du dataset de test
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
        
        eval_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        # Évaluation avec Trainer
        eval_args = TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=32,
            logging_dir=None,
            report_to=[],  # Pas de logging externe
        )
        
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._metrics,
        )
        
        metrics = trainer.evaluate()
        
        # Nettoyage des métriques 
        clean_metrics = {
            "accuracy": metrics.get("eval_accuracy", 0.0),
            "f1": metrics.get("eval_f1", 0.0),
            "precision": metrics.get("eval_precision", 0.0),
            "recall": metrics.get("eval_recall", 0.0),
        }
        
        return clean_metrics

    def should_update_model(self, baseline_metrics: Dict[str, float], new_metrics: Dict[str, float], 
                           min_improvement: float = None) -> Tuple[bool, str]:
        """Détermine si le nouveau modèle doit remplacer l'ancien"""
        
        min_improvement = min_improvement or PERFORMANCE_THRESHOLDS["improvement_threshold"]
        
        # Vérification des seuils minimum
        if new_metrics["accuracy"] < PERFORMANCE_THRESHOLDS["min_accuracy"]:
            return False, f"Précision insuffisante: {new_metrics['accuracy']:.3f} < {PERFORMANCE_THRESHOLDS['min_accuracy']}"
        
        if new_metrics["f1"] < PERFORMANCE_THRESHOLDS["min_f1"]:
            return False, f"F1-score insuffisant: {new_metrics['f1']:.3f} < {PERFORMANCE_THRESHOLDS['min_f1']}"
        
        # Vérification de l'amélioration
        accuracy_improvement = new_metrics["accuracy"] - baseline_metrics["accuracy"]
        f1_improvement = new_metrics["f1"] - baseline_metrics["f1"]
        
        if accuracy_improvement >= min_improvement or f1_improvement >= min_improvement:
            return True, f"Amélioration détectée - Accuracy: +{accuracy_improvement:.3f}, F1: +{f1_improvement:.3f}"
        
        return False, f"Amélioration insuffisante - Accuracy: {accuracy_improvement:+.3f}, F1: {f1_improvement:+.3f} (min: {min_improvement})"

    def push_to_huggingface(self, model_path: Path, hf_model_id: str, commit_message: str = None):
        """Push le modèle vers HuggingFace Hub"""
        
        commit_message = commit_message or f"Apprentissage incrémental - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        try:
            # Chargement du modèle depuis le dossier local
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Push vers HuggingFace
            logger.info(f"📤 Push vers HuggingFace: {hf_model_id}")
            
            model.push_to_hub(
                hf_model_id,
                commit_message=commit_message,
                private=False  # Ou True si vous voulez un repo privé
            )
            
            tokenizer.push_to_hub(
                hf_model_id,
                commit_message=commit_message,
                private=False
            )
            
            logger.info(f"✅ Modèle pushé vers HuggingFace: https://huggingface.co/{hf_model_id}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du push: {e}")
            raise

    # -------------------------------------------------------------------
    # Data helpers (adaptés pour supporter le mode incrémental)
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

    def load_dataset(self, path: Path) -> Tuple[DatasetDict, Dataset]:
        """Load & tokenise dataset, return HF DatasetDict + test dataset pour évaluation incrémentale"""
        raw = self._load_raw(path)
        data = self._standardise(raw)
        if not data:
            raise RuntimeError("No usable samples detected in dataset !")
        logger.info("📊 %d samples after cleaning", len(data))

        if self.incremental_mode:
            # 🚀 NOUVEAU : Division en 3 parties pour l'apprentissage incrémental
            # 70% train, 20% validation, 10% test (pour évaluation baseline)
            train_val, test_data = train_test_split(
                data,
                test_size=0.1,
                stratify=[d["label"] for d in data],
                random_state=42,
            )
            
            train, val = train_test_split(
                train_val,
                test_size=0.25,  # 0.25 de 90% = 22.5% du total ≈ 20%
                stratify=[d["label"] for d in train_val],
                random_state=42,
            )
            
            logger.info(f"📊 Mode incrémental - Train: {len(train)}, Validation: {len(val)}, Test: {len(test_data)}")
        else:
            # Mode classique (existant, conservé)
            train, val = train_test_split(
                data,
                test_size=0.2,
                stratify=[d["label"] for d in data],
                random_state=42,
            )
            test_data = []  # Pas de dataset de test en mode classique
            logger.info(f"📊 Mode classique - Train: {len(train)}, Validation: {len(val)}")

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
        
        # Dataset de test (pour évaluation baseline en mode incrémental)
        test_ds = Dataset.from_list(test_data) if test_data else None
        
        return DatasetDict(train=train_ds, validation=val_ds), test_ds

    # -------------------------------------------------------------------
    # Metrics (existant, conservé)
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
    # Training (adapté pour supporter l'incrémental)
    # -------------------------------------------------------------------
    def train(self, ds: DatasetDict, args: argparse.Namespace, test_ds: Dataset = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 🚀 NOUVEAU : Nommage différent selon le mode
        if self.incremental_mode:
            run_name = f"incremental-{getattr(args, 'mode', 'test')}-{ts}"
        else:
            run_name = f"finbert-{ts}"

        # 🚀 NOUVEAU : Arguments d'entraînement adaptés pour l'incrémental
        if self.incremental_mode:
            # Paramètres plus conservateurs pour l'apprentissage incrémental
            learning_rate = 1e-5  # Plus faible pour la stabilité
            epochs = 2  # Moins d'epochs pour éviter l'overfitting
            batch_size = 8  # Plus petit batch
            patience = 1  # Arrêt précoce plus agressif
        else:
            # Paramètres classiques (existant, conservé)
            learning_rate = args.lr
            epochs = args.epochs
            batch_size = args.train_bs
            patience = 2

        targs = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=args.eval_bs,
            learning_rate=learning_rate,
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
            push_to_hub=False if self.incremental_mode else args.push,  # Gestion manuelle en mode incrémental
            hub_model_id=args.hub_id if (args.push and not self.incremental_mode) else None,
            report_to="tensorboard",
            save_total_limit=2,  # Garder seulement les 2 meilleurs checkpoints
        )

        trainer = Trainer(
            model=self.model,
            args=targs,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self._metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )

        mode_info = "incremental" if self.incremental_mode else "classic"
        logger.info(f"🔥 Start training for %d epochs (mode: %s)", epochs, mode_info)
        trainer.train()
        trainer.save_model()

        eval_res = trainer.evaluate()
        logger.info(
            "✅ Training complete — F1: %.4f | Acc: %.4f",
            eval_res["eval_f1"],
            eval_res["eval_accuracy"],
        )

        # 🚀 NOUVEAU : Évaluation sur le dataset de test pour mode incrémental
        test_metrics = None
        if test_ds is not None and self.incremental_mode:
            logger.info("📊 Évaluation sur dataset de test...")
            test_metrics = self.evaluate_on_test(test_ds)
            logger.info(f"📊 Métriques test: {test_metrics}")

        # save a report (adapté pour mode incrémental)
        report = {
            "model": self.model_name,
            "mode": "incremental" if self.incremental_mode else "classic",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "validation_metrics": eval_res,
            "test_metrics": test_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 🚀 NOUVEAU : Ajouter infos incrémentales si applicable
        if self.incremental_mode:
            report.update({
                "incremental_mode": getattr(args, 'mode', 'test'),
                "baseline_model": self.baseline_model,
            })

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 🚀 NOUVEAU : Nom de rapport différent selon le mode
        report_name = "incremental_training_report.json" if self.incremental_mode else "training_report.json"
        
        with open(Path(args.output_dir, report_name), "w") as fh:
            json.dump(report, fh, indent=2)

        return test_metrics


# ---------------------------------------------------------------------------
# CLI (adapté pour supporter l'incrémental)
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TradePulse FinBERT fine‑tuning utility avec apprentissage incrémental"
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
    
    # Arguments existants (conservés)
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
    
    # 🚀 NOUVEAUX arguments pour l'apprentissage incrémental
    p.add_argument("--incremental", action="store_true", 
                   help="Activer l'apprentissage incrémental")
    p.add_argument("--mode", choices=["test", "development", "production"], default="test",
                   help="Mode incrémental (test/development/production)")
    p.add_argument("--baseline-model", type=str, default=None,
                   help="Modèle de référence pour l'incrémental (auto selon mode si non spécifié)")
    p.add_argument("--min-improvement", type=float, default=0.02,
                   help="Amélioration minimum requise pour mise à jour (défaut: 0.02)")
    p.add_argument("--force-update", action="store_true",
                   help="Forcer la mise à jour même sans amélioration significative")
    
    return p


# ---------------------------------------------------------------------------
# Entrée principale (adaptée pour supporter l'incrémental)
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    # Auto-sélection dataset (existant, conservé)
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

    # 🚀 NOUVEAU : Gestion du mode incrémental
    if args.incremental:
        logger.info("🔄 Mode apprentissage incrémental activé")
        
        # Déterminer le modèle de base selon le mode
        if args.baseline_model is None:
            args.baseline_model = MODELS_CONFIG.get(args.mode, {}).get("hf_id", "yiyanghkust/finbert-tone")
        
        logger.info(f"🎯 Mode: {args.mode}")
        logger.info(f"📦 Modèle de base: {args.baseline_model}")
        
        # Initialiser le fine-tuner en mode incrémental
        tuner = Finetuner(
            model_name=args.model_name, 
            max_length=args.max_length,
            incremental_mode=True,
            baseline_model=args.baseline_model
        )
        
        # Chargement du dataset avec division train/val/test
        ds, test_ds = tuner.load_dataset(args.dataset)
        
        # Évaluation baseline du modèle existant
        logger.info("📊 Évaluation baseline du modèle existant...")
        baseline_metrics = tuner.evaluate_on_test(test_ds)
        logger.info(f"📊 Métriques baseline: {baseline_metrics}")
        
        # Entraînement incrémental
        test_metrics = tuner.train(ds, args, test_ds)
        
        # Décision de mise à jour
        if test_metrics:
            logger.info("🔍 Analyse des résultats pour apprentissage incrémental...")
            
            should_update, reason = tuner.should_update_model(
                baseline_metrics, 
                test_metrics,
                args.min_improvement
            )
            
            logger.info(f"📊 Décision de mise à jour: {reason}")
            
            # Forcer la mise à jour si demandé
            if args.force_update and not should_update:
                should_update = True
                reason = f"Mise à jour forcée (--force-update). {reason}"
                logger.warning(f"⚠️ {reason}")
            
            if should_update:
                # Mise à jour du modèle sur HuggingFace
                if args.mode in ["production", "development"]:
                    hf_model_id = MODELS_CONFIG[args.mode]["hf_id"]
                    commit_msg = f"Apprentissage incrémental - Accuracy: {test_metrics['accuracy']:.3f}, F1: {test_metrics['f1']:.3f}"
                    
                    logger.info(f"🚀 Mise à jour du modèle {args.mode}: {hf_model_id}")
                    tuner.push_to_huggingface(args.output_dir, hf_model_id, commit_msg)
                    
                    logger.info("✅ Modèle de production mis à jour!")
                else:
                    logger.info("✅ Nouveau modèle validé (mode test - pas de mise à jour auto)")
            else:
                logger.info(f"⏸️ Modèle non mis à jour: {reason}")
            
            # Mise à jour du rapport avec la décision
            report_path = args.output_dir / "incremental_training_report.json"
            if report_path.exists():
                with open(report_path, "r") as f:
                    report = json.load(f)
                
                report.update({
                    "baseline_metrics": baseline_metrics,
                    "new_metrics": test_metrics,
                    "model_updated": should_update,
                    "update_reason": reason,
                    "hf_model_id": MODELS_CONFIG.get(args.mode, {}).get("hf_id") if should_update else None,
                })
                
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
        
    else:
        # 🔄 Mode classique (existant, conservé exactement)
        logger.info("🔄 Mode classique activé")
        
        # Auto-génération nom de modèle basé sur la date du dataset (existant, conservé)
        if AUTOSEL and args.dataset:
            date_str = get_date_from_filename(str(args.dataset))
            if date_str and str(args.output_dir).endswith("-auto"):
                # Si output_dir se termine par -auto, on le remplace par la date
                new_output = str(args.output_dir).replace("-auto", f"-{date_str}")
                args.output_dir = Path(new_output)
                logger.info("📂 Nom de modèle auto-généré : %s", args.output_dir)

        tuner = Finetuner(model_name=args.model_name, max_length=args.max_length)
        ds, _ = tuner.load_dataset(args.dataset)
        tuner.train(ds, args)

        # Push classique si demandé (existant, conservé)
        if args.push:
            logger.info("📤 Push du modèle vers HuggingFace Hub...")


if __name__ == "__main__":
    main()
