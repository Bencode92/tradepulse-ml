#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TradePulse – FinBERT Fine‑Tuning Utility avec Apprentissage Incrémental + Class Balancing + Corrélations
=========================================================================================================

🎯 NOUVEAU : Modèles Séparés !
- Sentiment → Bencode92/tradepulse-finbert-sentiment
- Importance → Bencode92/tradepulse-finbert-importance
- Corrélations → Bencode92/tradepulse-finbert-correlations (NOUVEAU!)
- Hub ID automatique selon --target-column

🌐 NOUVEAU : Support Corrélations Multi-Label !
- --target-column correlations : Entraîne sur les corrélations commodités
- Classification multi-label (133 labels possibles)
- Métriques adaptées (F1 micro/macro, Hamming loss)
- BCEWithLogitsLoss pour multi-label

🚀 NOUVEAU : Apprentissage Incrémental !
- Mode --incremental : Améliore un modèle existant au lieu de créer un nouveau
- Validation automatique avant mise à jour  
- Modèles production/développement séparés
- Rollback automatique si dégradation

🎯 NOUVEAU : Support Importance !
- --target-column importance : Entraîne sur l'importance (critique/importante/générale)
- --target-column label : Entraîne sur le sentiment (positive/negative/neutral)

⚖️ NOUVEAU : Class Balancing !
- --class-balancing weighted : Pondération automatique des classes
- --class-balancing focal : Focal Loss pour classes déséquilibrées
- Métriques F1 macro adaptées aux datasets déséquilibrés

🔧 NOUVEAU : Push HuggingFace optimisé avec API corrigée !
- Clone automatique des repos HF
- Sauvegarde directe dans repo cloné
- Model card auto-générée avec métriques
- Git push natif (fini les commits vides)
- API HuggingFace Hub ≥0.22.0 compatible

🚀 PRODUCTION READY !
- Token HF debugging + gestion d'erreurs
- TrainingArguments redirection intelligente
- Git status robuste + fallbacks
- .gitignore auto-généré
- Logs informatifs pour debugging

•  Charge un corpus (CSV/JSON) de textes financiers déjà étiquetés
  en **positive / neutral / negative** ou **critique / importante / générale**
  ou **corrélations multi-label**.
•  Découpe automatiquement en train / validation (80 / 20 stratifié).
•  Tokenise, fine‑tune et enregistre un FinBERT (ou autre modèle) déjà
  présent sur HuggingFace Hub.
•  Produit un *training_report.json* + logs TensorBoard dans <output_dir>.

Usage Sentiment avec Hub ID automatique:
---------------------------------------
$ python finetune.py \
    --dataset datasets/news_20250708.csv \
    --output_dir models/finbert-sentiment \
    --target-column label \
    --push

Usage Importance avec Hub ID automatique:
----------------------------------------
$ python finetune.py \
    --dataset datasets/news_20250708.csv \
    --output_dir models/finbert-importance \
    --target-column importance \
    --push

Usage Corrélations avec Hub ID automatique (NOUVEAU!):
-----------------------------------------------------
$ python finetune.py \
    --dataset datasets/news_20250724.csv \
    --output_dir models/finbert-correlations \
    --target-column correlations \
    --push

🚀 NOUVEAU - Usage Apprentissage Incrémental:
--------------------------------------------
# Mode production (améliore le modèle stable)
$ python finetune.py --incremental --mode production --target-column label --dataset datasets/news_20250707.csv

# Mode développement (améliore le modèle de dev)
$ python finetune.py --incremental --mode development --target-column importance --dataset datasets/news_20250707.csv

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
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, multilabel_confusion_matrix, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

# 🔧 MODIFICATION 1 : NOUVEAUX IMPORTS pour HuggingFace Git
from huggingface_hub import Repository, HfApi, create_repo

# 🌐 NOUVEAU : Import du mapping des corrélations
try:
    from config.correlation_mapping import COMMODITY_CODES, correlations_to_labels, labels_to_correlations
    CORRELATION_SUPPORT = True
except ImportError:
    logger = logging.getLogger("tradepulse-finetune")
    logger.warning("⚠️ correlation_mapping non trouvé, support des corrélations désactivé")
    CORRELATION_SUPPORT = False
    COMMODITY_CODES = []

# 🔧 PATCH : Import iterative stratification pour multi-label
try:
    from skmultilearn.model_selection import iterative_train_test_split
    ITERATIVE_SPLIT = True
except ImportError:
    ITERATIVE_SPLIT = False
    logger = logging.getLogger("tradepulse-finetune")
    logger.warning("⚠️ skmultilearn non installé, split stratifié multi-label désactivé")
    logger.info("💡 Installer avec: pip install scikit-multilearn")

# 🎯 NOUVEAU : Configuration des modèles spécialisés (ajout corrélations)
MODELS_CONFIG = {
    "sentiment": {
        "production": {
            "hf_id": "Bencode92/tradepulse-finbert-sentiment",
            "description": "Modèle sentiment pour TradePulse",
            "auto_update": True,
        },
        "development": {
            "hf_id": "Bencode92/tradepulse-finbert-sentiment-dev",
            "description": "Modèle sentiment dev pour tests",
            "auto_update": False,
        },
        "fallback": "yiyanghkust/finbert-tone"
    },
    "importance": {
        "production": {
            "hf_id": "Bencode92/tradepulse-finbert-importance",
            "description": "Modèle importance pour TradePulse",
            "auto_update": True,
        },
        "development": {
            "hf_id": "Bencode92/tradepulse-finbert-importance-dev",
            "description": "Modèle importance dev pour tests",
            "auto_update": False,
        },
        "fallback": "yiyanghkust/finbert-tone"
    },
    "correlations": {  # 🌐 NOUVEAU
        "production": {
            "hf_id": "Bencode92/tradepulse-finbert-correlations",
            "description": "Modèle corrélations multi-label pour TradePulse",
            "auto_update": True,
        },
        "development": {
            "hf_id": "Bencode92/tradepulse-finbert-correlations-dev",
            "description": "Modèle corrélations dev pour tests",
            "auto_update": False,
        },
        "fallback": "yiyanghkust/finbert-tone"
    }
}

# 🚀 NOUVEAU : Seuils de performance pour validation
PERFORMANCE_THRESHOLDS = {
    "min_accuracy": 0.70,      # Précision minimum acceptable (réduit pour petits datasets)
    "min_f1": 0.65,            # F1-score minimum (réduit pour petits datasets)
    "improvement_threshold": 0.02,  # Amélioration minimum pour mise à jour (2%)
    # 🌐 NOUVEAU : Seuils pour multi-label
    "min_f1_micro": 0.50,      # F1 micro minimum pour corrélations
    "min_hamming": 0.85,       # Score Hamming minimum (1 - hamming_loss)
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
# ⚖️ NOUVEAU : Custom Loss avec Class Weighting et Focal Loss
# ---------------------------------------------------------------------------
class WeightedCrossEntropyLoss(nn.Module):
    """Cross Entropy Loss avec pondération des classes"""
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = torch.FloatTensor(class_weights)
        
    def forward(self, outputs, labels):
        # Move class weights to same device as outputs
        if outputs.is_cuda:
            self.class_weights = self.class_weights.cuda()
        
        # Compute weighted cross entropy
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        return loss_fn(outputs, labels)


class FocalLoss(nn.Module):
    """Focal Loss pour traiter les déséquilibres de classes"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, outputs, labels):
        ce_loss = nn.functional.cross_entropy(outputs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[labels]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CustomTrainer(Trainer):
    """Trainer personnalisé avec support des loss functions custom"""
    def __init__(self, loss_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_loss_fn = loss_fn
        self.pos_weight = None  # 🔧 PATCH : Pour multi-label
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if self.custom_loss_fn is not None:
            loss = self.custom_loss_fn(outputs.logits, labels)
        else:
            # Use default loss
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                # 🌐 NOUVEAU : BCEWithLogitsLoss pour multi-label
                if hasattr(self, 'is_multi_label') and self.is_multi_label:
                    # 🔧 PATCH : Utiliser pos_weight par label si disponible
                    pos_weight = None
                    if hasattr(self, 'pos_weight') and self.pos_weight is not None:
                        pos_weight = self.pos_weight.to(outputs.logits.device)
                    
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        outputs.logits, labels.float(), pos_weight=pos_weight
                    )
                else:
                    loss = nn.functional.cross_entropy(outputs.logits, labels)
                
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Fine‑tuner class (adapté pour supporter l'incrémental + importance + class balancing + corrélations)
# ---------------------------------------------------------------------------
class Finetuner:
    # 😊 Labels pour sentiment (existant)
    SENTIMENT_LABEL_MAP: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
    SENTIMENT_ID2LABEL: Dict[int, str] = {v: k for k, v in SENTIMENT_LABEL_MAP.items()}
    
    # 🎯 NOUVEAU : Labels pour importance
    IMPORTANCE_LABEL_MAP: Dict[str, int] = {"générale": 0, "importante": 1, "critique": 2}
    IMPORTANCE_ID2LABEL: Dict[int, str] = {v: k for k, v in IMPORTANCE_LABEL_MAP.items()}

    def __init__(self, model_name: str, max_length: int, incremental_mode: bool = False, 
                 baseline_model: str = None, target_column: str = "label", 
                 class_balancing: str = None, mode: str = "production"):
        self.model_name = model_name
        self.max_length = max_length
        self.incremental_mode = incremental_mode
        self.baseline_model = baseline_model
        self.target_column = target_column  # 🎯 NOUVEAU
        self.class_balancing = class_balancing  # ⚖️ NOUVEAU
        self.mode = mode  # 🎯 NOUVEAU : Stocker le mode
        self.class_weights = None  # ⚖️ NOUVEAU
        
        # 🌐 NOUVEAU : Détection du type de tâche
        if target_column == "correlations":
            self.task_type = "correlations"
            self.is_multi_label = True
            self.num_labels = len(COMMODITY_CODES) if CORRELATION_SUPPORT else 133
        elif target_column == "importance":
            self.task_type = "importance"
            self.is_multi_label = False
            self.num_labels = 3
        else:
            self.task_type = "sentiment"
            self.is_multi_label = False
            self.num_labels = 3
        
        self.hub_id = self._get_hub_id()
        
        # 🔧 MODIFICATION 2 : Clone du repo HuggingFace dès l'initialisation avec corrections
        self.repo = None
        self.repo_dir = None
        
        if self.hub_id and self.mode in ["production", "development"]:
            try:
                # 🔧 CORRECTIF 5 : Vérification et debug du token HF
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    logger.error("❌ HF_TOKEN non trouvé dans l'environnement")
                    logger.error("💡 Ajoutez HF_TOKEN dans vos variables d'environnement ou GitHub Secrets")
                    self.repo = None
                    self.repo_dir = None
                    return
                else:
                    # Debug token (masqué pour sécurité)
                    token_preview = f"{hf_token[:7]}...{hf_token[-4:] if len(hf_token) > 11 else '***'}"
                    logger.info(f"🔑 HF_TOKEN détecté: {token_preview}")
                
                # Clone (ou update) le repo dans un dossier local
                self.repo_dir = Path(f"./hf-{self.task_type}-{self.mode}")
                
                # Vérifier si le repo existe déjà sur HF
                hf_api = HfApi()
                try:
                    hf_api.repo_info(self.hub_id)
                    repo_exists = True
                    logger.info(f"📦 Repo HuggingFace existant: {self.hub_id}")
                except:
                    repo_exists = False
                    logger.info(f"📦 Création du repo HuggingFace: {self.hub_id}")
                    
                if not repo_exists:
                    # Créer le repo s'il n'existe pas
                    create_repo(
                        self.hub_id,
                        token=hf_token,
                        private=False,
                        exist_ok=True
                    )
                
                # Cloner ou pull le repo
                if self.repo_dir.exists():
                    # Repo déjà cloné, faire un pull
                    self.repo = Repository(
                        local_dir=self.repo_dir,
                        clone_from=self.hub_id,
                        token=hf_token,
                        skip_lfs_files=True  # Évite le DL des gros poids
                    )
                    try:
                        self.repo.git_pull()
                        logger.info(f"🔄 Repo mis à jour: {self.repo_dir.resolve()}")
                    except Exception as e:
                        logger.warning(f"⚠️ Pull impossible (pas grave): {e}")
                else:
                    # Premier clone
                    self.repo = Repository(
                        local_dir=self.repo_dir,
                        clone_from=self.hub_id,
                        token=hf_token,
                        skip_lfs_files=True  # Évite le DL des gros poids
                    )
                    logger.info(f"📥 Repo cloné: {self.repo_dir.resolve()}")
                
                # 🔧 CORRECTIF 4 : Créer .gitignore pour ignorer les artefacts
                self._setup_gitignore()
                    
            except Exception as e:
                logger.warning(f"⚠️ Impossible de cloner le repo {self.hub_id}: {e}")
                logger.info("🔧 Le modèle sera sauvé localement uniquement")
                self.repo = None
                self.repo_dir = None
        
        # 🎯 NOUVEAU : Sélection des labels selon la colonne cible
        if target_column == "importance":
            self.LABEL_MAP = self.IMPORTANCE_LABEL_MAP
            self.ID2LABEL = self.IMPORTANCE_ID2LABEL
            logger.info(f"🎯 Mode entraînement : IMPORTANCE → {self.hub_id}")
        elif target_column == "correlations":
            # 🌐 NOUVEAU : Pas de label map pour multi-label
            self.LABEL_MAP = None
            self.ID2LABEL = {i: code for i, code in enumerate(COMMODITY_CODES)} if CORRELATION_SUPPORT else {}
            logger.info(f"🌐 Mode entraînement : CORRÉLATIONS MULTI-LABEL ({self.num_labels} labels) → {self.hub_id}")
        else:
            self.LABEL_MAP = self.SENTIMENT_LABEL_MAP
            self.ID2LABEL = self.SENTIMENT_ID2LABEL
            logger.info(f"😊 Mode entraînement : SENTIMENT → {self.hub_id}")
        
        # ⚖️ NOUVEAU : Info sur class balancing
        if class_balancing:
            logger.info(f"⚖️ Class balancing activé : {class_balancing}")
        
        if incremental_mode and baseline_model:
            # 🚀 NOUVEAU : Mode incrémental - charger modèle existant
            try:
                logger.info(f"📥 Tentative de chargement du modèle existant: {baseline_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(baseline_model)
                
                # 🌐 NOUVEAU : Configuration selon le type de tâche
                if self.is_multi_label:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        baseline_model,
                        num_labels=self.num_labels,
                        problem_type="multi_label_classification"
                    )
                else:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        baseline_model,
                        num_labels=self.num_labels,
                        id2label=self.ID2LABEL,
                        label2id=self.LABEL_MAP
                    )
                logger.info(f"✅ Modèle incrémental chargé: {baseline_model}")
                self.model_name = baseline_model  # Update model name
            except Exception as e:
                # Fallback sur le modèle de base
                logger.warning(f"⚠️ Impossible de charger {baseline_model}: {e}")
                logger.info(f"🔄 Fallback sur modèle de base: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                if self.is_multi_label:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=self.num_labels,
                        problem_type="multi_label_classification"
                    )
                else:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=self.num_labels,
                        id2label=self.ID2LABEL,
                        label2id=self.LABEL_MAP
                    )
                self.baseline_model = model_name
        else:
            # Mode classique (existant, conservé)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 🌐 NOUVEAU : Configuration selon le type de tâche
            if self.is_multi_label:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=self.num_labels,
                    problem_type="multi_label_classification"
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=self.num_labels,
                    id2label=self.ID2LABEL,
                    label2id=self.LABEL_MAP
                )
            logger.info("✅ Model & tokenizer loaded : %s", model_name)

    def _setup_gitignore(self):
        """🔧 CORRECTIF 4 : Créer .gitignore pour ignorer les artefacts"""
        if not self.repo_dir:
            return
            
        gitignore_path = self.repo_dir / ".gitignore"
        gitignore_content = """# TradePulse ML - Artefacts à ignorer
logs/
*.pt
*.pth
*.bin
checkpoints/
runs/
wandb/
*.log
__pycache__/
*.pyc
.DS_Store
tmp_eval/
"""
        
        try:
            if not gitignore_path.exists():
                gitignore_path.write_text(gitignore_content.strip(), encoding="utf-8")
                logger.info(f"📝 .gitignore créé: {gitignore_path}")
        except Exception as e:
            logger.warning(f"⚠️ Impossible de créer .gitignore: {e}")

    def _get_hub_id(self) -> str:
        """🎯 NOUVEAU : Retourne le hub_id approprié selon target_column et mode"""
        try:
            # Déterminer le type de tâche
            task_type = self.task_type
            
            if self.mode in ["production", "development"]:
                return MODELS_CONFIG[task_type][self.mode]["hf_id"]
            else:
                # Mode test : pas de hub_id spécifique
                return None
        except KeyError:
            logger.warning(f"⚠️ Configuration non trouvée pour {task_type}/{self.mode}")
            return None

    # -------------------------------------------------------------------
    # ⚖️ NOUVEAU : Méthodes de class balancing
    # -------------------------------------------------------------------
    def compute_class_weights(self, labels: List[int]) -> np.ndarray:
        """Calcule les poids des classes pour l'équilibrage"""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=labels
        )
        
        # Ensure weights are in correct order (0, 1, 2)
        weight_dict = dict(zip(unique_labels, class_weights))
        ordered_weights = [weight_dict.get(i, 1.0) for i in range(3)]
        
        logger.info(f"⚖️ Poids calculés: {dict(zip(range(3), ordered_weights))}")
        return np.array(ordered_weights)

    def create_loss_function(self, labels: List[int]):
        """Crée la fonction de loss selon le type de balancing"""
        if not self.class_balancing:
            return None
            
        if self.class_balancing == "weighted":
            self.class_weights = self.compute_class_weights(labels)
            return WeightedCrossEntropyLoss(self.class_weights)
        elif self.class_balancing == "focal":
            self.class_weights = self.compute_class_weights(labels)
            return FocalLoss(alpha=self.class_weights, gamma=2.0)
        else:
            logger.warning(f"⚠️ Type de balancing non reconnu: {self.class_balancing}")
            return None

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
        
        # 🌐 NOUVEAU : Utiliser les bonnes métriques selon le type
        compute_metrics_fn = self._metrics_multilabel if self.is_multi_label else self._metrics_balanced
        
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_fn,
        )
        
        metrics = trainer.evaluate()
        
        # Nettoyage des métriques 
        clean_metrics = {
            "accuracy": metrics.get("eval_accuracy", 0.0),
            "f1": metrics.get("eval_f1", 0.0),
            "f1_macro": metrics.get("eval_f1_macro", 0.0),  # ⚖️ NOUVEAU
            "precision": metrics.get("eval_precision", 0.0),
            "recall": metrics.get("eval_recall", 0.0),
        }
        
        # 🌐 NOUVEAU : Métriques spécifiques multi-label
        if self.is_multi_label:
            clean_metrics.update({
                "f1_micro": metrics.get("eval_f1_micro", 0.0),
                "hamming_score": metrics.get("eval_hamming_score", 0.0),
            })
        
        return clean_metrics

    def should_update_model(self, baseline_metrics: Dict[str, float], new_metrics: Dict[str, float], 
                           min_improvement: float = None) -> Tuple[bool, str]:
        """Détermine si le nouveau modèle doit remplacer l'ancien"""
        
        min_improvement = min_improvement or PERFORMANCE_THRESHOLDS["improvement_threshold"]
        
        # 🌐 NOUVEAU : Métriques différentes pour multi-label
        if self.is_multi_label:
            primary_metric = "f1_micro"
            primary_threshold = PERFORMANCE_THRESHOLDS["min_f1_micro"]
            
            # Vérification Hamming score
            if new_metrics.get("hamming_score", 0) < PERFORMANCE_THRESHOLDS["min_hamming"]:
                return False, f"Hamming score insuffisant: {new_metrics.get('hamming_score', 0):.3f} < {PERFORMANCE_THRESHOLDS['min_hamming']}"
        else:
            primary_metric = "f1_macro" if "f1_macro" in new_metrics else "f1"
            primary_threshold = PERFORMANCE_THRESHOLDS["min_f1"]
        
        # Vérification des seuils minimum
        if not self.is_multi_label and new_metrics["accuracy"] < PERFORMANCE_THRESHOLDS["min_accuracy"]:
            return False, f"Précision insuffisante: {new_metrics['accuracy']:.3f} < {PERFORMANCE_THRESHOLDS['min_accuracy']}"
        
        if new_metrics.get(primary_metric, 0) < primary_threshold:
            return False, f"{primary_metric} insuffisant: {new_metrics.get(primary_metric, 0):.3f} < {primary_threshold}"
        
        # Vérification de l'amélioration
        if self.is_multi_label:
            f1_improvement = new_metrics.get(primary_metric, 0) - baseline_metrics.get(primary_metric, 0)
            hamming_improvement = new_metrics.get("hamming_score", 0) - baseline_metrics.get("hamming_score", 0)
            
            if f1_improvement >= min_improvement or hamming_improvement >= min_improvement:
                return True, f"Amélioration détectée - {primary_metric}: +{f1_improvement:.3f}, Hamming: +{hamming_improvement:.3f}"
        else:
            accuracy_improvement = new_metrics["accuracy"] - baseline_metrics["accuracy"]
            f1_improvement = new_metrics[primary_metric] - baseline_metrics.get(primary_metric, baseline_metrics["f1"])
            
            if accuracy_improvement >= min_improvement or f1_improvement >= min_improvement:
                return True, f"Amélioration détectée - Accuracy: +{accuracy_improvement:.3f}, {primary_metric}: +{f1_improvement:.3f}"
        
        return False, f"Amélioration insuffisante (min: {min_improvement})"

    # 🔧 CORRECTION API HuggingFace Hub : push_to_huggingface() avec nouvelle API ≥0.22.0
    def push_to_huggingface(self, commit_message: str = None):
        """🎯 NOUVEAU : Push vers HuggingFace avec API ≥0.22.0 compatible"""
        
        if not self.repo:
            logger.warning("⚠️ Pas de repo à pusher (mode test ou erreur de clone)")
            return
            
        if not self.repo_dir or not self.repo_dir.exists():
            logger.error("❌ Dossier du repo non trouvé")
            return

        commit_message = commit_message or f"🏋️ Update {self.task_type} – {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        try:
            # 🔧 CORRECTION API : Vérification des changements avec nouvelle API
            has_changes = False
            
            try:
                # Méthode 1 : Utiliser self.repo.git("status") au lieu de git_status()
                status_output = self.repo.git("status", "--porcelain")
                if status_output and status_output.strip():
                    has_changes = True
                    logger.info("📊 Changements détectés via git status --porcelain")
                else:
                    # Vérifier aussi with git status classique
                    status_clean = self.repo.git("status")
                    if "nothing to commit" not in status_clean and "working tree clean" not in status_clean:
                        has_changes = True
                        logger.info("📊 Changements détectés via git status")
            except Exception as e:
                logger.warning(f"⚠️ git status failed: {e}")
                
                # Méthode 2 : Fallback avec git diff
                try:
                    diff_output = self.repo.git("diff", "--stat")
                    if diff_output and diff_output.strip():
                        has_changes = True
                        logger.info("📊 Changements détectés via git diff --stat")
                except Exception as e2:
                    # Méthode 3 : Forcer le push (assume changes)
                    logger.warning(f"⚠️ git diff failed: {e2}")
                    logger.info("🔧 Forcing push (assume changes)")
                    has_changes = True
            
            if not has_changes:
                logger.warning("⚠️ Aucun changement détecté dans le repo")
                return
                
            # 🔧 CORRECTION API : git_add() sans argument 'all' 
            # La nouvelle API accepte seulement pattern="." ou aucun argument
            self.repo.git_add()  # Par défaut ajoute tout (équivalent à git add .)
            logger.info("📦 Fichiers ajoutés au staging")
            
            # Commit avec message personnalisé
            self.repo.git_commit(commit_message)
            logger.info(f"💾 Commit créé: {commit_message}")
            
            # Push vers HuggingFace
            self.repo.git_push()
            logger.info(f"✅ Pushed vers HuggingFace: https://huggingface.co/{self.hub_id}")
            
            # Vérification post-push avec nouvelle API
            try:
                final_status = self.repo.git("status")
                if "Your branch is up to date" in final_status or "working tree clean" in final_status:
                    logger.info("🎯 Push confirmé - Repo synchronisé")
                else:
                    logger.info("🎯 Push probablement réussi")
            except Exception as e:
                logger.info("🎯 Push probablement réussi (vérification status impossible)")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du push: {e}")
            logger.info("🔧 Vous pouvez pusher manuellement:")
            logger.info(f"cd {self.repo_dir} && git add . && git commit -m '{commit_message}' && git push")
            raise

    # -------------------------------------------------------------------
    # Data helpers (adaptés pour supporter le mode incrémental + petits datasets + importance + corrélations)
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
                f"{row.get('title', '')} {row.get('content', '')}"
            ).strip()
            
            # 🌐 NOUVEAU : Gestion des corrélations multi-label
            if self.target_column == "correlations":
                correlations_str = row.get("correlations", "")
                if not correlations_str or not text:
                    continue
                
                # Convertir en labels binaires
                if CORRELATION_SUPPORT:
                    labels = correlations_to_labels(correlations_str)
                else:
                    # Fallback si pas de support
                    labels = [0] * self.num_labels
                
                out.append({"text": text, "labels": labels})
            else:
                # Gestion classique (sentiment/importance)
                if self.target_column == "importance":
                    label_raw = row.get("importance", "")
                else:
                    label_raw = (
                        row.get("label") 
                        or row.get("sentiment") 
                        or row.get("impact") 
                        or ""
                    )
                
                # 🔧 FIX : Nettoyage strict
                if label_raw is None:
                    label_raw = ""
                
                label = str(label_raw).strip().lower()
                
                # 🔧 FIX : Debug si label non reconnu
                if label and label not in self.LABEL_MAP:
                    logger.warning(f"⚠️ Label non reconnu: '{label}' (raw: '{label_raw}') pour target_column='{self.target_column}'")
                    logger.warning(f"Labels attendus: {list(self.LABEL_MAP.keys())}")
                    continue
                    
                if not text or not label:
                    continue
                    
                out.append({"text": text, "label": self.LABEL_MAP[label]})
        return out

    def _check_dataset_balance(self, data: List[Dict[str, str]]) -> bool:
        """Vérifie si le dataset est suffisamment équilibré pour un split stratifié"""
        # 🌐 NOUVEAU : Pas de stratification pour multi-label
        if self.is_multi_label:
            return False
        
        label_counts = Counter([d["label"] for d in data])
        
        logger.info(f"📊 Distribution des labels: {dict(label_counts)}")
        
        # ⚖️ NOUVEAU : Afficher les déséquilibres
        if self.class_balancing:
            total = sum(label_counts.values())
            for label_id, count in label_counts.items():
                label_name = self.ID2LABEL.get(label_id, f"label_{label_id}")
                percentage = (count / total) * 100
                logger.info(f"  {label_name}: {count} ({percentage:.1f}%)")
        
        # Vérifier que chaque classe a au moins 2 échantillons
        min_samples = min(label_counts.values()) if label_counts else 0
        
        if min_samples < 2:
            logger.warning(f"⚠️ Dataset déséquilibré: classe minimale = {min_samples} échantillon(s)")
            logger.warning(f"⚠️ Split stratifié impossible, utilisation d'un split simple")
            return False
        
        return True

    def load_dataset(self, path: Path) -> Tuple[DatasetDict, Dataset]:
        """Load & tokenise dataset, return HF DatasetDict + test dataset pour évaluation incrémentale"""
        raw = self._load_raw(path)
        data = self._standardise(raw)
        if not data:
            raise RuntimeError("No usable samples detected in dataset !")
        logger.info("📊 %d samples after cleaning", len(data))

        # 🌐 NOUVEAU : Gestion différente pour multi-label
        if self.is_multi_label:
            # 🔧 PATCH : Préparer les données pour split stratifié multi-label
            X = [d["text"] for d in data]
            y = np.array([d["labels"] for d in data])
            
            # Log distribution
            logger.info(f"📊 Labels positifs total: {y.sum()}")
            logger.info(f"📊 Labels avec ≥1 positif: {(y.sum(axis=0) > 0).sum()} / {y.shape[1]}")
            
            if ITERATIVE_SPLIT and len(data) >= 10:
                try:
                    # 🔧 PATCH : Split stratifié itératif pour multi-label
                    if self.incremental_mode:
                        # Mode incrémental : 70% train, 20% val, 10% test
                        X_train_val, y_train_val, X_test, y_test = iterative_train_test_split(
                            np.array(X).reshape(-1, 1), y, test_size=0.1
                        )
                        X_train, y_train, X_val, y_val = iterative_train_test_split(
                            X_train_val, y_train_val, test_size=0.22  # 0.22 de 90% ≈ 20% total
                        )
                    else:
                        # Mode classique : 80% train, 20% val
                        X_train, y_train, X_val, y_val = iterative_train_test_split(
                            np.array(X).reshape(-1, 1), y, test_size=0.2
                        )
                        X_test = y_test = None
                    
                    # Reconstruire les dicts
                    train = [{"text": X_train[i, 0], "labels": y_train[i].tolist()} 
                            for i in range(len(X_train))]
                    val = [{"text": X_val[i, 0], "labels": y_val[i].tolist()} 
                          for i in range(len(X_val))]
                    test_data = [{"text": X_test[i, 0], "labels": y_test[i].tolist()} 
                                for i in range(len(X_test))] if X_test is not None else []
                    
                    logger.info("✅ Split stratifié multi-label réussi")
                    
                    # Vérifier la distribution après split
                    y_train_check = np.array([d["labels"] for d in train])
                    y_val_check = np.array([d["labels"] for d in val])
                    
                    logger.info(f"📊 Train - Positifs: {y_train_check.sum()}, Labels actifs: {(y_train_check.sum(axis=0) > 0).sum()}")
                    logger.info(f"📊 Val - Positifs: {y_val_check.sum()}, Labels actifs: {(y_val_check.sum(axis=0) > 0).sum()}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Split stratifié échoué: {e}, fallback split simple")
                    # Fallback
                    split_idx = int(len(data) * 0.8)
                    train = data[:split_idx]
                    val = data[split_idx:]
                    test_data = []
            else:
                # Split simple pour petits datasets
                logger.warning("⚠️ Dataset trop petit ou skmultilearn non installé, split simple")
                split_idx = int(len(data) * 0.8)
                train = data[:split_idx]
                val = data[split_idx:]
                test_data = []
                
        else:
            # Single-label : code existant
            can_stratify = self._check_dataset_balance(data)
            label_field = "label"
            
            if self.incremental_mode:
                # 🚀 NOUVEAU : Division en 3 parties pour l'apprentissage incrémental
                # 70% train, 20% validation, 10% test (pour évaluation baseline)
                if len(data) >= 10 and can_stratify:
                    # Split stratifié si possible
                    train_val, test_data = train_test_split(
                        data,
                        test_size=0.1,
                        stratify=[d[label_field] for d in data],
                        random_state=42,
                    )
                    
                    train, val = train_test_split(
                        train_val,
                        test_size=0.25,  # 0.25 de 90% = 22.5% du total ≈ 20%
                        stratify=[d[label_field] for d in train_val] if self._check_dataset_balance(train_val) else None,
                        random_state=42,
                    )
                else:
                    # Split simple pour petits datasets
                    logger.warning("⚠️ Dataset trop petit, utilisation de proportions adaptées")
                    
                    if len(data) >= 5:
                        # Au moins 5 échantillons : 1 pour test, reste pour train/val
                        test_data = data[:1]
                        train_val = data[1:]
                        
                        if len(train_val) >= 2:
                            train = train_val[:-1]
                            val = train_val[-1:]
                        else:
                            train = train_val
                            val = []
                    else:
                        # Très petit dataset : tout en train, pas de validation
                        train = data
                        val = []
                        test_data = []
                
                logger.info(f"📊 Mode incrémental - Train: {len(train)}, Validation: {len(val)}, Test: {len(test_data)}")
            else:
                # Mode classique (existant, conservé mais avec gestion des petits datasets)
                if len(data) >= 4 and can_stratify:
                    # Split stratifié si possible
                    train, val = train_test_split(
                        data,
                        test_size=0.2,
                        stratify=[d[label_field] for d in data],
                        random_state=42,
                    )
                else:
                    # Split simple pour petits datasets
                    logger.warning("⚠️ Dataset trop petit, déséquilibré, split simple sans stratification")
                    split_idx = max(1, int(len(data) * 0.8))
                    train = data[:split_idx]
                    val = data[split_idx:] if len(data) > split_idx else []
                
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
        
        # Gérer le cas où il n'y a pas de validation
        if val:
            val_ds = Dataset.from_list(val).map(
                tok, batched=True, remove_columns=["text"]
            )
        else:
            # Utiliser une partie du train comme validation si pas de val
            logger.warning("⚠️ Pas de données de validation, utilisation d'une partie du train")
            val_ds = train_ds
        
        # Dataset de test (pour évaluation baseline en mode incrémental)
        test_ds = Dataset.from_list(test_data) if test_data else None
        
        return DatasetDict(train=train_ds, validation=val_ds), test_ds

    # -------------------------------------------------------------------
    # Metrics (⚖️ NOUVEAU : Métriques adaptées aux datasets déséquilibrés + multi-label)
    # -------------------------------------------------------------------
    @staticmethod
    def _metrics(pred: EvalPrediction) -> Dict[str, float]:
        """Métriques standard (conservées pour compatibilité)"""
        logits, labels = pred
        preds = np.argmax(logits, axis=1)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    @staticmethod
    def _metrics_balanced(pred: EvalPrediction) -> Dict[str, float]:
        """⚖️ NOUVEAU : Métriques adaptées aux datasets déséquilibrés"""
        logits, labels = pred
        preds = np.argmax(logits, axis=1)
        
        # Métriques weighted (existant)
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )
        
        # ⚖️ NOUVEAU : Métriques macro (toutes les classes équivalentes)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        
        acc = accuracy_score(labels, preds)
        
        return {
            "accuracy": acc,
            "precision": prec_weighted,
            "recall": rec_weighted,
            "f1": f1_weighted,
            "f1_macro": f1_macro,  # ⚖️ Métrique principale pour datasets déséquilibrés
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
        }

    @staticmethod
    def _metrics_multilabel(pred: EvalPrediction) -> Dict[str, float]:
        """🌐 NOUVEAU : Métriques pour classification multi-label avec seuils adaptés"""
        logits, labels = pred
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
        
        # 🔧 PATCH : Seuil adaptatif plus bas pour labels rares
        threshold = 0.25  # Plus bas pour détecter les corrélations rares
        preds = probs > threshold
        
        # 🔧 PATCH : Fallback top-k si aucune prédiction
        k = 1  # Au moins 1 prédiction par échantillon
        rows_no_pred = np.where(preds.sum(axis=1) == 0)[0]
        for i in rows_no_pred:
            top_indices = probs[i].argsort()[-k:]
            preds[i, top_indices] = True
        
        # Métriques
        f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        
        prec_micro, rec_micro, _, _ = precision_recall_fscore_support(
            labels, preds, average='micro', zero_division=0
        )
        prec_macro, rec_macro, _, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        
        h_loss = hamming_loss(labels, preds)
        hamming_score = 1 - h_loss
        subset_acc = np.mean(np.all(preds == labels, axis=1))
        
        # 🔧 PATCH : Log distribution pour debug
        logger.info(f"📊 Seuil: {threshold}, Prédictions positives: {preds.sum()} / {preds.size}")
        logger.info(f"📊 Labels actifs prédits: {(preds.sum(axis=0) > 0).sum()} / {preds.shape[1]}")
        
        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision_micro": prec_micro,
            "precision_macro": prec_macro,
            "recall_micro": rec_micro,
            "recall_macro": rec_macro,
            "hamming_score": hamming_score,
            "hamming_loss": h_loss,
            "subset_accuracy": subset_acc,
            # Pour compatibilité
            "accuracy": subset_acc,
            "f1": f1_micro,
            "precision": prec_micro,
            "recall": rec_micro,
        }

    # -------------------------------------------------------------------
    # Training (adapté pour supporter l'incrémental + petits datasets + class balancing + multi-label)
    # -------------------------------------------------------------------
    def train(self, ds: DatasetDict, args: argparse.Namespace, test_ds: Dataset = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 🚀 NOUVEAU : Nommage différent selon le mode
        if self.incremental_mode:
            run_name = f"incremental-{self.task_type}-{getattr(args, 'mode', 'test')}-{ts}"
        else:
            run_name = f"finbert-{self.task_type}-{ts}"  # 🎯 Inclure task_type

        # ⚖️ NOUVEAU : Créer la fonction de loss pour class balancing (pas pour multi-label)
        custom_loss_fn = None
        pos_weight_tensor = None
        
        if self.is_multi_label and len(ds["train"]) > 0:
            # 🔧 PATCH : Calculer pos_weight par label pour multi-label
            y_train = np.array(ds["train"]["labels"], dtype=np.float32)
            
            # Calculer pos_weight par label
            pos_counts = y_train.sum(axis=0)
            neg_counts = y_train.shape[0] - pos_counts
            
            # Éviter division par zéro et clipper
            pos_weight = neg_counts / np.clip(pos_counts, 1.0, None)
            pos_weight = np.clip(pos_weight, 1.0, 50.0)
            
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
            
            logger.info(f"⚖️ Pos weights - Min: {pos_weight.min():.2f}, Max: {pos_weight.max():.2f}, Mean: {pos_weight.mean():.2f}")
            logger.info(f"⚖️ Labels sans positifs: {(pos_counts == 0).sum()}")
            
        elif self.class_balancing and not self.is_multi_label and len(ds["train"]) > 0:
            train_labels = ds["train"]["labels"]
            custom_loss_fn = self.create_loss_function(train_labels)
            if custom_loss_fn:
                logger.info(f"⚖️ Loss function personnalisée: {type(custom_loss_fn).__name__}")

        # 🚀 NOUVEAU : Arguments d'entraînement adaptés pour l'incrémental + petits datasets
        if self.incremental_mode:
            # Paramètres plus conservateurs pour l'apprentissage incrémental
            learning_rate = 1e-5  # Plus faible pour la stabilité
            epochs = min(2, args.epochs)  # Moins d'epochs pour éviter l'overfitting
            batch_size = min(4, args.train_bs)  # Plus petit batch pour petits datasets
            patience = 1  # Arrêt précoce plus agressif
        else:
            # Paramètres classiques adaptés pour petits datasets
            learning_rate = args.lr
            epochs = min(args.epochs, 5)  # Limiter les epochs pour petits datasets
            batch_size = min(8, args.train_bs)  # Adapter la taille de batch
            patience = 2

        # 🌐 NOUVEAU : Métrique principale selon le type
        if self.is_multi_label:
            primary_metric = "f1_micro"
        elif self.class_balancing:
            primary_metric = "f1_macro"
        else:
            primary_metric = "f1"

        # 🔧 CORRECTIF 2 : Rediriger output_dir vers repo_dir si disponible
        output_dir = str(self.repo_dir) if self.repo_dir else args.output_dir
        logger.info(f"📂 Répertoire d'entraînement: {output_dir}")

        targs = TrainingArguments(
            output_dir=output_dir,  # 🔧 CORRECTIF 2
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=min(batch_size, args.eval_bs),
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=min(args.warmup, len(ds["train"]) // 4),  # Adapter warmup
            evaluation_strategy="epoch" if len(ds["validation"]) > 0 else "no",
            save_strategy="epoch",
            load_best_model_at_end=len(ds["validation"]) > 0,
            metric_for_best_model=primary_metric if len(ds["validation"]) > 0 else None,
            greater_is_better=True,
            logging_dir=os.path.join(output_dir, "logs"),  # 🔧 CORRECTIF 2
            logging_steps=max(1, args.logging_steps // 10),  # Plus de logs pour petits datasets
            seed=args.seed,
            push_to_hub=False,  # Gestion manuelle du push
            report_to="tensorboard",
            save_total_limit=1,  # Économiser l'espace disque
            dataloader_drop_last=False,  # Ne pas ignorer les derniers échantillons
        )

        # 🌐 NOUVEAU : Déterminer la fonction de métriques
        if self.is_multi_label:
            compute_metrics_fn = self._metrics_multilabel
        else:
            compute_metrics_fn = self._metrics_balanced if len(ds["validation"]) > 0 else None

        # 🌐 NOUVEAU : CustomTrainer avec support multi-label
        if custom_loss_fn or self.is_multi_label:
            trainer = CustomTrainer(
                loss_fn=custom_loss_fn,
                model=self.model,
                args=targs,
                train_dataset=ds["train"],
                eval_dataset=ds["validation"] if len(ds["validation"]) > 0 else None,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics_fn,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)] if len(ds["validation"]) > 0 else [],
            )
            trainer.is_multi_label = self.is_multi_label  # 🌐 Passer l'info au trainer
            
            # 🔧 PATCH : Passer pos_weight au trainer pour multi-label
            if pos_weight_tensor is not None:
                trainer.pos_weight = pos_weight_tensor
        else:
            trainer = Trainer(
                model=self.model,
                args=targs,
                train_dataset=ds["train"],
                eval_dataset=ds["validation"] if len(ds["validation"]) > 0 else None,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics_fn,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)] if len(ds["validation"]) > 0 else [],
            )

        mode_info = f"incremental-{self.task_type}" if self.incremental_mode else f"classic-{self.task_type}"
        logger.info(f"🔥 Start training for %d epochs (mode: %s)", epochs, mode_info)
        trainer.train()
        
        # 🔧 MODIFICATION 3 : Sauvegarder directement dans le repo cloné
        save_dir = self.repo_dir if self.repo_dir else Path(args.output_dir)
        trainer.save_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"💾 Modèle sauvé dans: {save_dir}")

        if len(ds["validation"]) > 0:
            eval_res = trainer.evaluate()
            f1_score_val = eval_res.get(f"eval_{primary_metric}", eval_res.get("eval_f1", 0.0))
            logger.info(
                "✅ Training complete — F1: %.4f | Acc: %.4f",
                f1_score_val,
                eval_res.get("eval_accuracy", 0.0),
            )
        else:
            logger.info("✅ Training complete (no validation data)")
            eval_res = {"eval_f1": 0.0, "eval_accuracy": 0.0, f"eval_{primary_metric}": 0.0}

        # 🔧 MODIFICATION 3 : Générer/mettre à jour la model card
        if self.repo_dir:
            card_path = self.repo_dir / "README.md"
            f1m = eval_res.get(f"eval_{primary_metric}", 0)
            acc = eval_res.get("eval_accuracy", 0)
            
            # Informations sur le dataset
            dataset_name = args.dataset.name if hasattr(args.dataset, 'name') else str(args.dataset)
            dataset_size = len(ds["train"]) + len(ds["validation"])
            
            # Création de la model card complète
            card = f"""---
language: en
license: apache-2.0
tags:
- finance
- sentiment-analysis
- finbert
- trading
{"- multi-label" if self.is_multi_label else ""}
pipeline_tag: text-classification
---

# {self.hub_id}

## Description
Fine-tuned FinBERT model for financial {self.task_type} analysis in TradePulse.

**Task**: {self.task_type.title()} Classification  
**Target Column**: `{self.target_column}`  
{"**Multi-Label**: Yes (" + str(self.num_labels) + " labels)" if self.is_multi_label else "**Labels**: " + str(list(self.LABEL_MAP.keys()) if self.LABEL_MAP else [])}

## Performance

*Last training: {datetime.now().strftime('%Y-%m-%d %H:%M')}*  
*Dataset: `{dataset_name}` ({dataset_size} samples)*

| Metric | Value |
|--------|-------|
| Loss | {eval_res.get('eval_loss', 'n/a'):.4f} |
{"| Accuracy | " + f"{acc:.4f} |" if not self.is_multi_label else "| Subset Accuracy | " + f"{eval_res.get('eval_subset_accuracy', 0):.4f} |"}
| F1 Score | {eval_res.get('eval_f1', 0):.4f} |
{"| F1 Micro | " + f"{eval_res.get('eval_f1_micro', 0):.4f} |" if self.is_multi_label else ""}
| F1 Macro | {f1m:.4f} |
{"| Hamming Score | " + f"{eval_res.get('eval_hamming_score', 0):.4f} |" if self.is_multi_label else ""}
| Precision | {eval_res.get('eval_precision', 0):.4f} |
| Recall | {eval_res.get('eval_recall', 0):.4f} |

## Training Details

- **Base Model**: {self.model_name}
- **Training Mode**: {"Incremental" if self.incremental_mode else "Classic"}
- **Epochs**: {epochs}
- **Learning Rate**: {learning_rate}
- **Batch Size**: {batch_size}
- **Class Balancing**: {self.class_balancing or "None"}
{"- **Problem Type**: Multi-Label Classification" if self.is_multi_label else ""}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("{self.hub_id}")
model = AutoModelForSequenceClassification.from_pretrained("{self.hub_id}")

# Example prediction
text = "Apple reported strong quarterly earnings beating expectations"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
{"# Multi-label: apply sigmoid and threshold" if self.is_multi_label else ""}
predictions = {"torch.sigmoid(outputs.logits).squeeze() > 0.5" if self.is_multi_label else "outputs.logits.softmax(dim=-1)"}
```

## Model Card Authors

- TradePulse ML Team
- Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            card_path.write_text(card.strip(), encoding="utf-8")
            logger.info(f"📄 Model card mise à jour: {card_path}")

        # 🚀 NOUVEAU : Évaluation sur le dataset de test pour mode incrémental
        test_metrics = None
        if test_ds is not None and self.incremental_mode and len(test_ds) > 0:
            logger.info("📊 Évaluation sur dataset de test...")
            test_metrics = self.evaluate_on_test(test_ds)
            logger.info(f"📊 Métriques test: {test_metrics}")

        # save a report (adapté pour mode incrémental + importance + class balancing + corrélations)
        report = {
            "model": self.model_name,
            "task_type": self.task_type,  # 🎯 NOUVEAU
            "hub_id": self.hub_id,        # 🎯 NOUVEAU
            "mode": "incremental" if self.incremental_mode else "classic",
            "target_column": self.target_column,  # 🎯 NOUVEAU
            "is_multi_label": self.is_multi_label,  # 🌐 NOUVEAU
            "num_labels": self.num_labels,  # 🌐 NOUVEAU
            "label_mapping": dict(self.LABEL_MAP) if self.LABEL_MAP else {"multi_label": self.num_labels},  # 🎯 NOUVEAU
            "class_balancing": self.class_balancing,  # ⚖️ NOUVEAU
            "class_weights": self.class_weights.tolist() if self.class_weights is not None else None,  # ⚖️ NOUVEAU
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
# CLI (adapté pour supporter l'incrémental + importance + class balancing + corrélations)
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TradePulse FinBERT fine‑tuning utility avec apprentissage incrémental, support importance, corrélations et class balancing"
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
        "--hub_id", type=str, default=None, help="HF repo id (auto-détecté selon --target-column)"
    )
    
    # 🎯 NOUVEAU argument pour la colonne cible (ajout corrélations)
    p.add_argument("--target-column", choices=["label", "importance", "correlations"], default="label",
                   help="Colonne à utiliser pour l'entraînement (label=sentiment, importance=importance, correlations=multi-label)")
    
    # ⚖️ NOUVEAU : Arguments pour class balancing
    p.add_argument("--class-balancing", choices=["weighted", "focal"], default=None,
                   help="Type d'équilibrage des classes (weighted=pondération, focal=focal loss)")
    
    # 🚀 NOUVEAUX arguments pour l'apprentissage incrémental
    p.add_argument("--incremental", action="store_true", 
                   help="Activer l'apprentissage incrémental")
    p.add_argument("--mode", choices=["test", "development", "production"], default="production",
                   help="Mode incrémental (test/development/production)")
    p.add_argument("--baseline-model", type=str, default=None,
                   help="Modèle de référence pour l'incrémental (auto selon mode si non spécifié)")
    p.add_argument("--min-improvement", type=float, default=0.02,
                   help="Amélioration minimum requise pour mise à jour (défaut: 0.02)")
    p.add_argument("--force-update", action="store_true",
                   help="Forcer la mise à jour même sans amélioration significative")
    
    return p


# ---------------------------------------------------------------------------
# Entrée principale (adaptée pour supporter l'incrémental + importance + class balancing + corrélations)
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    # 🌐 NOUVEAU : Vérifier le support des corrélations
    if args.target_column == "correlations" and not CORRELATION_SUPPORT:
        logger.error("❌ Support des corrélations non disponible")
        logger.info("💡 Assurez-vous que config/correlation_mapping.py existe")
        return

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

    # 🎯 NOUVEAU : Gestion du baseline model automatique
    if args.incremental and getattr(args, 'baseline_model', None) is None:
        task_type = "correlations" if args.target_column == "correlations" else (
            "sentiment" if args.target_column == "label" else "importance"
        )
        try:
            args.baseline_model = MODELS_CONFIG[task_type][args.mode]["hf_id"]
        except KeyError:
            args.baseline_model = MODELS_CONFIG[task_type]["fallback"]
            logger.warning(f"⚠️ Modèle {task_type}/{args.mode} non trouvé, utilisation du fallback")

    # 🚀 NOUVEAU : Gestion du mode incrémental
    if args.incremental:
        logger.info("🔄 Mode apprentissage incrémental activé")
        
        logger.info(f"🎯 Mode: {args.mode}")
        logger.info(f"📦 Modèle de base: {args.baseline_model}")
        
        # Initialiser le fine-tuner en mode incrémental
        tuner = Finetuner(
            model_name=args.model_name, 
            max_length=args.max_length,
            incremental_mode=True,
            baseline_model=args.baseline_model,
            target_column=args.target_column,  # 🎯 NOUVEAU
            class_balancing=args.class_balancing,  # ⚖️ NOUVEAU
            mode=args.mode
        )
        
        # Chargement du dataset avec division train/val/test
        ds, test_ds = tuner.load_dataset(args.dataset)
        
        # Évaluation baseline du modèle existant (seulement si test_ds existe et non vide)
        baseline_metrics = None
        if test_ds is not None and len(test_ds) > 0:
            logger.info("📊 Évaluation baseline du modèle existant...")
            baseline_metrics = tuner.evaluate_on_test(test_ds)
            logger.info(f"📊 Métriques baseline: {baseline_metrics}")
        else:
            logger.warning("⚠️ Pas de dataset de test pour évaluation baseline")
            # Métriques factices pour le test
            baseline_metrics = {"accuracy": 0.5, "f1": 0.5, "f1_macro": 0.5, "precision": 0.5, "recall": 0.5}
            if tuner.is_multi_label:
                baseline_metrics.update({"f1_micro": 0.5, "hamming_score": 0.5})
        
        # Entraînement incrémental
        test_metrics = tuner.train(ds, args, test_ds)
        
        # Décision de mise à jour
        if test_metrics and baseline_metrics:
            logger.info("🔍 Analyse des résultats pour apprentissage incrémental...")
            
            should_update, reason = tuner.should_update_model(
                baseline_metrics, 
                test_metrics,
                getattr(args, 'min_improvement', 0.02)
            )
            
            logger.info(f"📊 Décision de mise à jour: {reason}")
            
            # Forcer la mise à jour si demandé
            if getattr(args, 'force_update', False) and not should_update:
                should_update = True
                reason = f"Mise à jour forcée (--force-update). {reason}"
                logger.warning(f"⚠️ {reason}")
            
            if should_update:
                # Mise à jour du modèle sur HuggingFace
                if args.mode in ["production", "development"]:
                    # 🌐 NOUVEAU : Message de commit adapté pour multi-label
                    if tuner.is_multi_label:
                        commit_msg = f"🔄 Incremental {args.target_column} | F1µ: {test_metrics.get('f1_micro', 0):.3f}, Hamming: {test_metrics.get('hamming_score', 0):.3f}"
                    else:
                        commit_msg = f"🔄 Incremental {args.target_column} | Acc: {test_metrics['accuracy']:.3f}, F1: {test_metrics.get('f1_macro', test_metrics['f1']):.3f}"
                    
                    logger.info(f"🚀 Mise à jour du modèle {args.target_column}")
                    tuner.push_to_huggingface(commit_msg)
                    
                    logger.info("✅ Modèle mis à jour!")
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
                    "hf_model_id": tuner.hub_id if should_update else None,
                })
                
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
        else:
            # Pas d'évaluation possible, forcer la mise à jour si demandé
            logger.warning("⚠️ Pas d'évaluation possible, mise à jour conditionnelle")
            should_update = getattr(args, 'force_update', False)
            
            if should_update and args.mode in ["production", "development"]:
                commit_msg = f"🔄 Incremental {args.target_column} - Dataset petit"
                
                logger.info(f"🚀 Mise à jour forcée du modèle {args.target_column}")
                tuner.push_to_huggingface(commit_msg)
                
                # Mise à jour du rapport
                report_path = args.output_dir / "incremental_training_report.json"
                if report_path.exists():
                    with open(report_path, "r") as f:
                        report = json.load(f)
                    
                    report.update({
                        "baseline_metrics": baseline_metrics,
                        "new_metrics": test_metrics or {"accuracy": 0.0, "f1": 0.0, "f1_macro": 0.0},
                        "model_updated": should_update,
                        "update_reason": "Force update - petit dataset",
                        "hf_model_id": tuner.hub_id,
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

        tuner = Finetuner(
            model_name=args.model_name, 
            max_length=args.max_length,
            target_column=args.target_column,  # 🎯 NOUVEAU
            class_balancing=args.class_balancing,  # ⚖️ NOUVEAU
            mode=args.mode
        )

        ds, _ = tuner.load_dataset(args.dataset)
        tuner.train(ds, args)

        # Push classique si demandé (existant, conservé mais modifié)
        if args.push:
            logger.info("📤 Push du modèle vers HuggingFace Hub...")
            commit_msg = f"🏋️ Classic training {args.target_column} | Dataset: {args.dataset.name if hasattr(args.dataset, 'name') else str(args.dataset)}"
            tuner.push_to_huggingface(commit_msg)


if __name__ == "__main__":
    main()
