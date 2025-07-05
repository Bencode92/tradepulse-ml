#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradePulse Dataset Validation Tool
=================================

Valide la qualité et la cohérence des datasets d'entraînement FinBERT.

Usage:
    python scripts/validate_dataset.py datasets/news_20250705.csv

Vérifications:
- Structure des colonnes (text, label)
- Labels valides (positive/neutral/negative)
- Détection des doublons
- Longueur des textes
- Distribution des classes
- Qualité du contenu
"""

import sys
import pandas as pd
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("dataset-validator")

# Regex pour validation des labels
RE_LABEL = re.compile(r"^(negative|neutral|positive)$", re.IGNORECASE)

class DatasetValidator:
    def __init__(self, max_length: int = 512, min_samples: int = 10):
        self.max_length = max_length
        self.min_samples = min_samples
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate(self, csv_path: Path) -> Tuple[bool, Dict]:
        """Valide un dataset CSV et retourne (succès, rapport)"""
        logger.info(f"🔍 Validation de: {csv_path}")
        
        if not csv_path.exists():
            self.errors.append(f"❌ Fichier introuvable: {csv_path}")
            return False, self._generate_report()
            
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self.errors.append(f"❌ Erreur lecture CSV: {e}")
            return False, self._generate_report()
            
        # Série de validations
        self._check_structure(df)
        self._check_content(df)
        self._check_distribution(df)
        self._check_quality(df)
        
        success = len(self.errors) == 0
        report = self._generate_report(df if success else None)
        
        return success, report
    
    def _check_structure(self, df: pd.DataFrame):
        """Vérifie la structure du DataFrame"""
        expected_cols = {"text", "label"}
        actual_cols = set(df.columns)
        
        if actual_cols != expected_cols:
            self.errors.append(
                f"❌ Colonnes incorrectes. Attendu: {expected_cols}, "
                f"Trouvé: {actual_cols}"
            )
            
        if df.empty:
            self.errors.append("❌ Dataset vide")
            return
            
        if len(df) < self.min_samples:
            self.warnings.append(
                f"⚠️ Peu d'échantillons: {len(df)} < {self.min_samples}"
            )
    
    def _check_content(self, df: pd.DataFrame):
        """Vérifie le contenu des données"""
        if "text" not in df.columns or "label" not in df.columns:
            return  # Déjà signalé dans _check_structure
            
        # Vérification des valeurs manquantes
        null_texts = df["text"].isnull().sum()
        null_labels = df["label"].isnull().sum()
        
        if null_texts:
            self.errors.append(f"❌ {null_texts} textes manquants")
        if null_labels:
            self.errors.append(f"❌ {null_labels} labels manquants")
            
        # Vérification des textes vides
        empty_texts = (df["text"].str.strip() == "").sum()
        if empty_texts:
            self.errors.append(f"❌ {empty_texts} textes vides")
            
        # Vérification des labels valides
        invalid_labels = ~df["label"].astype(str).str.match(RE_LABEL, na=False)
        bad_labels = df[invalid_labels]
        
        if not bad_labels.empty:
            unique_bad = bad_labels["label"].unique()
            self.errors.append(
                f"❌ {len(bad_labels)} labels invalides: {list(unique_bad)}"
            )
            
        # Détection des doublons
        duplicates = df["text"].duplicated().sum()
        if duplicates:
            self.warnings.append(f"⚠️ {duplicates} textes dupliqués détectés")
            
        # Textes trop longs
        long_texts = df["text"].str.len() > self.max_length
        if long_texts.any():
            count = long_texts.sum()
            max_len = df["text"].str.len().max()
            self.warnings.append(
                f"⚠️ {count} textes > {self.max_length} caractères "
                f"(max: {max_len})"
            )
    
    def _check_distribution(self, df: pd.DataFrame):
        """Vérifie la distribution des classes"""
        if "label" not in df.columns:
            return
            
        label_counts = df["label"].value_counts()
        total = len(df)
        
        # Vérification de l'équilibre des classes
        for label in ["positive", "negative", "neutral"]:
            count = label_counts.get(label, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            
            if count == 0:
                self.warnings.append(f"⚠️ Aucun exemple '{label}'")
            elif percentage < 15:  # Moins de 15% de la distribution
                self.warnings.append(
                    f"⚠️ Classe '{label}' sous-représentée: "
                    f"{count} ({percentage:.1f}%)"
                )
            elif percentage > 70:  # Plus de 70% de la distribution
                self.warnings.append(
                    f"⚠️ Classe '{label}' sur-représentée: "
                    f"{count} ({percentage:.1f}%)"
                )
    
    def _check_quality(self, df: pd.DataFrame):
        """Vérifie la qualité du contenu"""
        if "text" not in df.columns:
            return
            
        # Textes très courts (potentiellement non informatifs)
        short_texts = df["text"].str.len() < 20
        if short_texts.any():
            count = short_texts.sum()
            self.warnings.append(f"⚠️ {count} textes très courts (< 20 caractères)")
            
        # Détection de caractères suspects
        special_chars = df["text"].str.contains(r'[^\w\s\.\,\!\?\"\'\-\(\)\:\;]', regex=True, na=False)
        if special_chars.any():
            count = special_chars.sum()
            self.warnings.append(f"⚠️ {count} textes avec caractères spéciaux")
    
    def _generate_report(self, df: pd.DataFrame = None) -> Dict:
        """Génère un rapport de validation"""
        report = {
            "validation_success": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": {}
        }
        
        if df is not None:
            report["statistics"] = {
                "total_samples": len(df),
                "avg_text_length": df["text"].str.len().mean(),
                "max_text_length": df["text"].str.len().max(),
                "min_text_length": df["text"].str.len().min(),
                "label_distribution": df["label"].value_counts().to_dict(),
                "duplicates": df["text"].duplicated().sum()
            }
            
        return report
    
    def print_report(self, report: Dict):
        """Affiche le rapport de validation"""
        print("\n" + "="*50)
        print("🔍 RAPPORT DE VALIDATION DATASET")
        print("="*50)
        
        # Erreurs
        if report["errors"]:
            print("\n❌ ERREURS CRITIQUES:")
            for error in report["errors"]:
                print(f"  {error}")
                
        # Warnings
        if report["warnings"]:
            print("\n⚠️ AVERTISSEMENTS:")
            for warning in report["warnings"]:
                print(f"  {warning}")
                
        # Statistiques
        if report["statistics"]:
            stats = report["statistics"]
            print(f"\n📊 STATISTIQUES:")
            print(f"  Total échantillons: {stats['total_samples']}")
            print(f"  Longueur moyenne: {stats['avg_text_length']:.1f} caractères")
            print(f"  Longueur min/max: {stats['min_text_length']}/{stats['max_text_length']}")
            print(f"  Doublons: {stats['duplicates']}")
            
            print(f"\n📈 DISTRIBUTION DES LABELS:")
            for label, count in stats['label_distribution'].items():
                percentage = (count / stats['total_samples']) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Résultat final
        status = "✅ VALIDATION RÉUSSIE" if report["validation_success"] else "❌ VALIDATION ÉCHOUÉE"
        print(f"\n{status}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description="Validation de datasets TradePulse FinBERT"
    )
    parser.add_argument(
        "dataset_path", 
        type=Path,
        help="Chemin vers le fichier CSV à valider"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=512,
        help="Longueur maximale des textes (défaut: 512)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Nombre minimum d'échantillons requis (défaut: 10)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Mode silencieux (seulement exit code)"
    )
    
    args = parser.parse_args()
    
    validator = DatasetValidator(
        max_length=args.max_length,
        min_samples=args.min_samples
    )
    
    success, report = validator.validate(args.dataset_path)
    
    if not args.quiet:
        validator.print_report(report)
    
    # Exit code pour intégration CI/CD
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
