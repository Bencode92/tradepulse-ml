#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradePulse Dataset Validation Tool
=================================

Valide la qualité et la cohérence des datasets d'entraînement FinBERT.

Usage:
    python scripts/validate_dataset.py datasets/news_20250705.csv
    python scripts/validate_dataset.py  # Auto-sélection du dernier dataset

Vérifications:
- Structure des colonnes (text, label)
- Labels valides (positive/neutral/negative)
- Détection des doublons
- Longueur des textes
- Distribution des classes
- Qualité du contenu
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Auto-sélection helper
try:
    from utils import latest_dataset

    AUTOSEL = True
except ImportError:
    AUTOSEL = False

# Configuration des logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("dataset-validator")

# Regex pour validation des labels
RE_LABEL = re.compile(r"^(negative|neutral|positive)$", re.IGNORECASE)


def numpy_json_encoder(obj: Any) -> Any:
    """Encoder personnalisé pour sérialiser les types numpy en JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    elif hasattr(obj, "__int__"):
        return int(obj)
    elif hasattr(obj, "__float__"):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class ValidationError:
    """Représente une erreur de validation avec contexte détaillé"""

    def __init__(
        self,
        type_: str,
        message: str,
        line_number: Optional[int] = None,
        field: Optional[str] = None,
        severity: str = "error",
    ):
        self.type = type_
        self.message = message
        self.line_number = line_number
        self.field = field
        self.severity = severity  # "error" or "warning"

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "message": self.message,
            "line_number": self.line_number,
            "field": self.field,
            "severity": self.severity,
        }

    def __str__(self) -> str:
        prefix = "❌" if self.severity == "error" else "⚠️"
        line_info = f" (ligne {self.line_number})" if self.line_number else ""
        return f"{prefix} {self.message}{line_info}"


class DatasetValidator:
    def __init__(self, max_length: int = 512, min_samples: int = 10):
        self.max_length = max_length
        self.min_samples = min_samples
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.overrep_threshold = 0.7  # 70% threshold for overrepresented class

    def add_error(
        self,
        type_: str,
        message: str,
        line_number: Optional[int] = None,
        field: Optional[str] = None,
    ):
        """Ajoute une erreur critique"""
        error = ValidationError(type_, message, line_number, field, "error")
        self.errors.append(error)
        logger.error(str(error))

    def add_warning(
        self,
        type_: str,
        message: str,
        line_number: Optional[int] = None,
        field: Optional[str] = None,
    ):
        """Ajoute un avertissement"""
        warning = ValidationError(type_, message, line_number, field, "warning")
        self.warnings.append(warning)
        logger.warning(str(warning))

    def _add_warning(
        self, type_: str, message: str, line_number: Optional[int] = None
    ):
        """Helper method for backward compatibility"""
        self.add_warning(type_, message, line_number)

    def validate(self, csv_path: Path) -> Tuple[bool, Dict]:
        """Valide un dataset CSV et retourne (succès, rapport)"""
        logger.info(f"🔍 Validation de: {csv_path}")

        if not csv_path.exists():
            self.add_error("file_not_found", f"Fichier introuvable: {csv_path}")
            return False, self._generate_report()

        try:
            # CORRECTION: Configuration pour tests line accuracy
            df = pd.read_csv(csv_path, keep_default_na=False)
        except Exception as e:
            self.add_error("csv_parse_error", f"Erreur lecture CSV: {e}")
            return False, self._generate_report()

        # Série de validations avec numéros de ligne
        self._check_structure(df)
        self._check_content_detailed(df)
        self._check_class_distribution(df)
        self._check_quality_detailed(df)

        success = len(self.errors) == 0
        report = self._generate_report(df if success else None)

        return success, report

    def _check_structure(self, df: pd.DataFrame):
        """Vérifie la structure du DataFrame"""
        expected_cols = {"text", "label"}
        actual_cols = set(df.columns)

        if actual_cols != expected_cols:
            missing = expected_cols - actual_cols
            extra = actual_cols - expected_cols

            if missing:
                self.add_error(
                    "missing_columns", f"Colonnes manquantes: {missing}"
                )
            if extra:
                self.add_warning(
                    "extra_columns", f"Colonnes supplémentaires: {extra}"
                )

        if df.empty:
            self.add_error("empty_dataset", "Dataset vide")
            return

        if len(df) < self.min_samples:
            self.add_warning(
                "insufficient_samples",
                f"Peu d'échantillons: {len(df)} < {self.min_samples}",
            )

    def _check_content_detailed(self, df: pd.DataFrame):
        """Vérifie le contenu avec détails ligne par ligne - CORRECTION NUMÉROTATION"""
        if "text" not in df.columns or "label" not in df.columns:
            return  # Déjà signalé dans _check_structure

        # Vérification des valeurs manquantes ligne par ligne
        for idx, row in df.iterrows():
            real_line = idx + 2  # +1 pour passer à 1-based, +1 pour l'en-tête

            # CORRECTION : Logique simplifiée avec keep_default_na=False
            text_value = row.get("text")

            # Avec keep_default_na=False : cellules vides deviennent ""
            if text_value == "" or (isinstance(text_value, str) and text_value.strip() == ""):
                self.add_error(
                    "empty_text", f"Texte vide", real_line, "text"
                )
            elif pd.isnull(text_value) or str(text_value).lower() == "nan":
                # Cas vraiment rares avec keep_default_na=False
                self.add_error(
                    "missing_text", f"Texte manquant", real_line, "text"
                )

            # Vérification des labels
            label_value = row.get("label")
            if pd.isnull(label_value) or label_value == "":
                self.add_error(
                    "missing_label", f"Label manquant", real_line, "label"
                )
            else:
                label_str = str(label_value).strip()
                if not RE_LABEL.match(label_str):
                    self.add_error(
                        "invalid_label",
                        f"Label invalide: '{label_value}' "
                        f"(doit être positive/negative/neutral)",
                        real_line,
                        "label",
                    )

        # Détection des doublons avec numéros de ligne
        if "text" in df.columns:
            duplicated_mask = df["text"].duplicated(keep=False)
            if duplicated_mask.any():
                duplicate_groups = df[duplicated_mask].groupby("text")
                for text, group in duplicate_groups:
                    line_numbers = [idx + 2 for idx in group.index]  # Cohérent : +2
                    self.add_warning(
                        "duplicate_text",
                        f"Texte dupliqué aux lignes {line_numbers}: "
                        f"'{text[:50]}...'",
                    )

    def _check_class_distribution(self, df: pd.DataFrame):
        """Vérifie la distribution des classes"""
        if "label" not in df.columns:
            return

        # Filtrer les labels valides pour les stats
        valid_labels = df[
            df["label"].astype(str).str.match(RE_LABEL, na=False)
        ]
        if valid_labels.empty:
            self.add_error("no_valid_labels", "Aucun label valide trouvé")
            return

        label_counts = valid_labels["label"].value_counts()
        total = len(valid_labels)

        # Vérification de l'équilibre des classes avec seuil 70%
        for label in ["positive", "negative", "neutral"]:
            count = label_counts.get(label, 0)
            percentage = (count / total) if total > 0 else 0

            if count == 0:
                self.add_warning("missing_class", f"Aucun exemple '{label}'")
            elif percentage < 0.1:  # Moins de 10%
                self.add_warning(
                    "underrepresented_class",
                    f"Classe '{label}' sous-représentée: "
                    f"{count} ({percentage:.1%})",
                )
            elif percentage >= self.overrep_threshold:  # 70% threshold
                self.add_warning(
                    "overrepresented_class",
                    f"Classe '{label}' sur-représentée: "
                    f"{count} ({percentage:.1%})",
                )

    def _check_quality_detailed(self, df: pd.DataFrame):
        """Vérifie la qualité du contenu avec détails - CORRECTION NUMÉROTATION"""
        if "text" not in df.columns:
            return

        for idx, row in df.iterrows():
            if pd.isnull(row["text"]) or row["text"] == "":
                continue

            text = str(row["text"])
            real_line = idx + 2  # +1 pour passer à 1-based, +1 pour l'en-tête

            # Textes très courts
            if len(text.strip()) < 10:
                self.add_warning(
                    "very_short_text",
                    f"Texte très court ({len(text)} caractères)",
                    real_line,
                    "text",
                )

            # Textes très longs
            if len(text) > self.max_length:
                self.add_warning(
                    "text_too_long",
                    f"Texte trop long ({len(text)} > {self.max_length} "
                    "caractères)",
                    real_line,
                    "text",
                )

            # Caractères suspects (non-texte)
            if re.search(
                r"[^\w\s\.\,\!\?\"\'\'\-\(\)\:\;\%\$\+\=\&\@\#]", text
            ):
                self.add_warning(
                    "suspicious_characters",
                    f"Caractères spéciaux détectés",
                    real_line,
                    "text",
                )

            # Texte en majuscules (potentiel spam)
            if len(text) > 20 and text.isupper():
                self.add_warning(
                    "all_caps_text",
                    f"Texte entièrement en majuscules",
                    real_line,
                    "text",
                )

    def _generate_report(self, df: pd.DataFrame = None) -> Dict:
        """Génère un rapport de validation détaillé avec conversion numpy"""
        report = {
            "validation_success": len(self.errors) == 0,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "error_count": int(len(self.errors)),  # Conversion explicite
            "warning_count": int(len(self.warnings)),  # Conversion explicite
            "statistics": {},
        }

        if df is not None and not df.empty:
            # Statistiques sur les labels valides seulement
            valid_labels = (
                df[df["label"].astype(str).str.match(RE_LABEL, na=False)]
                if "label" in df.columns
                else pd.DataFrame()
            )

            # Conversion explicite des types numpy en types Python natifs
            text_lengths = (
                df["text"].str.len()
                if "text" in df.columns
                else pd.Series(dtype=float)
            )

            report["statistics"] = {
                "total_samples": int(len(df)),
                "valid_samples": int(len(valid_labels)),
                "avg_text_length": float(text_lengths.mean())
                if not text_lengths.empty
                else 0.0,
                "max_text_length": int(text_lengths.max())
                if not text_lengths.empty
                else 0,
                "min_text_length": int(text_lengths.min())
                if not text_lengths.empty
                else 0,
                "label_distribution": {
                    str(k): int(v)
                    for k, v in valid_labels["label"]
                    .value_counts()
                    .to_dict()
                    .items()
                }
                if not valid_labels.empty
                else {},
                "duplicates": int(df["text"].duplicated().sum())
                if "text" in df.columns
                else 0,
            }

        return report

    def print_report(self, report: Dict):
        """Affiche le rapport de validation"""
        print("\n" + "=" * 60)
        print("🔍 RAPPORT DE VALIDATION DATASET")
        print("=" * 60)

        # Résumé
        print(f"\n📊 RÉSUMÉ:")
        print(f"  Erreurs critiques: {report['error_count']}")
        print(f"  Avertissements: {report['warning_count']}")

        # Erreurs détaillées
        if report["errors"]:
            print(f"\n❌ ERREURS CRITIQUES ({len(report['errors'])})")
            for error in report["errors"]:
                line_info = (
                    f" (ligne {error['line_number']})"
                    if error["line_number"]
                    else ""
                )
                print(f"  • {error['message']}{line_info}")

        # Avertissements détaillés
        if report["warnings"]:
            print(f"\n⚠️ AVERTISSEMENTS ({len(report['warnings'])})")
            for warning in report["warnings"]:
                line_info = (
                    f" (ligne {warning['line_number']})"
                    if warning["line_number"]
                    else ""
                )
                print(f"  • {warning['message']}{line_info}")

        # Statistiques
        if report["statistics"]:
            stats = report["statistics"]
            print(f"\n📈 STATISTIQUES:")
            print(f"  Total échantillons: {stats['total_samples']}")
            print(f"  Échantillons valides: {stats['valid_samples']}")
            if stats["valid_samples"] > 0:
                print(
                    f"  Longueur moyenne: {stats['avg_text_length']:.1f} "
                    "caractères"
                )
                print(
                    f"  Longueur min/max: {stats['min_text_length']}/"
                    f"{stats['max_text_length']}"
                )
                print(f"  Doublons: {stats['duplicates']}")

                print(f"\n📊 DISTRIBUTION DES LABELS:")
                for label, count in stats["label_distribution"].items():
                    percentage = (count / stats["valid_samples"]) * 100
                    print(f"  {label}: {count} ({percentage:.1f}%)")

        # Résultat final
        status = (
            "✅ VALIDATION RÉUSSIE"
            if report["validation_success"]
            else "❌ VALIDATION ÉCHOUÉE"
        )
        print(f"\n{status}")
        print("=" * 60)

    def save_errors_for_pr(
        self, output_file: Path = Path("validation_errors.txt")
    ):
        """Sauve les erreurs dans un format pour commentaire PR"""
        if not self.errors and not self.warnings:
            return

        lines = []

        if self.errors:
            lines.append(f"❌ {len(self.errors)} ERREUR(S) CRITIQUE(S):")
            lines.append("")
            for error in self.errors:
                line_info = (
                    f" (ligne {error.line_number})" if error.line_number else ""
                )
                lines.append(f"• {error.message}{line_info}")
            lines.append("")

        if self.warnings:
            lines.append(f"⚠️ {len(self.warnings)} AVERTISSEMENT(S):")
            lines.append("")
            for warning in self.warnings:
                line_info = (
                    f" (ligne {warning.line_number})"
                    if warning.line_number
                    else ""
                )
                lines.append(f"• {warning.message}{line_info}")
            lines.append("")

        lines.append(
            "💡 Corrigez ces problèmes avant de merger la Pull Request."
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    # Auto-sélection si aucun arg fourni
    if len(sys.argv) == 1 and AUTOSEL:
        ds = latest_dataset()
        if ds:
            print(f"🕵️  Auto-sélection : {ds}")
            sys.argv.append(str(ds))
        else:
            print("❌ Aucun news_*.csv trouvé")
            print(
                "💡 Ajoutez des fichiers au format news_YYYYMMDD.csv dans "
                "datasets/"
            )
            return

    parser = argparse.ArgumentParser(
        description="Validation de datasets TradePulse FinBERT"
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        nargs="?",  # Rendre optionnel
        help="Chemin vers le fichier CSV à valider (auto-détection si omis)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Longueur maximale des textes (défaut: 512)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Nombre minimum d'échantillons requis (défaut: 10)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Mode silencieux (seulement exit code)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Sauvegarder le rapport au format JSON",
    )
    parser.add_argument(
        "--save-pr-errors",
        action="store_true",
        help="Sauvegarder les erreurs pour commentaire PR",
    )

    args = parser.parse_args()

    # Si pas de dataset_path après parsing, essayer auto-sélection
    if args.dataset_path is None and AUTOSEL:
        args.dataset_path = latest_dataset()
        if args.dataset_path:
            print(f"🕵️  Auto-sélection : {args.dataset_path}")
        else:
            print("❌ Aucun dataset trouvé")
            print(
                "💡 Ajoutez des fichiers au format news_YYYYMMDD.csv dans "
                "datasets/"
            )
            sys.exit(1)
    elif args.dataset_path is None:
        print("❌ Aucun dataset spécifié et auto-sélection non disponible")
        print(
            "💡 Utilisez: python scripts/validate_dataset.py "
            "datasets/votre_fichier.csv"
        )
        sys.exit(1)

    validator = DatasetValidator(
        max_length=args.max_length, min_samples=args.min_samples
    )

    success, report = validator.validate(args.dataset_path)

    if not args.quiet:
        validator.print_report(report)

    # Sauvegarder le rapport JSON si demandé avec encoder personnalisé
    if args.output_json:
        try:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(
                    report,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=numpy_json_encoder,
                )
            print(f"\n📄 Rapport sauvegardé: {args.output_json}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde JSON: {e}")
            # Ne pas faire échouer le script pour ça

    # Sauvegarder les erreurs pour PR si demandé
    if args.save_pr_errors:
        try:
            validator.save_errors_for_pr()
            print(f"\n📝 Erreurs PR sauvegardées: validation_errors.txt")
        except Exception as e:
            logger.error(f"Erreur sauvegarde erreurs PR: {e}")

    # Exit code pour intégration CI/CD
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
