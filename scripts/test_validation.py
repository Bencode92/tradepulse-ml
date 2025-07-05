#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires pour le script de validation des datasets TradePulse
==================================================================

Usage:
    python scripts/test_validation.py
    pytest scripts/test_validation.py -v

Tests:
- Datasets valides et invalides
- Détection d'erreurs ligne par ligne
- Formats CSV et JSON
- Métriques et rapports
"""

import pytest
import tempfile
import json
from pathlib import Path
import pandas as pd
import sys
import os

# Ajouter le dossier scripts au path pour importer validate_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validate_dataset import DatasetValidator, ValidationError

class TestDatasetValidator:
    
    def setup_method(self):
        """Setup avant chaque test"""
        self.validator = DatasetValidator(max_length=100, min_samples=5)
        
    def create_temp_csv(self, content: str) -> Path:
        """Crée un fichier CSV temporaire"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)
    
    def test_valid_dataset(self):
        """Test avec un dataset parfaitement valide"""
        content = """text,label
"Apple stock rose 5% after earnings beat",positive
"Market volatility increased amid uncertainty",negative
"Oil prices remained stable today",neutral
"Tesla announced new factory construction",positive
"Fed maintained interest rates unchanged",neutral
"Crypto markets faced selling pressure",negative"""
        
        csv_path = self.create_temp_csv(content)
        try:
            success, report = self.validator.validate(csv_path)
            
            assert success == True
            assert report["validation_success"] == True
            assert report["error_count"] == 0
            assert len(report["errors"]) == 0
            assert report["statistics"]["total_samples"] == 6
            assert report["statistics"]["valid_samples"] == 6
            
            # Vérifier la distribution
            distribution = report["statistics"]["label_distribution"]
            assert distribution["positive"] == 2
            assert distribution["negative"] == 2
            assert distribution["neutral"] == 2
            
        finally:
            csv_path.unlink()
    
    def test_missing_columns(self):
        """Test avec colonnes manquantes"""
        content = """title,sentiment
"Apple earnings beat expectations",good
"Market declined today",bad"""
        
        csv_path = self.create_temp_csv(content)
        try:
            success, report = self.validator.validate(csv_path)
            
            assert success == False
            assert report["error_count"] > 0
            
            # Vérifier qu'on a bien une erreur de colonnes manquantes
            error_types = [error["type"] for error in report["errors"]]
            assert "missing_columns" in error_types
            
        finally:
            csv_path.unlink()
    
    def test_invalid_labels(self):
        """Test avec labels invalides"""
        content = """text,label
"Apple stock rose",positive
"Market declined",bad
"Oil stable",good
"Tesla news",neutral
"Fed decision",unknown"""
        
        csv_path = self.create_temp_csv(content)
        try:
            success, report = self.validator.validate(csv_path)
            
            assert success == False
            assert report["error_count"] > 0
            
            # Vérifier les erreurs de labels invalides
            invalid_label_errors = [
                error for error in report["errors"] 
                if error["type"] == "invalid_label"
            ]
            assert len(invalid_label_errors) == 3  # bad, good, unknown
            
            # Vérifier les numéros de ligne
            line_numbers = [error["line_number"] for error in invalid_label_errors]
            assert 3 in line_numbers  # "bad" à la ligne 3
            assert 4 in line_numbers  # "good" à la ligne 4
            assert 6 in line_numbers  # "unknown" à la ligne 6
            
        finally:
            csv_path.unlink()
    
    def test_empty_and_missing_text(self):
        """Test avec textes vides et manquants"""
        content = """text,label
"Apple earnings beat",positive
"",negative
,neutral
"   ",positive
"Tesla news",negative"""
        
        csv_path = self.create_temp_csv(content)
        try:
            success, report = self.validator.validate(csv_path)
            
            assert success == False
            assert report["error_count"] > 0
            
            # Vérifier les types d'erreurs de texte
            error_types = [error["type"] for error in report["errors"]]
            assert "empty_text" in error_types
            assert "missing_text" in error_types
            
        finally:
            csv_path.unlink()
    
    def test_duplicate_detection(self):
        """Test de détection des doublons"""
        content = """text,label
"Apple stock rose 5%",positive
"Market declined today",negative
"Apple stock rose 5%",positive
"Oil prices stable",neutral
"Market declined today",negative"""
        
        csv_path = self.create_temp_csv(content)
        try:
            success, report = self.validator.validate(csv_path)
            
            # Les doublons sont des warnings, pas des erreurs critiques
            assert report["warning_count"] > 0
            
            # Vérifier qu'on a bien des warnings de doublons
            warning_types = [warning["type"] for warning in report["warnings"]]
            assert "duplicate_text" in warning_types
            
            # Les stats devraient indiquer des doublons
            assert report["statistics"]["duplicates"] == 2
            
        finally:
            csv_path.unlink()
    
    def test_text_length_validation(self):
        """Test de validation de longueur des textes"""
        # Texte très long (> max_length=100)
        long_text = "A" * 150
        
        content = f"""text,label
"Short text",positive
"{long_text}",negative
"X",neutral
"Normal length text here",positive"""
        
        csv_path = self.create_temp_csv(content)
        try:
            success, report = self.validator.validate(csv_path)
            
            # Vérifier les warnings de longueur
            warning_types = [warning["type"] for warning in report["warnings"]]
            assert "text_too_long" in warning_types  # Texte trop long
            assert "very_short_text" in warning_types  # "X" est trop court
            
        finally:
            csv_path.unlink()
    
    def test_class_distribution_warnings(self):
        """Test des warnings de distribution déséquilibrée"""
        content = """text,label
"Text 1",positive
"Text 2",positive
"Text 3",positive
"Text 4",positive
"Text 5",positive
"Text 6",positive
"Text 7",positive
"Text 8",positive
"Text 9",negative
"Text 10",neutral"""
        
        csv_path = self.create_temp_csv(content)
        try:
            success, report = self.validator.validate(csv_path)
            
            # Devrait réussir mais avec des warnings
            assert success == True  # Pas d'erreurs critiques
            assert report["warning_count"] > 0
            
            # Vérifier les warnings de distribution
            warning_types = [warning["type"] for warning in report["warnings"]]
            assert "overrepresented_class" in warning_types  # positive sur-représentée
            
        finally:
            csv_path.unlink()
    
    def test_insufficient_samples(self):
        """Test avec trop peu d'échantillons"""
        content = """text,label
"Text 1",positive
"Text 2",negative"""
        
        csv_path = self.create_temp_csv(content)
        try:
            success, report = self.validator.validate(csv_path)
            
            # Devrait réussir mais avec warning
            assert report["warning_count"] > 0
            
            warning_types = [warning["type"] for warning in report["warnings"]]
            assert "insufficient_samples" in warning_types
            
        finally:
            csv_path.unlink()
    
    def test_error_line_numbers(self):
        """Test que les numéros de ligne sont corrects"""
        content = """text,label
"Valid text",positive
"",invalid_label
,positive
"Another valid",negative"""
        
        csv_path = self.create_temp_csv(content)
        try:
            success, report = self.validator.validate(csv_path)
            
            assert success == False
            
            # Vérifier les numéros de ligne spécifiques
            errors = report["errors"]
            
            # Trouver l'erreur de label invalide (ligne 3)
            invalid_label_error = next(
                (e for e in errors if e["type"] == "invalid_label"), None
            )
            assert invalid_label_error is not None
            assert invalid_label_error["line_number"] == 3
            
            # Trouver l'erreur de texte manquant (ligne 4) 
            missing_text_error = next(
                (e for e in errors if e["type"] == "missing_text"), None
            )
            assert missing_text_error is not None
            assert missing_text_error["line_number"] == 4
            
        finally:
            csv_path.unlink()
    
    def test_file_not_found(self):
        """Test avec fichier inexistant"""
        fake_path = Path("nonexistent_file.csv")
        success, report = self.validator.validate(fake_path)
        
        assert success == False
        assert report["error_count"] > 0
        
        error_types = [error["type"] for error in report["errors"]]
        assert "file_not_found" in error_types
    
    def test_empty_dataset(self):
        """Test avec dataset vide"""
        content = "text,label"  # Juste l'en-tête
        
        csv_path = self.create_temp_csv(content)
        try:
            success, report = self.validator.validate(csv_path)
            
            assert success == False
            assert report["error_count"] > 0
            
            error_types = [error["type"] for error in report["errors"]]
            assert "empty_dataset" in error_types
            
        finally:
            csv_path.unlink()
    
    def test_save_errors_for_pr(self):
        """Test de sauvegarde des erreurs pour PR"""
        content = """text,label
"Valid text",positive
"",bad_label
,positive"""
        
        csv_path = self.create_temp_csv(content)
        temp_error_file = Path("test_validation_errors.txt")
        
        try:
            success, report = self.validator.validate(csv_path)
            assert success == False
            
            # Sauvegarder les erreurs
            self.validator.save_errors_for_pr(temp_error_file)
            
            # Vérifier que le fichier existe et contient les erreurs
            assert temp_error_file.exists()
            
            content = temp_error_file.read_text(encoding="utf-8")
            assert "ERREUR(S) CRITIQUE(S)" in content
            assert "ligne" in content  # Doit contenir des numéros de ligne
            
        finally:
            csv_path.unlink()
            if temp_error_file.exists():
                temp_error_file.unlink()
    
    def test_json_output(self):
        """Test de la sortie JSON"""
        content = """text,label
"Valid text",positive
"Another text",negative"""
        
        csv_path = self.create_temp_csv(content)
        try:
            success, report = self.validator.validate(csv_path)
            
            # Vérifier que le rapport est sérialisable en JSON
            json_str = json.dumps(report)
            parsed_report = json.loads(json_str)
            
            assert parsed_report["validation_success"] == success
            assert "statistics" in parsed_report
            assert "errors" in parsed_report
            assert "warnings" in parsed_report
            
        finally:
            csv_path.unlink()


def run_manual_tests():
    """Lance les tests manuellement sans pytest"""
    print("🧪 Lancement des tests de validation...")
    
    test_cases = [
        ("test_valid_dataset", "Dataset valide"),
        ("test_missing_columns", "Colonnes manquantes"), 
        ("test_invalid_labels", "Labels invalides"),
        ("test_empty_and_missing_text", "Textes vides"),
        ("test_duplicate_detection", "Détection doublons"),
        ("test_text_length_validation", "Validation longueur"),
        ("test_class_distribution_warnings", "Distribution classes"),
        ("test_insufficient_samples", "Échantillons insuffisants"),
        ("test_error_line_numbers", "Numéros de ligne"),
        ("test_file_not_found", "Fichier inexistant"),
        ("test_empty_dataset", "Dataset vide"),
        ("test_save_errors_for_pr", "Sauvegarde erreurs PR"),
        ("test_json_output", "Sortie JSON")
    ]
    
    test_instance = TestDatasetValidator()
    passed = 0
    failed = 0
    
    for test_method, description in test_cases:
        try:
            test_instance.setup_method()
            getattr(test_instance, test_method)()
            print(f"✅ {description}")
            passed += 1
        except Exception as e:
            print(f"❌ {description}: {e}")
            failed += 1
    
    print(f"\n📊 Résultats: {passed} réussis, {failed} échoués")
    
    if failed == 0:
        print("🎉 Tous les tests sont passés !")
        return True
    else:
        print("🚨 Certains tests ont échoué")
        return False


if __name__ == "__main__":
    # Si pytest n'est pas disponible, lancer les tests manuellement
    try:
        import pytest
        print("🚀 Utilisation de pytest pour les tests...")
        exit_code = pytest.main([__file__, "-v"])
        sys.exit(exit_code)
    except ImportError:
        print("⚠️ pytest non trouvé, lancement des tests manuellement...")
        success = run_manual_tests()
        sys.exit(0 if success else 1)
