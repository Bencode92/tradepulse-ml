#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradePulse - Pipeline Unifié d'Apprentissage Incrémental
========================================================

🎯 RÉSOUT LE PROBLÈME PRINCIPAL : Un seul modèle qui s'améliore progressivement

Au lieu de créer de nouveaux modèles à chaque fois :
✅ Collecte automatique d'actualités + labelling ML
✅ Apprentissage incrémental (améliore le modèle existant)
✅ Validation automatique avant mise à jour
✅ Un seul modèle stable sur HuggingFace pour votre site

Usage:
------
python unified_pipeline.py                          # Mode interactif
python unified_pipeline.py --mode quick             # Mode rapide (30 articles, mode test)
python unified_pipeline.py --mode production        # Mode production (met à jour le modèle stable)
python unified_pipeline.py --collect-only           # Collecte seulement
python unified_pipeline.py --train-only             # Apprentissage incrémental seulement
python unified_pipeline.py --status                 # État des modèles
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configuration des couleurs pour l'affichage
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color

# Configuration
PRODUCTION_MODEL = "Bencode92/tradepulse-finbert-prod"
DEVELOPMENT_MODEL = "Bencode92/tradepulse-finbert-dev"
FALLBACK_MODEL = "yiyanghkust/finbert-tone"

DEFAULT_COUNT = 30
DEFAULT_DAYS = 2
DEFAULT_CONFIDENCE = 0.75

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("unified_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("unified-pipeline")

def log_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.NC}")

def log_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.NC}")

def log_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.NC}")

def log_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.NC}")

def log_header(msg: str):
    print(f"\n{Colors.PURPLE}🔶 {msg}{Colors.NC}")
    print(f"{Colors.PURPLE}{'=' * 60}{Colors.NC}")

def show_banner():
    print(f"{Colors.PURPLE}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              🚀 TradePulse ML - Pipeline Unifié             ║")
    print("║         Apprentissage Incrémental Automatisé                ║")
    print("║                                                              ║")
    print("║  🎯 UN SEUL modèle qui s'améliore (pas de nouveaux modèles) ║")
    print("║  🤖 Collecte + Labelling + Apprentissage automatique        ║")
    print("║  ✅ Validation avant mise à jour production                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.NC}\n")

def check_prerequisites() -> bool:
    """Vérification des prérequis"""
    log_header("Vérification des prérequis")
    
    # Vérifier Python
    if sys.version_info < (3, 8):
        log_error("Python 3.8+ requis")
        return False
    
    # Vérifier les scripts
    required_scripts = [
        "scripts/collect_news.py",
        "scripts/finetune.py"
    ]
    
    for script in required_scripts:
        if not Path(script).exists():
            log_error(f"Script manquant: {script}")
            return False
    
    # Vérifier les dépendances Python (basique)
    try:
        import transformers
        import torch
        import pandas
        log_success("Dépendances Python disponibles")
    except ImportError as e:
        log_error(f"Dépendance manquante: {e}")
        log_info("Installez avec: pip install transformers torch pandas")
        return False
    
    log_success("Prérequis validés")
    return True

def run_command(cmd: List[str], description: str) -> bool:
    """Exécute une commande et gère les erreurs"""
    log_info(f"{description}...")
    logger.info(f"Exécution: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            logger.info(f"Sortie: {result.stdout}")
        
        log_success(f"{description} terminé")
        return True
        
    except subprocess.CalledProcessError as e:
        log_error(f"{description} échoué")
        logger.error(f"Erreur: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False

def find_latest_dataset() -> Optional[Path]:
    """Trouve le dernier dataset"""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        return None
    
    csv_files = list(datasets_dir.glob("news_*.csv"))
    if not csv_files:
        return None
    
    # Trier par nom (assume format news_YYYYMMDD.csv)
    csv_files.sort(reverse=True)
    return csv_files[0]

def run_collect_and_label(count: int = 30, days: int = 2, mode: str = "test", 
                         use_newsapi: bool = False, confidence: float = 0.75) -> Optional[Path]:
    """Collecte automatique + labelling ML"""
    log_header("Collecte Automatique + Labelling ML")
    
    # Déterminer le modèle selon le mode
    if mode == "production":
        ml_model = "production"
    elif mode == "development":
        ml_model = "development"
    else:
        ml_model = "fallback"
    
    # Construire la commande
    cmd = [
        "python", "scripts/collect_news.py",
        "--source", "mixed",
        "--count", str(count),
        "--days", str(days),
        "--auto-label",
        "--ml-model", ml_model,
        "--confidence-threshold", str(confidence)
    ]
    
    # Ajouter la clé NewsAPI si disponible
    if use_newsapi and os.getenv("NEWSAPI_KEY"):
        cmd.extend(["--newsapi-key", os.getenv("NEWSAPI_KEY")])
    
    # Exécuter la collecte
    if run_command(cmd, "Collecte automatique + labelling ML"):
        # Trouver le fichier créé
        latest_dataset = find_latest_dataset()
        if latest_dataset:
            log_success(f"Dataset créé: {latest_dataset}")
            return latest_dataset
        else:
            log_error("Dataset non trouvé après collecte")
            return None
    else:
        return None

def run_incremental_training(dataset_path: Path, mode: str = "test", 
                           force_update: bool = False) -> bool:
    """Apprentissage incrémental"""
    log_header("Apprentissage Incrémental")
    
    # Générer le répertoire de sortie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/incremental_{mode}_{timestamp}"
    
    # Construire la commande
    cmd = [
        "python", "scripts/finetune.py",
        "--incremental",
        "--mode", mode,
        "--dataset", str(dataset_path),
        "--output_dir", output_dir,
    ]
    
    if force_update:
        cmd.append("--force-update")
    
    # Exécuter l'entraînement
    if run_command(cmd, f"Apprentissage incrémental (mode: {mode})"):
        # Vérifier les résultats
        report_path = Path(output_dir) / "incremental_training_report.json"
        if report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)
                
                baseline_acc = report.get("baseline_metrics", {}).get("accuracy", 0)
                new_acc = report.get("new_metrics", {}).get("accuracy", 0)
                model_updated = report.get("model_updated", False)
                
                log_info(f"Baseline accuracy: {baseline_acc:.3f}")
                log_info(f"Nouvelle accuracy: {new_acc:.3f}")
                
                if model_updated:
                    log_success("🚀 Modèle mis à jour sur HuggingFace!")
                    if report.get("hf_model_id"):
                        log_info(f"Modèle: https://huggingface.co/{report['hf_model_id']}")
                else:
                    log_warning("Modèle non mis à jour (amélioration insuffisante)")
                
                return True
                
            except Exception as e:
                log_error(f"Erreur lecture rapport: {e}")
                return False
        else:
            log_warning("Rapport d'entraînement non trouvé")
            return False
    else:
        return False

def show_status():
    """Affichage du statut des modèles"""
    log_header("État des Modèles TradePulse")
    
    print(f"{Colors.WHITE}Modèles configurés:{Colors.NC}")
    print(f"   🏭 Production: {Colors.GREEN}{PRODUCTION_MODEL}{Colors.NC}")
    print(f"   🔬 Development: {Colors.YELLOW}{DEVELOPMENT_MODEL}{Colors.NC}")
    print(f"   🔄 Fallback: {Colors.BLUE}{FALLBACK_MODEL}{Colors.NC}")
    
    print(f"\n{Colors.WHITE}Datasets locaux:{Colors.NC}")
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        csv_files = list(datasets_dir.glob("*.csv"))
        print(f"   📁 Fichiers CSV: {Colors.GREEN}{len(csv_files)}{Colors.NC}")
        
        if csv_files:
            print(f"   📊 Derniers datasets:")
            for csv_file in sorted(csv_files, reverse=True)[:5]:
                size = csv_file.stat().st_size
                size_str = f"{size//1024}KB" if size > 1024 else f"{size}B"
                print(f"      {Colors.CYAN}{csv_file.name}{Colors.NC} ({size_str})")
    else:
        print(f"   {Colors.RED}Dossier datasets/ introuvable{Colors.NC}")
    
    print(f"\n{Colors.WHITE}Modèles locaux:{Colors.NC}")
    models_dir = Path("models")
    if models_dir.exists():
        incremental_dirs = list(models_dir.glob("incremental_*"))
        print(f"   🤖 Modèles entraînés: {Colors.GREEN}{len(incremental_dirs)}{Colors.NC}")
        
        if incremental_dirs:
            print(f"   🕐 Derniers modèles:")
            for model_dir in sorted(incremental_dirs, reverse=True)[:3]:
                print(f"      {Colors.CYAN}{model_dir.name}{Colors.NC}")
    else:
        print(f"   {Colors.RED}Dossier models/ introuvable{Colors.NC}")
    
    # État du cache
    cache_file = Path("datasets/.article_cache.json")
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cache_data = json.load(f)
            cache_size = len(cache_data.get("articles", []))
            print(f"\n{Colors.WHITE}Cache de déduplication:{Colors.NC}")
            print(f"   🗄️  Articles connus: {Colors.GREEN}{cache_size}{Colors.NC}")
        except:
            pass

def interactive_config() -> Dict:
    """Configuration interactive"""
    log_header("Configuration Interactive")
    
    print(f"{Colors.WHITE}Configurons votre pipeline d'apprentissage incrémental:{Colors.NC}\n")
    
    # Mode d'exécution
    print("🎯 Mode d'exécution:")
    print("   1) Test (validation locale, pas de mise à jour)")
    print("   2) Development (met à jour le modèle de développement)")
    print("   3) Production (met à jour le modèle stable)")
    mode_choice = input("Choisissez (1-3) [1]: ").strip() or "1"
    
    mode_map = {"1": "test", "2": "development", "3": "production"}
    mode = mode_map.get(mode_choice, "test")
    
    # Paramètres de collecte
    count = input(f"📊 Nombre d'articles à collecter [{DEFAULT_COUNT}]: ").strip()
    count = int(count) if count.isdigit() else DEFAULT_COUNT
    
    days = input(f"📅 Période de collecte en jours [{DEFAULT_DAYS}]: ").strip()
    days = int(days) if days.isdigit() else DEFAULT_DAYS
    
    confidence = input(f"🎯 Seuil de confiance ML (0.5-0.95) [{DEFAULT_CONFIDENCE}]: ").strip()
    confidence = float(confidence) if confidence else DEFAULT_CONFIDENCE
    
    # Options avancées
    print(f"\n🔧 Options avancées:")
    use_newsapi = input("   Utiliser NewsAPI? (y/N): ").strip().lower().startswith('y')
    force_update = input("   Forcer la mise à jour même sans amélioration? (y/N): ").strip().lower().startswith('y')
    
    config = {
        "mode": mode,
        "count": count,
        "days": days,
        "confidence": confidence,
        "use_newsapi": use_newsapi,
        "force_update": force_update
    }
    
    print(f"\n{Colors.GREEN}Configuration terminée:{Colors.NC}")
    print(f"   Mode: {Colors.YELLOW}{mode}{Colors.NC}")
    print(f"   Articles: {Colors.YELLOW}{count}{Colors.NC}")
    print(f"   Période: {Colors.YELLOW}{days} jours{Colors.NC}")
    print(f"   Confiance: {Colors.YELLOW}{confidence}{Colors.NC}")
    
    confirm = input("Continuer? (Y/n): ").strip().lower()
    if confirm.startswith('n'):
        log_info("Pipeline annulé")
        sys.exit(0)
    
    return config

def run_full_pipeline(config: Dict) -> bool:
    """Pipeline complet"""
    log_header("Pipeline Complet - Apprentissage Incrémental")
    
    print(f"{Colors.WHITE}🚀 Démarrage du pipeline unifié TradePulse{Colors.NC}")
    print(f"   Mode: {Colors.YELLOW}{config['mode']}{Colors.NC}")
    print(f"   Articles: {Colors.YELLOW}{config['count']}{Colors.NC} ({Colors.YELLOW}{config['days']}{Colors.NC} jours)")
    
    target_model = PRODUCTION_MODEL if config['mode'] == 'production' else DEVELOPMENT_MODEL
    print(f"   Modèle cible: {Colors.GREEN}{target_model}{Colors.NC}")
    
    # Étape 1: Collecte automatique
    dataset_path = run_collect_and_label(
        count=config['count'],
        days=config['days'], 
        mode=config['mode'],
        use_newsapi=config['use_newsapi'],
        confidence=config['confidence']
    )
    
    if not dataset_path:
        log_error("Pipeline interrompu lors de la collecte")
        return False
    
    # Étape 2: Apprentissage incrémental
    if not run_incremental_training(
        dataset_path=dataset_path,
        mode=config['mode'],
        force_update=config['force_update']
    ):
        log_error("Pipeline interrompu lors de l'apprentissage")
        return False
    
    log_success("Pipeline complet terminé avec succès!")
    
    # Résumé final
    print(f"\n{Colors.WHITE}📋 Résumé du pipeline:{Colors.NC}")
    print(f"   ✅ Collecte automatique terminée")
    print(f"   ✅ Labelling ML automatique terminé")
    print(f"   ✅ Apprentissage incrémental terminé")
    
    if config['mode'] == 'production':
        print(f"   🚀 {Colors.GREEN}Modèle de production potentiellement mis à jour{Colors.NC}")
        print(f"   💡 Vérifiez les métriques pour confirmer l'amélioration")
    else:
        print(f"   🔬 Modèle testé en mode {config['mode']}")
    
    print(f"\n{Colors.WHITE}🔄 Prochaines étapes suggérées:{Colors.NC}")
    print(f"   1. Vérifier les performances du modèle")
    print(f"   2. Tester le modèle sur votre site TradePulse")
    print(f"   3. Surveiller les performances en production")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="TradePulse - Pipeline Unifié d'Apprentissage Incrémental")
    
    # Modes d'exécution
    parser.add_argument("--mode", choices=["quick", "production", "interactive"], 
                       help="Mode d'exécution")
    parser.add_argument("--collect-only", action="store_true", 
                       help="Collecte seulement")
    parser.add_argument("--train-only", action="store_true", 
                       help="Apprentissage incrémental seulement (sur dernier dataset)")
    parser.add_argument("--status", action="store_true", 
                       help="Afficher l'état des modèles")
    
    # Paramètres de collecte
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                       help=f"Nombre d'articles (défaut: {DEFAULT_COUNT})")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                       help=f"Période en jours (défaut: {DEFAULT_DAYS})")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                       help=f"Seuil de confiance ML (défaut: {DEFAULT_CONFIDENCE})")
    
    # Options avancées
    parser.add_argument("--use-newsapi", action="store_true",
                       help="Utiliser NewsAPI (nécessite NEWSAPI_KEY)")
    parser.add_argument("--force-update", action="store_true",
                       help="Forcer la mise à jour même sans amélioration")
    parser.add_argument("--training-mode", choices=["test", "development", "production"], 
                       default="test", help="Mode d'entraînement")
    
    args = parser.parse_args()
    
    # Banner
    show_banner()
    
    # Vérification des prérequis
    if not check_prerequisites():
        sys.exit(1)
    
    # Traitement selon les arguments
    if args.status:
        show_status()
    
    elif args.collect_only:
        dataset_path = run_collect_and_label(
            count=args.count,
            days=args.days,
            mode=args.training_mode,
            use_newsapi=args.use_newsapi,
            confidence=args.confidence
        )
        if dataset_path:
            log_success(f"Collecte terminée: {dataset_path}")
        else:
            log_error("Échec de la collecte")
            sys.exit(1)
    
    elif args.train_only:
        dataset_path = find_latest_dataset()
        if not dataset_path:
            log_error("Aucun dataset trouvé pour l'entraînement")
            log_info("Lancez d'abord: python unified_pipeline.py --collect-only")
            sys.exit(1)
        
        if run_incremental_training(
            dataset_path=dataset_path,
            mode=args.training_mode,
            force_update=args.force_update
        ):
            log_success("Apprentissage incrémental terminé")
        else:
            log_error("Échec de l'apprentissage")
            sys.exit(1)
    
    elif args.mode == "quick":
        # Mode rapide
        config = {
            "mode": "test",
            "count": 20,
            "days": 1,
            "confidence": 0.75,
            "use_newsapi": False,
            "force_update": False
        }
        log_info("Mode rapide activé")
        if not run_full_pipeline(config):
            sys.exit(1)
    
    elif args.mode == "production":
        # Mode production avec confirmation
        log_warning("Mode PRODUCTION - mettra à jour le modèle stable!")
        confirm = input("Confirmer? (y/N): ").strip().lower()
        if not confirm.startswith('y'):
            log_info("Annulé")
            sys.exit(0)
        
        config = {
            "mode": "production",
            "count": 50,
            "days": 3,
            "confidence": 0.80,
            "use_newsapi": bool(os.getenv("NEWSAPI_KEY")),
            "force_update": args.force_update
        }
        if not run_full_pipeline(config):
            sys.exit(1)
    
    elif args.mode == "interactive" or not args.mode:
        # Mode interactif
        config = interactive_config()
        if not run_full_pipeline(config):
            sys.exit(1)
    
    else:
        # Configuration personnalisée via arguments
        config = {
            "mode": args.training_mode,
            "count": args.count,
            "days": args.days,
            "confidence": args.confidence,
            "use_newsapi": args.use_newsapi,
            "force_update": args.force_update
        }
        if not run_full_pipeline(config):
            sys.exit(1)

if __name__ == "__main__":
    main()