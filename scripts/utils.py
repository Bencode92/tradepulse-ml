# scripts/utils.py
"""
Utilitaires pour TradePulse ML - Auto-sélection des datasets
"""
from pathlib import Path
import re
from typing import Optional, List

CSV_PATTERN = re.compile(r"news_(\d{8})\.csv$")


def latest_dataset(dir_path: str = "datasets") -> Optional[Path]:
    """
    Retourne le news_YYYYMMDD.csv le plus récent, sinon None.

    Args:
        dir_path: Répertoire à scanner (défaut: "datasets")

    Returns:
        Path du fichier le plus récent ou None si aucun trouvé
    """
    try:
        datasets_dir = Path(dir_path)
        if not datasets_dir.exists():
            print(f"⚠️ Répertoire {dir_path} introuvable")
            return None

        # Chercher tous les fichiers news_*.csv et les trier par nom (date décroissante)
        csvs = sorted(datasets_dir.glob("news_*.csv"), reverse=True)

        if not csvs:
            print(f"⚠️ Aucun fichier news_*.csv trouvé dans {dir_path}")
            return None

        latest = csvs[0]
        return latest
    except Exception as e:
        print(f"❌ Erreur lors de la recherche du dataset : {e}")
        return None


def get_date_from_filename(filename: str) -> Optional[str]:
    """
    Extrait 'YYYYMMDD' du nom de fichier news_YYYYMMDD.csv

    Args:
        filename: Nom du fichier (avec ou sans chemin)

    Returns:
        Date YYYYMMDD ou None si format invalide
    """
    filename = Path(filename).name  # Prendre seulement le nom, pas le chemin
    match = CSV_PATTERN.search(filename)
    return match.group(1) if match else None


def validate_filename_format(filename: str) -> bool:
    """Valide que le nom de fichier suit le format news_YYYYMMDD.csv"""
    filename = Path(filename).name
    return bool(CSV_PATTERN.match(filename))


def list_available_datasets(dir_path: str = "datasets") -> List[Path]:
    """Liste tous les datasets disponibles triés par date décroissante"""
    try:
        datasets_dir = Path(dir_path)
        if not datasets_dir.exists():
            return []

        csvs = sorted(datasets_dir.glob("news_*.csv"), reverse=True)
        return csvs
    except Exception:
        return []


if __name__ == "__main__":
    # Test des fonctions
    print("🔧 Test des utilitaires TradePulse ML")
    print("-" * 40)

    # Test latest_dataset
    latest = latest_dataset()
    if latest:
        print(f"📄 Dernier dataset : {latest}")

        # Test extraction de date
        date = get_date_from_filename(latest.name)
        print(f"📅 Date extraite : {date}")

        # Test validation format
        is_valid = validate_filename_format(latest.name)
        print(f"✅ Format valide : {is_valid}")
    else:
        print("❌ Aucun dataset trouvé")

    # Liste tous les datasets
    all_datasets = list_available_datasets()
    print(f"\n📚 Datasets disponibles ({len(all_datasets)}) :")
    for i, ds in enumerate(all_datasets[:5]):  # Afficher les 5 plus récents
        date = get_date_from_filename(ds.name)
        print(f"  {i+1}. {ds.name} (date: {date})")

    if len(all_datasets) > 5:
        print(f"  ... et {len(all_datasets) - 5} autres")
