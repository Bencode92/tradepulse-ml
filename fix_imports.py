#!/usr/bin/env python3
"""Correction des imports pour utiliser les modules locaux"""

import re

# Lire collect_news.py
with open("scripts/collect_news.py", "r", encoding="utf-8") as f:
    content = f.read()

# Remplacer l'import externe par l'import local
old_import = """# Import CommodityCorrelator
try:
    # Essayer d'importer depuis stock-analysis-platform
    platform_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 "stock-analysis-platform", "scripts")
    if os.path.exists(platform_path):
        sys.path.append(platform_path)
        from commodity_correlator import CommodityCorrelator
        CORRELATOR_AVAILABLE = True
        logger.info("✅ CommodityCorrelator importé avec succès")
    else:
        CORRELATOR_AVAILABLE = False
        logger.warning("⚠️ CommodityCorrelator non disponible - chemin non trouvé")
except ImportError as e:
    CORRELATOR_AVAILABLE = False
    logger.warning(f"⚠️ CommodityCorrelator non disponible: {e}")"""

new_import = """# Import CommodityCorrelator
try:
    from commodity_correlator import CommodityCorrelator
    CORRELATOR_AVAILABLE = True
    logger.info("✅ CommodityCorrelator importé (local)")
except ImportError as e:
    CORRELATOR_AVAILABLE = False
    logger.warning(f"⚠️ CommodityCorrelator non disponible: {e}")"""

# Remplacer
content = content.replace(old_import, new_import)

# Sauvegarder
with open("scripts/collect_news.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ collect_news.py mis à jour pour utiliser le correlator local")
