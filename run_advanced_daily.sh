#!/bin/bash
# TradePulse - Script de collecte d'actualités avancée
# Usage: ./run_advanced_daily.sh [source] [count] [days]
# 
# Exemples:
#   ./run_advanced_daily.sh mixed 60 3    # Mode mixte, 60 articles, 3 jours
#   ./run_advanced_daily.sh rss 50 5      # RSS seulement, 50 articles, 5 jours
#   ./run_advanced_daily.sh newsapi 40 2  # NewsAPI seulement (nécessite clé)

set -euo pipefail  # Arrêt en cas d'erreur, variables non définies, erreurs de pipe

# Configuration par défaut
DEFAULT_SOURCE="mixed"
DEFAULT_COUNT=50
DEFAULT_DAYS=3

# Arguments
SOURCE=${1:-$DEFAULT_SOURCE}
COUNT=${2:-$DEFAULT_COUNT}
DAYS=${3:-$DEFAULT_DAYS}

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 TradePulse - Collecte Avancée d'Actualités${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""
echo -e "📊 Configuration:"
echo -e "  Source: ${YELLOW}$SOURCE${NC}"
echo -e "  Articles: ${YELLOW}$COUNT${NC}"
echo -e "  Période: ${YELLOW}$DAYS jours${NC}"
echo ""

# Vérification des dépendances
echo -e "${BLUE}🔍 Vérification des dépendances...${NC}"

if ! python3 -c "import feedparser" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  feedparser manquant. Installation...${NC}"
    pip install feedparser
fi

if ! python3 -c "import requests" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  requests manquant. Installation...${NC}"
    pip install requests
fi

# Vérification clé NewsAPI si nécessaire
if [[ "$SOURCE" == "newsapi" || "$SOURCE" == "mixed" ]]; then
    if [[ -z "${NEWSAPI_KEY:-}" ]]; then
        echo -e "${YELLOW}⚠️  Variable NEWSAPI_KEY non définie${NC}"
        echo -e "   Pour utiliser NewsAPI, définissez: export NEWSAPI_KEY='votre_clé'"
        echo -e "   Ou utilisez: ./run_advanced_daily.sh rss $COUNT $DAYS"
        echo ""
    else
        echo -e "${GREEN}✅ Clé NewsAPI détectée${NC}"
    fi
fi

# Vérification du répertoire datasets
if [[ ! -d "datasets" ]]; then
    echo -e "${BLUE}📁 Création du répertoire datasets/${NC}"
    mkdir -p datasets
fi

echo ""
echo -e "${BLUE}🔄 Lancement de la collecte...${NC}"
echo ""

# Construction de la commande
CMD="python3 scripts/collect_news.py --source $SOURCE --count $COUNT --days $DAYS"

# Ajouter la clé NewsAPI si disponible
if [[ -n "${NEWSAPI_KEY:-}" && ("$SOURCE" == "newsapi" || "$SOURCE" == "mixed") ]]; then
    CMD="$CMD --newsapi-key $NEWSAPI_KEY"
fi

# Exécution
if eval $CMD; then
    echo ""
    echo -e "${GREEN}✅ Collecte terminée avec succès !${NC}"
    echo ""
    
    # Affichage du dernier dataset créé
    LATEST_CSV=$(ls -t datasets/news_*.csv 2>/dev/null | head -1 || echo "")
    if [[ -n "$LATEST_CSV" ]]; then
        echo -e "${BLUE}📄 Dernier dataset: ${YELLOW}$LATEST_CSV${NC}"
        
        # Statistiques rapides
        if command -v wc &> /dev/null; then
            LINES=$(wc -l < "$LATEST_CSV" 2>/dev/null || echo "0")
            echo -e "${BLUE}📊 Lignes: ${YELLOW}$((LINES - 1))${NC} (hors header)"
        fi
        
        # Métadonnées si disponibles
        JSON_FILE="${LATEST_CSV%.csv}.json"
        if [[ -f "$JSON_FILE" ]]; then
            echo -e "${BLUE}📁 Métadonnées: ${YELLOW}$JSON_FILE${NC}"
        fi
    fi
    
    echo ""
    echo -e "${BLUE}🚀 Prochaines étapes recommandées:${NC}"
    echo -e "  ${GREEN}1.${NC} Validation: python scripts/validate_dataset.py"
    echo -e "  ${GREEN}2.${NC} Édition: open news_editor.html"
    echo -e "  ${GREEN}3.${NC} Pipeline: ./scripts/auto-pipeline.sh pipeline"
    echo ""
    
else
    echo ""
    echo -e "${RED}❌ Erreur lors de la collecte${NC}"
    echo ""
    echo -e "${YELLOW}💡 Solutions possibles:${NC}"
    echo -e "  • Vérifier la connexion internet"
    echo -e "  • Installer les dépendances: pip install feedparser requests"
    echo -e "  • Tester avec placeholder: python scripts/collect_news.py --source placeholder"
    
    if [[ "$SOURCE" == "newsapi" || "$SOURCE" == "mixed" ]]; then
        echo -e "  • Vérifier la clé NewsAPI: export NEWSAPI_KEY='votre_clé'"
        echo -e "  • Utiliser RSS seulement: ./run_advanced_daily.sh rss $COUNT $DAYS"
    fi
    
    echo ""
    exit 1
fi

# Affichage des articles en cache (plus robuste sans jq)
if [[ -f "datasets/.article_cache.json" ]]; then
    # Utiliser Python au lieu de jq pour éviter les broken pipes
    CACHE_SIZE=$(python3 -c "
import json
import sys
try:
    with open('datasets/.article_cache.json', 'r') as f:
        data = json.load(f)
        print(len(data.get('articles', [])))
except Exception:
    print('?')
" 2>/dev/null || echo "?")
    echo -e "${BLUE}🗄️  Cache de déduplication: ${YELLOW}$CACHE_SIZE${NC} articles connus"
fi

echo ""
echo -e "${GREEN}🎉 Collecte d'actualités terminée !${NC}"
