#!/usr/bin/env bash

# TradePulse News Collection - Script quotidien
# =============================================
# 
# Ce script collecte automatiquement les actualités financières
# du jour et génère un dataset au format news_YYYYMMDD.csv
#
# Usage:
#   ./run_daily.sh                    # RSS par défaut
#   ./run_daily.sh newsapi            # Avec NewsAPI (nécessite clé)
#   ./run_daily.sh rss 40             # RSS avec 40 articles
#
# Installation des dépendances (une fois):
#   pip install feedparser requests

set -euo pipefail  # Arrêt sur erreur

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/collect_news.py"
DATASETS_DIR="$PROJECT_ROOT/datasets"
LOG_FILE="$DATASETS_DIR/collect.log"

# Paramètres
SOURCE="${1:-rss}"           # Source par défaut: RSS
COUNT="${2:-30}"             # Nombre d'articles par défaut: 30
DATE=$(date +%Y%m%d)         # Format YYYYMMDD
OUTPUT_FILE="$DATASETS_DIR/news_${DATE}.csv"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction d'affichage avec couleurs
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Banner
echo -e "${BLUE}"
echo "🤖 TradePulse News Collector"
echo "============================"
echo -e "${NC}"

# Vérifications préliminaires
log_info "Vérification de l'environnement..."

# Créer le dossier datasets s'il n'existe pas
mkdir -p "$DATASETS_DIR"

# Vérifier que le script Python existe
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    log_error "Script collect_news.py non trouvé: $PYTHON_SCRIPT"
    exit 1
fi

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    log_error "Python3 non trouvé. Installez Python 3.8+"
    exit 1
fi

# Vérifier les dépendances selon la source
if [[ "$SOURCE" == "rss" ]]; then
    if ! python3 -c "import feedparser" &> /dev/null; then
        log_warning "feedparser non installé"
        log_info "Installation automatique..."
        pip install feedparser || {
            log_error "Échec installation feedparser. Installez avec: pip install feedparser"
            exit 1
        }
    fi
fi

if [[ "$SOURCE" == "newsapi" ]]; then
    if ! python3 -c "import requests" &> /dev/null; then
        log_warning "requests non installé"
        log_info "Installation automatique..."
        pip install requests || {
            log_error "Échec installation requests. Installez avec: pip install requests"
            exit 1
        }
    fi
    
    if [[ -z "${NEWSAPI_KEY:-}" ]]; then
        log_warning "Variable NEWSAPI_KEY non définie"
        log_info "Définissez votre clé API: export NEWSAPI_KEY=your_key_here"
    fi
fi

# Affichage des paramètres
log_info "Configuration:"
echo "  📅 Date: $DATE"
echo "  📰 Source: $SOURCE"
echo "  📊 Articles: $COUNT"
echo "  📁 Sortie: $OUTPUT_FILE"
echo

# Vérifier si le fichier existe déjà
if [[ -f "$OUTPUT_FILE" ]]; then
    log_warning "Le fichier $OUTPUT_FILE existe déjà"
    read -p "Voulez-vous le remplacer? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Opération annulée"
        exit 0
    fi
fi

# Lancement de la collecte
log_info "🚀 Lancement de la collecte..."

# Construire la commande
CMD="python3 \"$PYTHON_SCRIPT\" --source \"$SOURCE\" --count $COUNT --output \"$OUTPUT_FILE\""

# Ajouter la clé NewsAPI si nécessaire
if [[ "$SOURCE" == "newsapi" && -n "${NEWSAPI_KEY:-}" ]]; then
    CMD="$CMD --newsapi-key \"$NEWSAPI_KEY\""
fi

# Exécuter la commande et capturer la sortie
if eval "$CMD" 2>&1 | tee -a "$LOG_FILE"; then
    log_success "Collecte terminée avec succès!"
    
    # Afficher quelques statistiques
    if [[ -f "$OUTPUT_FILE" ]]; then
        LINES=$(wc -l < "$OUTPUT_FILE")
        ARTICLES=$((LINES - 1))  # Soustraire l'en-tête
        log_success "📊 $ARTICLES articles collectés dans $OUTPUT_FILE"
        
        # Aperçu des labels
        if command -v python3 &> /dev/null; then
            echo
            log_info "Distribution des labels:"
            python3 -c "
import pandas as pd
try:
    df = pd.read_csv('$OUTPUT_FILE')
    distribution = df['label'].value_counts()
    for label, count in distribution.items():
        percent = (count / len(df)) * 100
        print(f'  {label}: {count} ({percent:.1f}%)')
except Exception as e:
    print(f'  Erreur lecture: {e}')
"
        fi
        
        echo
        log_info "🚀 Prochaines étapes recommandées:"
        echo "  1. Validation: python3 scripts/validate_dataset.py $OUTPUT_FILE"
        echo "  2. Édition: open news_editor.html (interface web)"
        echo "  3. Commit: git add $OUTPUT_FILE && git commit -m 'Daily dataset $DATE'"
        echo "  4. Push: git push (déclenche le fine-tuning automatique)"
        
    else
        log_error "Fichier de sortie non créé"
        exit 1
    fi
else
    log_error "Échec de la collecte"
    exit 1
fi

# Nettoyage des anciens logs (garder seulement les 10 derniers jours)
find "$DATASETS_DIR" -name "*.log" -mtime +10 -delete 2>/dev/null || true

log_success "🎉 Processus terminé!"