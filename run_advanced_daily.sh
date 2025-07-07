#!/usr/bin/env bash

# TradePulse Advanced News Collection
# ==================================
# 
# Script optimisé pour la version avancée avec déduplication
# et collecte multi-sources
#
# Usage:
#   ./run_advanced_daily.sh                    # Mixed sources, 3 jours, 40 articles
#   ./run_advanced_daily.sh rss 60 5          # RSS, 60 articles, 5 jours  
#   ./run_advanced_daily.sh mixed 80 2        # Mixed, 80 articles, 2 jours
#   ./run_advanced_daily.sh newsapi 50 1      # NewsAPI only, 50 articles, 1 jour

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/collect_news.py"
DATASETS_DIR="$PROJECT_ROOT/datasets"

# Paramètres avec valeurs optimisées
SOURCE="${1:-mixed}"        # mixed par défaut pour diversité maximale
COUNT="${2:-40}"            # Plus d'articles par défaut
DAYS="${3:-3}"              # Fenêtre de 3 jours pour plus de contenu
DATE=$(date +%Y%m%d)
OUTPUT_FILE="$DATASETS_DIR/news_${DATE}.csv"
CACHE_FILE="$DATASETS_DIR/.article_cache.json"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_highlight() { echo -e "${PURPLE}🚀 $1${NC}"; }

# Banner amélioré
echo -e "${PURPLE}"
echo "🤖 TradePulse Advanced News Collector"
echo "====================================="
echo "Multi-sources • Déduplication • Fenêtre temporelle"
echo -e "${NC}"

# Vérifications
log_info "Vérification de l'environnement..."
mkdir -p "$DATASETS_DIR"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    log_error "Script collect_news.py non trouvé: $PYTHON_SCRIPT"
    exit 1
fi

# Installation automatique des dépendances selon la source
log_info "Vérification des dépendances pour source: $SOURCE"

if [[ "$SOURCE" == "rss" || "$SOURCE" == "mixed" ]]; then
    if ! python3 -c "import feedparser" &> /dev/null; then
        log_warning "feedparser manquant - installation..."
        pip install feedparser || {
            log_error "Échec installation feedparser"
            exit 1
        }
        log_success "feedparser installé"
    fi
fi

if [[ "$SOURCE" == "newsapi" || "$SOURCE" == "mixed" ]]; then
    if ! python3 -c "import requests" &> /dev/null; then
        log_warning "requests manquant - installation..."
        pip install requests || {
            log_error "Échec installation requests"
            exit 1
        }
        log_success "requests installé"
    fi
    
    if [[ "$SOURCE" == "newsapi" && -z "${NEWSAPI_KEY:-}" ]]; then
        log_warning "NEWSAPI_KEY non définie pour source NewsAPI"
        log_info "Définissez: export NEWSAPI_KEY=your_key_here"
        log_info "Ou utilisez: ./run_advanced_daily.sh mixed (RSS + NewsAPI optionnel)"
    fi
fi

# Affichage configuration avancée
log_highlight "Configuration de collecte avancée:"
echo "  📅 Date: $DATE"
echo "  📰 Source: $SOURCE"
echo "  📊 Articles cible: $COUNT"
echo "  📆 Fenêtre temporelle: $DAYS jours"
echo "  📁 Sortie: $OUTPUT_FILE"
echo "  🗄️  Cache déduplication: $([ -f "$CACHE_FILE" ] && echo "Activé ($(jq '.articles | length' "$CACHE_FILE" 2>/dev/null || echo "?") articles)" || echo "Nouveau")"
echo

# Gestion fichier existant
if [[ -f "$OUTPUT_FILE" ]]; then
    log_warning "Fichier existant: $OUTPUT_FILE"
    
    # Afficher un aperçu du fichier existant
    if command -v python3 &> /dev/null; then
        echo "  📊 Aperçu actuel:"
        python3 -c "
import pandas as pd
try:
    df = pd.read_csv('$OUTPUT_FILE')
    print(f'    Articles: {len(df)}')
    dist = df['label'].value_counts()
    for label, count in dist.items():
        print(f'    {label}: {count}')
except: pass
" 2>/dev/null
    fi
    
    read -p "Remplacer? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Opération annulée"
        exit 0
    fi
fi

# Construction commande avancée
log_highlight "🚀 Lancement collecte avancée..."

CMD="python3 \"$PYTHON_SCRIPT\" --source \"$SOURCE\" --count $COUNT --days $DAYS --output \"$OUTPUT_FILE\""

# Options avancées
if [[ "$SOURCE" == "newsapi" || "$SOURCE" == "mixed" ]] && [[ -n "${NEWSAPI_KEY:-}" ]]; then
    CMD="$CMD --newsapi-key \"$NEWSAPI_KEY\""
fi

# Exécution avec monitoring
echo "🔄 Commande: $CMD"
echo

if eval "$CMD" 2>&1 | tee -a "$DATASETS_DIR/collect_advanced.log"; then
    log_success "Collecte terminée!"
    
    # Statistiques détaillées
    if [[ -f "$OUTPUT_FILE" ]]; then
        log_highlight "📈 Analyse du dataset généré:"
        
        # Stats basiques
        LINES=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo "0")
        ARTICLES=$((LINES - 1))
        echo "  📊 Articles collectés: $ARTICLES"
        
        # Distribution et métadonnées
        if command -v python3 &> /dev/null; then
            python3 -c "
import pandas as pd
import json
from pathlib import Path

try:
    # Dataset principal
    df = pd.read_csv('$OUTPUT_FILE')
    print('  📈 Distribution des labels:')
    dist = df['label'].value_counts()
    for label, count in dist.items():
        percent = (count / len(df)) * 100
        print(f'    {label}: {count} ({percent:.1f}%)')
    
    # Longueur moyenne des textes
    avg_length = df['text'].str.len().mean()
    print(f'  📝 Longueur moyenne: {avg_length:.0f} caractères')
    
    # Métadonnées si disponibles
    metadata_file = Path('$OUTPUT_FILE').with_suffix('.json')
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            meta = json.load(f)
        print(f'  🗄️  Cache: {meta.get(\"cache_size\", 0)} articles connus')
        print(f'  ⏰ Créé: {meta.get(\"created_at\", \"Unknown\")[:16]}')
    
except Exception as e:
    print(f'  ⚠️  Erreur analyse: {e}')
" 2>/dev/null
        fi
        
        echo
        log_highlight "🎯 Vérifications qualité:"
        
        # Validation automatique si disponible
        if [[ -f "$PROJECT_ROOT/scripts/validate_dataset.py" ]]; then
            echo "  🔍 Validation automatique..."
            if python3 "$PROJECT_ROOT/scripts/validate_dataset.py" "$OUTPUT_FILE" --quiet 2>/dev/null; then
                echo "  ✅ Dataset valide"
            else
                echo "  ⚠️  Validation a détecté des problèmes (voir logs complets)"
            fi
        fi
        
        echo
        log_highlight "🚀 Prochaines étapes recommandées:"
        echo "  1. 📝 Édition: open news_editor.html (interface web)"
        echo "  2. 🔍 Validation: python3 scripts/validate_dataset.py $OUTPUT_FILE"
        echo "  3. 📤 Commit: git add $OUTPUT_FILE && git commit -m 'Add diversified dataset $DATE'"
        echo "  4. 🚀 Push: git push (déclenche fine-tuning automatique)"
        echo "  5. 🔄 Pipeline: ./scripts/auto-pipeline.sh pipeline"
        
        # Suggestions d'optimisation
        echo
        log_info "💡 Suggestions pour la prochaine collecte:"
        if [[ $ARTICLES -lt $((COUNT * 8 / 10)) ]]; then
            echo "  • Augmenter la fenêtre temporelle: --days $((DAYS + 1))"
            echo "  • Utiliser source 'mixed' pour plus de diversité"
        fi
        if [[ "$SOURCE" != "mixed" ]]; then
            echo "  • Essayer source 'mixed' pour combiner RSS + NewsAPI"
        fi
        echo "  • Vérifier le cache de déduplication: $CACHE_FILE"
        
    else
        log_error "Fichier de sortie non créé"
        exit 1
    fi
else
    log_error "Échec de la collecte avancée"
    exit 1
fi

# Nettoyage et maintenance
log_info "🧹 Nettoyage maintenance..."

# Nettoyer anciens logs
find "$DATASETS_DIR" -name "*.log" -mtime +7 -delete 2>/dev/null || true

# Nettoyer anciens datasets si trop nombreux
DATASET_COUNT=$(find "$DATASETS_DIR" -name "news_*.csv" | wc -l)
if [[ $DATASET_COUNT -gt 20 ]]; then
    log_warning "Plus de 20 datasets détectés"
    echo "  Considérez archiver les anciens: find datasets/ -name 'news_*.csv' -mtime +30"
fi

log_success "🎉 Collecte avancée terminée avec succès!"

# Afficher la commande pour répéter la même collecte
echo
log_info "🔁 Pour répéter cette collecte:"
echo "  ./run_advanced_daily.sh $SOURCE $COUNT $DAYS"