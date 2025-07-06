#!/usr/bin/env bash

# ===============================================
# TradePulse ML - Pipeline automatique express
# ===============================================

set -euo pipefail

# Couleurs pour l'affichage
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
log_step() { echo -e "${PURPLE}🔄 $1${NC}"; }

# Fonction pour trouver le dernier CSV
get_latest_csv() {
    local CSV=$(ls datasets/news_*.csv 2>/dev/null | sort -r | head -n1)
    
    if [[ -z "$CSV" ]]; then
        log_error "Aucun fichier news_*.csv trouvé dans datasets/"
        exit 1
    fi
    
    echo "$CSV"
}

# Fonction principale - validation + fine-tuning
run_pipeline() {
    local CSV=$(get_latest_csv)
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local OUTPUT_DIR="models/finbert-${TIMESTAMP}"
    
    log_step "PIPELINE TRADEPULSE ML"
    echo "=================================================="
    log_info "Dataset détecté : $CSV"
    log_info "Modèle de sortie : $OUTPUT_DIR"
    echo ""
    
    # Étape 1 : Validation
    log_step "Validation du dataset"
    if python scripts/validate_dataset.py "$CSV"; then
        log_success "Dataset validé"
    else
        log_error "Validation échouée"
        exit 1
    fi
    
    echo ""
    
    # Étape 2 : Fine-tuning
    log_step "Fine-tuning du modèle"
    if python scripts/finetune.py \
        --dataset "$CSV" \
        --output_dir "$OUTPUT_DIR" \
        --epochs 3 \
        --lr 2e-5; then
        log_success "Fine-tuning terminé"
        log_info "Modèle sauvé dans : $OUTPUT_DIR"
    else
        log_error "Fine-tuning échoué"
        exit 1
    fi
    
    echo ""
    log_success "Pipeline terminé avec succès ! 🎉"
    echo ""
    log_info "Prochaines étapes :"
    echo "  - Tester le modèle : $0 test $OUTPUT_DIR"
    echo "  - Déployer sur HF : $0 deploy $OUTPUT_DIR"
}

# Fonction de validation seule
run_validation() {
    local DATASET="${1:-}"
    
    if [[ -z "$DATASET" ]]; then
        DATASET=$(get_latest_csv)
        log_info "Auto-sélection : $DATASET"
    fi
    
    log_step "Validation du dataset : $DATASET"
    python scripts/validate_dataset.py "$DATASET"
}

# Fonction de fine-tuning seul
run_training() {
    local DATASET="${1:-}"
    local OUTPUT_DIR="${2:-}"
    
    if [[ -z "$DATASET" ]]; then
        DATASET=$(get_latest_csv)
        log_info "Auto-sélection dataset : $DATASET"
    fi
    
    if [[ -z "$OUTPUT_DIR" ]]; then
        local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_DIR="models/finbert-${TIMESTAMP}"
        log_info "Auto-génération output : $OUTPUT_DIR"
    fi
    
    log_step "Fine-tuning : $DATASET → $OUTPUT_DIR"
    python scripts/finetune.py \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --epochs 3 \
        --lr 2e-5
}

# Fonction de test d'un modèle
test_model() {
    local MODEL_DIR="${1:-}"
    
    if [[ -z "$MODEL_DIR" ]]; then
        log_error "Usage: $0 test <model_directory>"
        exit 1
    fi
    
    if [[ ! -d "$MODEL_DIR" ]]; then
        log_error "Répertoire modèle introuvable : $MODEL_DIR"
        exit 1
    fi
    
    log_step "Test du modèle : $MODEL_DIR"
    
    python -c "
import sys
sys.path.append('.')
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

try:
    classifier = pipeline('text-classification', model='$MODEL_DIR')
    
    test_texts = [
        'Apple reported strong quarterly earnings beating expectations',
        'Market volatility increased amid economic uncertainty', 
        'Oil prices remained stable following OPEC meeting'
    ]
    
    print('🧪 Test du modèle :')
    print('=' * 40)
    
    for text in test_texts:
        result = classifier(text)
        label = result[0]['label']
        score = result[0]['score']
        print(f'📰 {text[:50]}...')
        print(f'   → {label} ({score:.3f})')
        print()
    
    print('✅ Test terminé avec succès')
except Exception as e:
    print(f'❌ Erreur lors du test : {e}')
    sys.exit(1)
"
}

# Afficher le dernier dataset
show_latest() {
    local CSV=$(get_latest_csv)
    log_info "Dernier dataset : $CSV"
    
    # Afficher quelques stats si possible
    if command -v wc &> /dev/null; then
        local LINES=$(wc -l < "$CSV")
        log_info "Nombre de lignes : $((LINES - 1))"  # -1 pour header
    fi
    
    if [[ -f "$CSV" ]]; then
        echo ""
        log_info "Aperçu du dataset :"
        head -3 "$CSV"
    fi
}

# Lister tous les datasets
list_datasets() {
    log_info "Datasets disponibles :"
    echo ""
    
    local DATASETS=($(ls datasets/news_*.csv 2>/dev/null | sort -r))
    
    if [[ ${#DATASETS[@]} -eq 0 ]]; then
        log_warning "Aucun dataset trouvé"
        return
    fi
    
    for i in "${!DATASETS[@]}"; do
        local DS="${DATASETS[$i]}"
        local BASENAME=$(basename "$DS")
        local SIZE=""
        
        if command -v du &> /dev/null; then
            SIZE=" ($(du -h "$DS" | cut -f1))"
        fi
        
        if [[ $i -eq 0 ]]; then
            echo -e "${GREEN}  📄 $BASENAME${SIZE} ${YELLOW}← dernier${NC}"
        else
            echo "  📄 $BASENAME$SIZE"
        fi
    done
}

# Menu d'aide
show_help() {
    echo "🚀 TradePulse ML - Pipeline automatique"
    echo ""
    echo "Usage: $0 <commande> [options]"
    echo ""
    echo "Commandes disponibles :"
    echo "  latest              Affiche le dernier dataset"
    echo "  list               Liste tous les datasets"
    echo "  validate [file]    Valide un dataset (dernier si omis)"
    echo "  train [file] [out] Entraîne sur un dataset (dernier si omis)"  
    echo "  pipeline           Pipeline complet (validation + training)"
    echo "  test <model_dir>   Test rapide d'un modèle"
    echo ""
    echo "Exemples :"
    echo "  $0 latest"
    echo "  $0 pipeline"
    echo "  $0 validate datasets/news_20250706.csv"
    echo "  $0 train datasets/news_20250706.csv models/my-model"
    echo "  $0 test models/finbert-20250706_142230"
    echo ""
    echo "Variables d'environnement :"
    echo "  HF_TOKEN    Token HuggingFace pour déploiement"
}

# Script principal
main() {
    case "${1:-help}" in
        "latest")
            show_latest
            ;;
        "list")
            list_datasets
            ;;
        "validate")
            run_validation "${2:-}"
            ;;
        "train")
            run_training "${2:-}" "${3:-}"
            ;;
        "pipeline")
            run_pipeline
            ;;
        "test")
            test_model "${2:-}"
            ;;
        "help"|"-h"|"--help"|*)
            show_help
            ;;
    esac
}

# Vérification que nous sommes dans le bon répertoire
if [[ ! -f "scripts/validate_dataset.py" ]] || [[ ! -f "scripts/finetune.py" ]]; then
    log_error "Scripts TradePulse ML non trouvés"
    log_info "Assurez-vous d'être dans le répertoire racine de tradepulse-ml"
    exit 1
fi

# Exécuter la fonction principale avec tous les arguments
main "$@"
