#!/bin/bash

# 🔍 Script de vérification formatage pour TradePulse ML
# Usage: ./scripts/format-check.sh [--fix]

set -e

echo "🔍 TradePulse ML - Vérification Formatage"
echo "========================================"

FIX_MODE=false
if [ "$1" = "--fix" ]; then
    FIX_MODE=true
    echo "🔧 Mode correction activé"
fi

# Vérifier qu'on est dans le bon repo
if [ ! -d "scripts" ]; then
    echo "❌ Erreur: dossier scripts/ introuvable"
    exit 1
fi

echo ""
echo "📂 Vérification du dossier scripts/..."

# Initialiser les compteurs
ERRORS=0

# 1) Vérification Black
echo ""
echo "🖤 1. Vérification Black (line-length 88)..."
if $FIX_MODE; then
    black --line-length 88 scripts/
    echo "✅ Black appliqué"
else
    if black --check --line-length 88 scripts/; then
        echo "✅ Formatage Black OK"
    else
        echo "❌ Problèmes de formatage Black détectés"
        echo "💡 Correction: black --line-length 88 scripts/"
        ((ERRORS++))
    fi
fi

# 2) Vérification isort
echo ""
echo "📦 2. Vérification isort (profil Black)..."
if $FIX_MODE; then
    isort --profile black scripts/
    echo "✅ isort appliqué"
else
    if isort --check-only --profile black scripts/; then
        echo "✅ Tri des imports OK"
    else
        echo "❌ Problèmes de tri d'imports détectés"
        echo "💡 Correction: isort --profile black scripts/"
        ((ERRORS++))
    fi
fi

# 3) Vérification Ruff
echo ""
echo "⚡ 3. Vérification Ruff..."
if $FIX_MODE; then
    ruff check scripts/ --fix || echo "⚠️ Certains problèmes Ruff nécessitent une correction manuelle"
    echo "✅ Ruff appliqué"
else
    if ruff check scripts/ --quiet; then
        echo "✅ Linting Ruff OK"
    else
        echo "❌ Problèmes de linting Ruff détectés"
        echo "💡 Correction: ruff check scripts/ --fix"
        ((ERRORS++))
    fi
fi

# 4) Résumé
echo ""
echo "📊 RÉSUMÉ:"
echo "========="

if [ $ERRORS -eq 0 ]; then
    echo "✅ Tous les contrôles sont passés !"
    echo "🚀 Le code est prêt pour commit/push"
    exit 0
else
    echo "❌ $ERRORS problème(s) détecté(s)"
    echo ""
    echo "🛠️ CORRECTION RAPIDE:"
    echo "   ./scripts/format-check.sh --fix"
    echo ""
    echo "🛠️ CORRECTION MANUELLE:"
    echo "   black --line-length 88 scripts/"
    echo "   isort --profile black scripts/"
    echo "   ruff check scripts/ --fix"
    exit 1
fi
