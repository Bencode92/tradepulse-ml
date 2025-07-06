#!/bin/bash

# 🛠️ Script de correction express pour CI TradePulse ML
# Usage: ./fix-ci.sh

set -e  # Arrêter en cas d'erreur

echo "🚀 TradePulse ML - Correction CI Express"
echo "========================================"

# Vérifier qu'on est dans le bon repo
if [ ! -f "news_editor.html" ] || [ ! -d "scripts" ]; then
    echo "❌ Erreur: Ce script doit être exécuté depuis la racine du repo tradepulse-ml"
    exit 1
fi

# 1) Mise à jour
echo ""
echo "📥 1. Mise à jour du repository..."
git pull || echo "⚠️ Échec git pull (peut-être déjà à jour)"

# 2) Installation des outils
echo ""
echo "🔧 2. Installation des outils de formatage..."
python -m pip install --upgrade black isort ruff

# 3) Formatage avec Black
echo ""
echo "🖤 3. Formatage du code avec Black (line-length 88)..."
black --line-length 88 scripts/
echo "✅ Black appliqué"

# 4) Tri des imports avec isort
echo ""
echo "📦 4. Tri des imports avec isort..."
isort --profile black scripts/
echo "✅ isort appliqué"

# 5) Correction des lints avec Ruff
echo ""
echo "⚡ 5. Correction des problèmes de linting avec Ruff..."
ruff check scripts/ --fix || echo "⚠️ Certains problèmes Ruff nécessitent une correction manuelle"
echo "✅ Ruff appliqué"

# 6) Vérification des changements
echo ""
echo "🔍 6. Vérification des changements..."
if git diff --quiet scripts/; then
    echo "ℹ️ Aucun changement détecté - le code était déjà formaté correctement"
else
    echo "📝 Changements détectés:"
    git diff --name-only scripts/
fi

# 7) Ajout et commit
echo ""
echo "💾 7. Ajout des fichiers modifiés..."
git add scripts/

if git diff --cached --quiet; then
    echo "ℹ️ Aucun changement à commiter"
else
    echo "📝 Commit des changements..."
    git commit -m "style: format code with black/isort/ruff (CI fix)

- Applied black --line-length 88 to scripts/
- Fixed import ordering with isort --profile black
- Auto-fixed linting issues with ruff --fix"
    
    echo ""
    echo "🚀 8. Push vers GitHub..."
    git push
    
    echo ""
    echo "✅ SUCCÈS ! Les changements ont été poussés."
    echo "🔍 La CI devrait maintenant passer."
    echo "📋 Vérifiez l'onglet Actions de votre repository GitHub."
fi

echo ""
echo "🎉 Script terminé avec succès !"
echo ""
echo "💡 CONSEIL: Pour éviter ces problèmes à l'avenir:"
echo "   • Installez pre-commit: pip install pre-commit && pre-commit install"
echo "   • Configurez votre IDE pour utiliser Black avec line-length 88"
echo "   • Exécutez ./scripts/format-check.sh avant chaque commit"
