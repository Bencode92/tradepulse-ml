#!/bin/bash
# apply-isort-setup.sh - Configuration complète isort + pre-commit pour TradePulse ML

set -e  # Arrêt en cas d'erreur

echo "🛠️ Configuration isort + pre-commit pour TradePulse ML"
echo "=================================================="

# 1. Vérifier si isort est installé
echo "📦 Vérification des dépendances..."
if ! command -v isort &> /dev/null; then
    echo "⚠️ isort non trouvé, installation recommandée :"
    echo "   pip install --upgrade 'isort==5.13.2' 'ruff==0.3.4' pre-commit"
    echo ""
fi

# 2. Appliquer isort sur tous les fichiers Python
echo "🔧 Application d'isort sur scripts/..."
if command -v isort &> /dev/null; then
    isort --profile black scripts/ || {
        echo "⚠️ isort non disponible, passé"
    }
else
    echo "⚠️ isort non installé, formatage manuel nécessaire"
fi

# 3. Installer pre-commit hooks si pre-commit est disponible
echo "🎣 Configuration pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installés"
    
    # Test des hooks
    echo "🧪 Test des hooks pre-commit..."
    pre-commit run --files scripts/validate_dataset.py || {
        echo "⚠️ Hooks trouvés des problèmes (normaux en première installation)"
    }
else
    echo "⚠️ pre-commit non installé"
    echo "   Installation: pip install pre-commit"
fi

# 4. Vérification finale
echo ""
echo "📊 État de la configuration :"

# Vérifier .pre-commit-config.yaml
if [[ -f ".pre-commit-config.yaml" ]]; then
    echo "✅ .pre-commit-config.yaml présent"
else
    echo "❌ .pre-commit-config.yaml manquant"
fi

# Vérifier si Black est retiré du workflow
if grep -q "black" .github/workflows/tests.yml 2>/dev/null; then
    echo "⚠️ Black encore présent dans tests.yml"
else
    echo "✅ Black retiré de tests.yml"
fi

# Vérifier si isort est configuré
if grep -q "isort" .github/workflows/tests.yml 2>/dev/null; then
    echo "✅ isort configuré dans tests.yml"
else
    echo "⚠️ isort non trouvé dans tests.yml"
fi

echo ""
echo "🎯 Configuration terminée !"
echo ""
echo "📋 Prochaines étapes :"
echo "  1. Si pas encore fait :"
echo "     pip install --upgrade 'isort==5.13.2' 'ruff==0.3.4' pre-commit"
echo ""
echo "  2. Pour développement quotidien :"
echo "     git add scripts/mon_fichier.py"
echo "     git commit -m 'feat: nouvelle fonctionnalité'"
echo "     # ↳ isort s'exécute automatiquement !"
echo ""
echo "  3. En cas de problème CI :"
echo "     isort --profile black scripts/"
echo "     git add scripts/ && git commit -m 'style: fix imports' && git push"
echo ""
echo "✨ Développement sans stress avec formatage automatique ! 🚀"
