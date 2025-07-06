#!/bin/bash

# 🔧 Script de configuration environnement de développement TradePulse ML
# Usage: ./setup-dev.sh

set -e

echo "🔧 TradePulse ML - Configuration Environnement Développement"
echo "==========================================================="

# Vérifications préliminaires
if [ ! -f "news_editor.html" ]; then
    echo "❌ Ce script doit être exécuté depuis la racine du repo tradepulse-ml"
    exit 1
fi

# 1. Installation des outils de développement
echo ""
echo "📦 1. Installation des outils de développement..."
python -m pip install --upgrade pip
pip install black isort ruff pre-commit pytest

# 2. Configuration pre-commit
echo ""
echo "🎣 2. Configuration des hooks pre-commit..."

# Installer les hooks (le fichier .pre-commit-config.yaml existe déjà)
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    echo "✅ Hooks pre-commit installés"
else
    echo "⚠️ Fichier .pre-commit-config.yaml introuvable"
    echo "💡 Assurez-vous que ce script est exécuté après le push du fichier de config"
fi

# 3. Configuration IDE (VSCode)
echo ""
echo "💻 3. Configuration IDE (VSCode)..."

mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.sortImports.args": ["--profile", "black"],
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.blackEnabled": false,
    "python.linting.pylintEnabled": false,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/logs": true,
        "**/models/*/": true
    }
}
EOF
echo "✅ Configuration VSCode créée dans .vscode/settings.json"

# 4. Test initial du formatage
echo ""
echo "🧪 4. Test initial du formatage..."
if [ -f "scripts/format-check.sh" ]; then
    chmod +x scripts/format-check.sh
    if ./scripts/format-check.sh --fix; then
        echo "✅ Formatage initial appliqué"
    else
        echo "⚠️ Quelques ajustements manuels peuvent être nécessaires"
    fi
else
    echo "⚠️ Script format-check.sh introuvable, application manuelle..."
    if command -v black &> /dev/null; then
        black --line-length 88 scripts/ || echo "⚠️ Erreur Black"
        isort --profile black scripts/ || echo "⚠️ Erreur isort"  
        ruff check scripts/ --fix || echo "⚠️ Erreur Ruff"
    fi
fi

# 5. Test des hooks pre-commit
echo ""
echo "🎣 5. Test des hooks pre-commit..."
if command -v pre-commit &> /dev/null && [ -f ".pre-commit-config.yaml" ]; then
    pre-commit run --all-files || echo "⚠️ Quelques hooks ont fait des corrections"
else
    echo "⚠️ Pre-commit non configuré, ignoré"
fi

# 6. Rendre les scripts exécutables
echo ""
echo "🔧 6. Configuration des permissions scripts..."
chmod +x fix-ci.sh 2>/dev/null || echo "fix-ci.sh introuvable"
chmod +x scripts/format-check.sh 2>/dev/null || echo "scripts/format-check.sh introuvable"
chmod +x scripts/auto-pipeline.sh 2>/dev/null || echo "scripts/auto-pipeline.sh introuvable"

echo ""
echo "🎉 CONFIGURATION TERMINÉE !"
echo "=========================="
echo ""
echo "✅ Environnement de développement configuré"
echo "✅ Pre-commit hooks installés (si config disponible)"
echo "✅ Configuration VSCode ajoutée"
echo "✅ Scripts rendus exécutables"
echo ""
echo "📋 COMMANDES UTILES:"
echo "   ./fix-ci.sh                       # Corriger CI immédiatement"
echo "   ./scripts/format-check.sh         # Vérifier le formatage"
echo "   ./scripts/format-check.sh --fix   # Corriger le formatage" 
echo "   pre-commit run --all-files        # Exécuter tous les hooks"
echo "   git commit                        # Les hooks s'exécutent automatiquement"
echo ""
echo "💡 PROCHAINES ÉTAPES:"
echo "   1. Exécutez ./fix-ci.sh pour corriger la CI immédiatement"
echo "   2. Le formatage sera vérifié automatiquement à chaque commit"
echo "   3. Configurez votre IDE avec les settings VSCode fournis"
