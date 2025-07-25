#!/bin/bash

# Patch pour corriger le problème os.chdir() dans finetune.py
# Exécuter depuis la racine du projet

echo "🔧 Application du patch pour finetune.py..."

# Créer une sauvegarde
cp scripts/finetune.py scripts/finetune.py.backup

# Appliquer le patch avec sed
# Supprimer la ligne os.chdir
sed -i '/:$/! s/os\.chdir(self\.repo_dir)/#os.chdir(self.repo_dir)  # CORRIGÉ: Ne pas changer de répertoire/' scripts/finetune.py

# Vérifier si le patch a été appliqué
if grep -q "#os.chdir(self.repo_dir)" scripts/finetune.py; then
    echo "✅ Patch appliqué avec succès!"
    echo ""
    echo "Changements effectués :"
    echo "- os.chdir() commenté pour maintenir l'accès aux modules config/"
    echo ""
    echo "Pour lancer l'entraînement des corrélations :"
    echo "python scripts/finetune.py \\"
    echo "  --dataset datasets/news_20250724.csv \\"
    echo "  --output_dir models/finbert-correlations \\"
    echo "  --target-column correlations \\"
    echo "  --epochs 3 --mode production --incremental --push"
else
    echo "❌ Erreur lors de l'application du patch"
    echo "Appliquer manuellement : commenter la ligne os.chdir(self.repo_dir)"
fi
