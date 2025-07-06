# 🛠️ Guide de Correction CI - Formatage Code

Ce guide vous aide à résoudre rapidement les erreurs de formatage dans la CI TradePulse ML.

## 🚨 Problème : CI échoue sur le formatage

Votre CI échoue avec des erreurs comme :
- `❌ Code formatting issues found` (Black)
- `❌ Import sorting issues found` (isort) 
- `❌ Linting issues found` (Ruff)

## ⚡ Solution Express (2 minutes)

### Option 1: Script automatique (RECOMMANDÉ)

```bash
# 1. Cloner/mettre à jour le repo
git pull

# 2. Exécuter le script de correction
chmod +x fix-ci.sh
./fix-ci.sh
```

Le script fait tout automatiquement : formatage, commit, push.

### Option 2: Commandes manuelles

```bash
# 1. Installer les outils
python -m pip install --upgrade black isort ruff

# 2. Appliquer les corrections
black --line-length 88 scripts/
isort --profile black scripts/
ruff check scripts/ --fix

# 3. Commiter les changements
git add scripts/
git commit -m "style: format code with black/isort/ruff (CI fix)"
git push
```

## 🔧 Configuration Préventive

Pour ne plus jamais avoir ce problème :

### 1. Installation environnement complet

```bash
chmod +x setup-dev.sh
./setup-dev.sh
```

### 2. Hooks pre-commit automatiques

```bash
pip install pre-commit
pre-commit install
# Les hooks s'exécutent maintenant à chaque commit !
```

### 3. Vérification avant commit

```bash
# Vérifier le formatage
./scripts/format-check.sh

# Corriger automatiquement
./scripts/format-check.sh --fix
```

## 📋 Outils Disponibles

| Script | Usage | Description |
|--------|-------|-------------|
| `fix-ci.sh` | `./fix-ci.sh` | Correction express pour CI |
| `scripts/format-check.sh` | `./scripts/format-check.sh [--fix]` | Vérification formatage |
| `setup-dev.sh` | `./setup-dev.sh` | Installation environnement complet |

## 🎯 Configuration IDE (VSCode)

Le fichier `.vscode/settings.json` est configuré pour :
- ✅ Black avec line-length 88
- ✅ isort avec profil Black  
- ✅ Ruff pour le linting
- ✅ Formatage automatique à la sauvegarde

## 🔍 Détails Techniques

La CI vérifie exactement :
```bash
black --check --line-length=88 scripts/
isort --check-only --profile=black scripts/
ruff check scripts/ --show-source
```

Nos outils utilisent les **mêmes paramètres** pour garantir la compatibilité.

## ❓ Dépannage

### "Permission denied" sur les scripts
```bash
chmod +x fix-ci.sh setup-dev.sh scripts/format-check.sh
```

### "Module not found" pour les outils
```bash
python -m pip install --upgrade black isort ruff pre-commit
```

### Vérifier la CI après correction
1. Allez dans l'onglet **Actions** de GitHub
2. Trouvez le workflow "🧪 Tests & Code Quality"
3. Vérifiez que le job "code-quality" passe ✅

## 🎉 Prochaines Étapes

1. **IMMÉDIAT** : `./fix-ci.sh` → Corriger la CI
2. **AUJOURD'HUI** : `./setup-dev.sh` → Environnement complet
3. **DEMAIN** : Développement sans friction ! 🚀

---

💡 **Astuce** : Avec pre-commit installé, le formatage est vérifié automatiquement à chaque commit, plus besoin d'y penser !
