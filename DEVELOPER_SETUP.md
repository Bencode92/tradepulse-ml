# 🛠️ Guide Setup Développeur - TradePulse ML

Ce guide vous aide à configurer votre environnement de développement pour être **100% aligné** avec la CI GitHub Actions.

## ⚡ **Setup Express (Recommandé)**

```bash
# 1. Installer les outils avec les MÊMES versions que la CI
pip install --upgrade 'isort==5.13.2' 'ruff==0.3.4' pre-commit

# 2. Appliquer isort sur tous les fichiers Python
isort --profile black scripts/

# 3. Installer les hooks pre-commit (automation)
pre-commit install

# 4. Tester la configuration
pre-commit run --all-files

# 5. Commit des changements si nécessaire
git add scripts/ .pre-commit-config.yaml
git commit -m "chore: apply isort and add pre-commit hook"
git push
```

## 🎯 **Que fait cette configuration ?**

### **isort avec --profile black**
- **Groupe les imports** : standard library → third-party → local imports
- **Tri automatique** des imports dans chaque groupe
- **Compatible avec Black** (si on le réutilise un jour)
- **Cohérence** entre tous les développeurs

### **Hooks pre-commit automatiques**
- **isort** : Reformate automatiquement les imports avant chaque commit
- **ruff** : Linting rapide avec corrections automatiques  
- **Hooks système** : Supprime les espaces, vérifie YAML/JSON, etc.

### **Synchronisation CI**
- **Mêmes versions** exactes que GitHub Actions
- **Mêmes arguments** et configuration
- **Zero surprise** : ce qui passe en local passe en CI

## 🔧 **Utilisation quotidienne**

Une fois configuré, c'est **zéro effort** :

```bash
# Développement normal
git add scripts/mon_script.py
git commit -m "feat: nouvelle fonctionnalité"
# ↳ Les hooks se déclenchent automatiquement !
# ↳ isort reformate les imports si nécessaire
# ↳ ruff corrige les problèmes de style détectables

git push
# ↳ La CI passe ✅ car tout est déjà conforme
```

## 🎨 **Configuration IDE (Optionnel)**

### **VSCode**
Le projet inclut déjà `.vscode/settings.json` :
- isort activé avec `--profile black`
- ruff comme linter
- Formatage automatique à la sauvegarde

### **PyCharm/IntelliJ**
1. **Settings → Tools → External Tools**
2. **Ajouter isort** :
   - Program: `isort`
   - Arguments: `--profile black $FilePath$`
3. **Assigner un raccourci clavier**

## 🚨 **Dépannage**

### **"isort check failed" en CI**
```bash
# Correction locale :
isort --profile black scripts/
git add scripts/ && git commit -m "style: fix import sorting" && git push
```

### **"ruff check failed" en CI**  
```bash
# Correction automatique locale :
ruff check scripts/ --fix
git add scripts/ && git commit -m "style: fix linting issues" && git push
```

### **Pre-commit ne fonctionne pas**
```bash
# Réinstallation :
pre-commit uninstall
pre-commit install
pre-commit run --all-files
```

## 📊 **Avantages de cette approche**

| Avantage | Impact |
|----------|--------|
| **🔄 Automation** | Plus jamais de "failed formatting" en CI |
| **👥 Cohérence équipe** | Même style pour tous les développeurs |
| **⚡ Productivité** | Focus sur le code, pas sur le style |
| **🎯 CI stable** | Pipeline qui passe à coup sûr |
| **📚 Lisibilité** | Imports toujours organisés de la même façon |

## 🎉 **Vous êtes prêt !**

Après ce setup :
- ✅ **Environnement aligné** avec la CI
- ✅ **Automation complète** du formatage  
- ✅ **Zero effort quotidien** 
- ✅ **CI toujours verte** 🟢

**Bon développement !** 🚀

---

*TradePulse ML - Configuration développeur optimisée*
