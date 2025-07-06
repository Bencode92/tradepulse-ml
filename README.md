# 🤖 TradePulse ML - FinBERT Fine-tuning

Repository privé pour le fine-tuning de modèles FinBERT pour l'analyse de sentiment financier de TradePulse avec **validation automatique des datasets** et **interface web moderne**.

## 🚀 Utilisation rapide

### 1. Via l'éditeur web (NOUVEAU ! 🔥)

1. **Ouvrir l'éditeur** : `open news_editor.html` ou double-clic
2. **Charger un dataset** : Import local ou depuis GitHub
3. **Éditer visuellement** : Tableau interactif avec validation temps réel
4. **Sauvegarder** : Download local ou commit direct vers GitHub
5. **Auto-déclenchement** : Le commit lance automatiquement le fine-tuning !

### 2. Via pipeline express (NOUVEAU ! ⚡)

```bash
# Pipeline complet en une commande
./scripts/auto-pipeline.sh pipeline

# Commandes rapides
./scripts/auto-pipeline.sh latest      # Affiche le dernier dataset
./scripts/auto-pipeline.sh validate    # Valide le dernier dataset  
./scripts/auto-pipeline.sh train       # Entraîne sur le dernier dataset
```

### 3. Via scripts avec auto-sélection (NOUVEAU ! 🎯)

```bash
# Plus besoin de spécifier le fichier !
python scripts/validate_dataset.py     # Auto-sélectionne le dernier CSV
python scripts/finetune.py --output_dir models/test  # Auto-sélectionne le dernier CSV

# Rétrocompatibilité : les anciens appels fonctionnent toujours
python scripts/validate_dataset.py datasets/news_20250706.csv
python scripts/finetune.py --dataset datasets/news_20250706.csv --output_dir models/test
```

### 4. Via GitHub Actions (existant)

1. **Préparez votre dataset** dans le dossier `datasets/`
2. **Validation automatique** : Le système vérifie la qualité de vos données
3. Allez dans l'onglet **Actions** de ce repository
4. Sélectionnez "🤖 TradePulse FinBERT Fine-tuning"
5. Cliquez "Run workflow" et configurez :
   - **Dataset**: `auto-latest` (dernier dataset) ou nom spécifique
   - **Model**: `yiyanghkust/finbert-tone`
   - **Epochs**: `3`
   - **Learning rate**: `2e-5`
   - **Push to HuggingFace**: `true/false`

## 📁 Structure du repository

```
tradepulse-ml/
├── 🌐 news_editor.html              # NOUVEAU ! Éditeur web moderne
├── 📁 .github/workflows/            # GitHub Actions
│   ├── finetune-model.yml           # Workflow de fine-tuning (amélioré)
│   ├── dataset-quality-gate.yml     # Validation des datasets
│   ├── collect-dataset.yml          # Collecte automatique
│   └── tests.yml                    # Tests automatisés
├── 📁 datasets/                     # Datasets d'entraînement
│   ├── news_20250705.csv            # Exemple dataset (15 échantillons)
│   ├── financial_news_20250706.csv  # Dataset test (20 échantillons)
│   ├── news_20250706.csv            # Dataset test (25 échantillons)
│   ├── 📁 raw/                      # Données brutes
│   └── 📁 labeled/                  # Données étiquetées
├── 📁 models/                       # Modèles entraînés (généré)
├── 📁 scripts/                      # Scripts Python
│   ├── finetune.py                  # Script principal de fine-tuning (+ auto-sélection)
│   ├── validate_dataset.py          # Validation datasets (+ auto-sélection)
│   ├── utils.py                     # NOUVEAU ! Utilitaires d'auto-sélection
│   ├── auto-pipeline.sh             # NOUVEAU ! Pipeline express
│   ├── collect_news.py              # Collecte d'actualités
│   └── test_validation.py           # Tests de validation
├── 📁 config/                       # Configuration
├── 📁 notebooks/                    # Jupyter notebooks
├── requirements.txt                 # Dépendances Python de base
├── requirements-ml.txt              # Dépendances ML complètes
├── DATASET_WORKFLOW.md             # Guide validation
├── ENTERPRISE_UPGRADE.md           # Guide entreprise
├── BUG_FIXES_AND_FEATURES.md      # Changelog
└── README.md                       # Ce fichier
```

## 🎨 Éditeur web moderne (NOUVEAU !)

### ✨ Fonctionnalités

- **Interface glassmorphism** avec design moderne
- **Validation temps réel** avec statistiques visuelles  
- **Intégration GitHub** : charger/sauver directement
- **Édition inline** : tableau entièrement éditable
- **Auto-détection** du format et des erreurs
- **Déclenchement workflows** depuis l'interface

### 🔧 Utilisation

```bash
# 1. Ouvrir l'éditeur
open news_editor.html  # ou double-clic

# 2. Configurer GitHub (optionnel)
Repository: Bencode92/tradepulse-ml
Branch: main
Token: ghp_xxx... (scope: repo)

# 3. Charger un dataset
- "Charger dernier CSV GitHub" (auto-sélection)
- "Importer CSV local" (fichier local)

# 4. Éditer les données
- Cliquer dans les cellules pour éditer
- Ajouter des lignes avec le bouton "+"
- Validation automatique en temps réel

# 5. Sauvegarder
- "Télécharger CSV" (export local)
- "Commit vers GitHub" (push direct + déclenche CI)
```

## ⚡ Pipeline express (NOUVEAU !)

### 🚀 Commandes disponibles

```bash
# Affichage
./scripts/auto-pipeline.sh latest     # Dernier dataset + aperçu
./scripts/auto-pipeline.sh list       # Liste tous les datasets

# Workflows
./scripts/auto-pipeline.sh validate   # Validation seule
./scripts/auto-pipeline.sh train      # Fine-tuning seul  
./scripts/auto-pipeline.sh pipeline   # Validation + Fine-tuning

# Tests & déploiement
./scripts/auto-pipeline.sh test models/finbert-xxx    # Test rapide
./scripts/auto-pipeline.sh deploy models/finbert-xxx  # Push HuggingFace
```

### 💫 Exemples d'utilisation

```bash
# Workflow quotidien simplifié
./scripts/auto-pipeline.sh latest     # Vérifier le dernier dataset
./scripts/auto-pipeline.sh pipeline   # Lancer le pipeline complet
# ↓ Résultat : validation + entraînement + modèle prêt !

# Workflow avec fichier spécifique
./scripts/auto-pipeline.sh validate datasets/news_20250706.csv
./scripts/auto-pipeline.sh train datasets/news_20250706.csv models/my-model

# Test d'un modèle entraîné
./scripts/auto-pipeline.sh test models/finbert-20250706_142230
```

## 🎯 Auto-sélection intelligente (NOUVEAU !)

Tous les scripts détectent automatiquement le **dernier dataset** au format `news_YYYYMMDD.csv` :

### ✅ Scripts avec auto-sélection

```bash
# Validation automatique
python scripts/validate_dataset.py
# ↓ Équivalent à :
# python scripts/validate_dataset.py datasets/news_20250706.csv

# Fine-tuning automatique  
python scripts/finetune.py --output_dir models/test
# ↓ Équivalent à :
# python scripts/finetune.py --dataset datasets/news_20250706.csv --output_dir models/test

# Pipeline express
./scripts/auto-pipeline.sh pipeline
# ↓ Détecte automatiquement le dernier dataset et l'utilise
```

### 🔧 Comment ça fonctionne

1. **Détection** : Scan du dossier `datasets/` pour fichiers `news_*.csv`
2. **Tri** : Tri alphabétique décroissant (YYYYMMDD) = plus récent en premier
3. **Sélection** : Utilise automatiquement le premier fichier trouvé
4. **Fallback** : Si échec, erreur claire avec suggestions

## 🔍 Validation automatique des datasets

Le système valide automatiquement vos datasets pour garantir la qualité :

### ✅ Vérifications automatiques
- **Structure** : Colonnes `text` et `label` requises
- **Labels** : Seulement `positive`, `negative`, `neutral`
- **Qualité** : Détection doublons, textes vides, longueur
- **Distribution** : Équilibrage des classes
- **Format** : CSV et JSON supportés

### 📊 Rapport de validation
```
🔍 RAPPORT DE VALIDATION DATASET
==================================================

📊 STATISTIQUES:
  Total échantillons: 20
  Longueur moyenne: 156.4 caractères
  Doublons: 0

📈 DISTRIBUTION DES LABELS:
  positive: 8 (40.0%)
  negative: 6 (30.0%)  
  neutral: 6 (30.0%)

✅ VALIDATION RÉUSSIE
```

## 📊 Format des datasets

### Format CSV (Recommandé)
```csv
text,label
"Apple reported strong earnings beating expectations...",positive
"Market volatility increased amid economic uncertainty...",negative
"Oil prices remained stable following OPEC meeting...",neutral
```

### Format JSON
```json
[
  {
    "text": "Apple reported strong earnings...",
    "label": "positive"
  },
  {
    "text": "Market volatility increased...",
    "label": "negative"
  }
]
```

### 📏 Critères de qualité
- **Labels valides** : `positive`, `negative`, `neutral` uniquement
- **Longueur texte** : 20-512 caractères recommandés
- **Pas de doublons** dans les textes
- **Distribution équilibrée** : Éviter >70% d'une seule classe
- **Minimum** : 10 échantillons (50+ recommandé)

## 🤖 Workflows automatisés

### 1. Dataset Quality Gate
- **Déclenchement** : Pull Request touchant `datasets/`
- **Validation** : Structure, contenu, distribution
- **Rapport** : Commentaire automatique sur PR
- **Blocage** : Empêche merge si validation échoue

### 2. Smart Fine-tuning
- **Sélection auto** : Dernier dataset validé (`auto-latest`)
- **Déclenchement** : Push sur `datasets/` après validation
- **Configuration** : Paramètres optimisés par défaut
- **Artifacts** : Modèles et logs automatiquement sauvés

### 3. Collecte automatique (existant)
- **Source** : APIs d'actualités financières
- **Filtrage** : Mots-clés financiers
- **Nettoyage** : Suppression doublons et contenu non pertinent
- **Format** : Export direct en CSV prêt pour labeling

## ⚙️ Configuration

### Variables d'environnement (GitHub Secrets)

Pour utiliser les fonctionnalités avancées, configurez ces secrets dans **Settings > Secrets**:

- `HF_TOKEN`: Token HuggingFace (obligatoire pour push de modèles)
- `WANDB_API_KEY`: Token Weights & Biases (optionnel)

### Arguments des scripts (mise à jour)

#### Script de validation
```bash
python scripts/validate_dataset.py [dataset_path] [options]
```

| Argument | Description | Défaut |
|----------|-------------|---------|
| `dataset_path` | Chemin vers le CSV/JSON (optionnel avec auto-sélection) | Auto-détection |
| `--max-length` | Longueur max des textes | `512` |
| `--min-samples` | Échantillons minimum | `10` |
| `--quiet` | Mode silencieux | `False` |
| `--output-json` | Rapport JSON | `None` |

#### Script de fine-tuning
```bash
python scripts/finetune.py --output_dir models/test [options]
```

| Argument | Description | Défaut |
|----------|-------------|---------|
| `--dataset` | Chemin vers le dataset (optionnel avec auto-sélection) | Auto-détection |
| `--output_dir` | Répertoire de sortie | **Requis** |
| `--model_name` | Modèle de base | `yiyanghkust/finbert-tone` |
| `--epochs` | Nombre d'époques | `3` |
| `--lr` | Taux d'apprentissage | `2e-5` |
| `--train_bs` | Batch size train | `16` |
| `--eval_bs` | Batch size eval | `32` |
| `--push` | Push vers HF Hub | `False` |
| `--hub_id` | ID du repo HF | `None` |

## 📈 Monitoring & MLOps

### Logs d'entraînement
- **TensorBoard**: `models/[model_name]/logs/`
- **Fichiers log**: `finetune.log`
- **Rapport**: `models/[model_name]/training_report.json`

### GitHub Actions
- **Artifacts** : Modèles et logs téléchargeables
- **Notifications** : Statut des jobs par email
- **History** : Historique complet des entraînements

### Métriques générées
- **Accuracy**: Précision globale
- **F1-Score**: Score F1 pondéré
- **Precision**: Précision par classe
- **Recall**: Rappel par classe

## 🔧 Modèles supportés

- `yiyanghkust/finbert-tone` (Recommandé)
- `ProsusAI/finbert`
- `nlptown/bert-base-multilingual-uncased-sentiment`

## 📝 Exemples d'utilisation complète

### 1. Workflow moderne avec éditeur web

```bash
# 1. Ouvrir l'éditeur
open news_editor.html

# 2. Charger le dernier dataset depuis GitHub
# Clic sur "Charger dernier CSV GitHub"

# 3. Éditer visuellement les données
# - Corriger les labels invalides (rouge → vert)
# - Ajouter des lignes manquantes
# - Validation temps réel

# 4. Commit vers GitHub
# Clic sur "Commit vers GitHub"
# → Déclenche automatiquement le fine-tuning !
```

### 2. Workflow pipeline express

```bash
# Pipeline ultra-rapide
./scripts/auto-pipeline.sh pipeline

# Résultat :
# ✅ Dataset auto-détecté : datasets/news_20250706.csv
# ✅ Validation réussie
# ✅ Fine-tuning terminé
# ✅ Modèle sauvé : models/finbert-20250706_142230
```

### 3. Workflow traditionnel amélioré

```bash
# Plus simple qu'avant !
python scripts/validate_dataset.py     # Auto-sélection
python scripts/finetune.py --output_dir models/finbert-auto

# Équivalent à l'ancienne méthode :
# python scripts/validate_dataset.py datasets/news_20250706.csv
# python scripts/finetune.py --dataset datasets/news_20250706.csv --output_dir models/finbert-auto
```

### 4. Test et déploiement

```bash
# Test rapide du modèle
./scripts/auto-pipeline.sh test models/finbert-20250706_142230

# Déploiement vers HuggingFace
./scripts/auto-pipeline.sh deploy models/finbert-20250706_142230 Bencode92/tradepulse-finbert-v2
```

## 🚨 Dépannage

### Auto-sélection

```bash
# Vérifier les datasets disponibles
./scripts/auto-pipeline.sh list

# Tester l'auto-sélection
python -c "from scripts.utils import latest_dataset; print(latest_dataset())"

# Format requis pour l'auto-détection
datasets/news_YYYYMMDD.csv  # Exemple : news_20250706.csv
```

### Éditeur web

```bash
# Problèmes de connexion GitHub
# → Vérifier le token (scope: repo)
# → Vérifier le nom du repository (owner/nom)

# Problèmes de validation
# → Colonnes requises : text, label
# → Labels valides : positive, negative, neutral

# Performance
# → Fonctionne jusqu'à ~1000 lignes
# → Pour plus : utiliser les scripts Python
```

### Pipeline express

```bash
# Script non exécutable
chmod +x scripts/auto-pipeline.sh

# Aucun dataset trouvé
# → Ajouter des fichiers au format news_YYYYMMDD.csv dans datasets/

# Erreur de dépendances
pip install -r requirements.txt
```

## 📚 Documentation détaillée

- **[DATASET_WORKFLOW.md](DATASET_WORKFLOW.md)** : Guide complet du workflow de validation
- **[ENTERPRISE_UPGRADE.md](ENTERPRISE_UPGRADE.md)** : Guide de mise à niveau entreprise
- **[BUG_FIXES_AND_FEATURES.md](BUG_FIXES_AND_FEATURES.md)** : Changelog complet
- **Validation locale** : `python scripts/validate_dataset.py --help`
- **Fine-tuning** : `python scripts/finetune.py --help`
- **Pipeline express** : `./scripts/auto-pipeline.sh help`

## 🔄 Intégration avec TradePulse

Le modèle fine-tuné peut être intégré dans le système principal :

1. **Push vers HuggingFace Hub** avec `--push --hub_id`
2. **Configurer le modèle custom** dans `fmp_news_updater.py`:
   ```python
   _FINBERT_MODEL = "Bencode92/tradepulse-finbert-custom"
   USE_CUSTOM_FINBERT = True
   ```

## 🎯 Nouveautés et améliorations

### ✨ Version actuelle
- **🌐 Éditeur web moderne** avec interface glassmorphism
- **⚡ Pipeline express** en une commande
- **🎯 Auto-sélection intelligente** dans tous les scripts
- **🔧 Utilitaires Python** réutilisables (`utils.py`)
- **📊 Validation temps réel** dans l'interface web
- **🚀 Déclenchement workflows** depuis l'éditeur

### 🔮 Prochaines étapes
- [ ] Intégration WANDB pour tracking avancé
- [ ] Tests A/B automatiques entre modèles  
- [ ] Pipeline de data augmentation
- [ ] Annotation collaborative multi-utilisateurs
- [ ] Monitoring de drift des données
- [ ] API REST pour intégration externe

## 📞 Support

Pour toute question ou problème :
- 🐛 **Issues** : Ouvrir une issue dans ce repository
- 📋 **Logs** : Consulter Actions → Job logs
- 📖 **Documentation** : DATASET_WORKFLOW.md + ce README
- 🔍 **Validation** : `python scripts/validate_dataset.py --help`
- ⚡ **Pipeline** : `./scripts/auto-pipeline.sh help`

---

**TradePulse ML** - Fine-tuning FinBERT avec interface web moderne et automation complète 🚀✨

*Nouveau : Éditeur web + Auto-sélection + Pipeline express pour une productivité maximale !*
