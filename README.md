# 🤖 TradePulse ML - FinBERT Fine-tuning

Repository privé pour le fine-tuning de modèles FinBERT pour l'analyse de sentiment financier de TradePulse avec **validation automatique des datasets**.

## 🚀 Utilisation rapide

### 1. Via GitHub Actions (Recommandé)

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

### 2. Workflow Pull Request (Nouveauté 🔥)

```bash
# 1. Créer une branche pour votre dataset
git checkout -b feature/dataset-20250706

# 2. Ajouter votre dataset
cp mon_dataset.csv datasets/financial_news_20250706.csv
git add datasets/financial_news_20250706.csv
git commit -m "Add Q4 financial news dataset"
git push origin feature/dataset-20250706

# 3. Créer une Pull Request
# → Validation automatique + rapport de qualité
# → Commentaire auto sur la PR avec résultats
# → Merge = déclenchement automatique du fine-tuning
```

### 3. En local

```bash
# Cloner le repository
git clone https://github.com/Bencode92/tradepulse-ml.git
cd tradepulse-ml

# Installer les dépendances
pip install -r requirements.txt

# Valider votre dataset (NOUVEAU)
python scripts/validate_dataset.py datasets/mon_dataset.csv

# Lancer le fine-tuning
python scripts/finetune.py \
    --dataset datasets/mon_dataset.csv \
    --output_dir models/finbert-v1 \
    --epochs 3 \
    --lr 2e-5
```

## 📁 Structure du repository

```
tradepulse-ml/
├── 📁 .github/workflows/          # GitHub Actions
│   ├── finetune-model.yml         # Workflow de fine-tuning (amélioré)
│   └── dataset-quality-gate.yml   # Validation des datasets (NOUVEAU)
├── 📁 datasets/                   # Datasets d'entraînement
│   ├── news_20250705.csv          # Exemple dataset (15 échantillons)
│   ├── financial_news_20250706.csv # Dataset test (20 échantillons)
│   ├── 📁 raw/                    # Données brutes
│   └── 📁 labeled/                # Données étiquetées
├── 📁 models/                     # Modèles entraînés (généré)
├── 📁 scripts/                    # Scripts Python
│   ├── finetune.py                # Script principal de fine-tuning
│   └── validate_dataset.py        # Validation datasets (NOUVEAU)
├── requirements.txt               # Dépendances Python
├── DATASET_WORKFLOW.md           # Guide validation (NOUVEAU)
└── README.md                     # Ce fichier
```

## 🔍 Validation automatique des datasets (NOUVEAU !)

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
- **Sélection auto** : Dernier dataset validé
- **Déclenchement** : Push sur `datasets/` après validation
- **Configuration** : Paramètres optimisés par défaut
- **Artifacts** : Modèles et logs automatiquement sauvés

## ⚙️ Configuration

### Variables d'environnement (GitHub Secrets)

Pour utiliser les fonctionnalités avancées, configurez ces secrets dans **Settings > Secrets**:

- `HF_TOKEN`: Token HuggingFace (obligatoire pour push de modèles)
- `WANDB_API_KEY`: Token Weights & Biases (optionnel)

### Arguments du script de validation

```bash
python scripts/validate_dataset.py --help
```

| Argument | Description | Défaut |
|----------|-------------|---------|
| `dataset_path` | Chemin vers le CSV/JSON | **Requis** |
| `--max-length` | Longueur max des textes | `512` |
| `--min-samples` | Échantillons minimum | `10` |
| `--quiet` | Mode silencieux | `False` |

### Arguments du script de fine-tuning

| Argument | Description | Défaut |
|----------|-------------|---------|
| `--dataset` | Chemin vers le dataset | **Requis** |
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

## 📝 Exemple d'utilisation complète

### 1. Préparer un dataset

```python
import pandas as pd

# Créer un dataset depuis vos données TradePulse
data = [
    {"text": "Tesla stock surged after earnings beat expectations", "label": "positive"},
    {"text": "Market correction continues amid economic uncertainty", "label": "negative"},
    {"text": "Oil prices stable following OPEC+ meeting decision", "label": "neutral"}
]

df = pd.DataFrame(data)
df.to_csv("datasets/my_dataset_20250706.csv", index=False)
```

### 2. Valider le dataset

```bash
# Validation locale
python scripts/validate_dataset.py datasets/my_dataset_20250706.csv

# Ou via GitHub Actions (manuel)
# Actions → Dataset Quality Gate → Run workflow
```

### 3. Fine-tuning automatique

```bash
# Option A: Push direct (validation + training auto)
git add datasets/my_dataset_20250706.csv
git commit -m "Add validated dataset for training"
git push origin main

# Option B: Via Pull Request (recommandé)
git checkout -b feature/dataset-20250706
git add datasets/my_dataset_20250706.csv
git commit -m "Add comprehensive dataset for Q4"
git push origin feature/dataset-20250706
# → Créer PR → Validation → Merge → Training auto
```

### 4. Utiliser le modèle entraîné

```python
from transformers import pipeline

# Charger le modèle fine-tuné
classifier = pipeline(
    "text-classification",
    model="models/finbert-20250706_143022",  # Dossier généré
    tokenizer="models/finbert-20250706_143022"
)

# Analyser un texte
result = classifier("Apple reported strong quarterly results")
print(result)
# [{'label': 'positive', 'score': 0.95}]
```

## 🚨 Dépannage

### Erreurs de validation

```bash
# Voir les détails
python scripts/validate_dataset.py datasets/problematic.csv

# Erreurs communes et solutions :
# ❌ "Labels invalides" → Utiliser positive/negative/neutral
# ❌ "Colonnes incorrectes" → Renommer en text,label  
# ❌ "Textes manquants" → Supprimer lignes vides
# ⚠️ "Classe déséquilibrée" → Ajouter échantillons minoritaires
```

### Workflow GitHub Actions

1. **Dataset validation failed**
   - Consulter les logs dans Actions → Dataset Quality Gate
   - Corriger les erreurs selon le rapport
   - Re-push pour relancer la validation

2. **Training not triggered**
   - Vérifier que la validation a réussi
   - S'assurer que le fichier est dans `datasets/` (pas sous-dossier)
   - Vérifier les logs Actions → TradePulse FinBERT Fine-tuning

### Logs utiles

```bash
# Logs locaux
tail -f finetune.log

# TensorBoard
tensorboard --logdir models/[model_name]/logs/

# GitHub Actions
# Actions → [Workflow] → [Run] → [Job] → Logs détaillés
```

## 📚 Documentation détaillée

- **[DATASET_WORKFLOW.md](DATASET_WORKFLOW.md)** : Guide complet du workflow de validation
- **Validation locale** : `python scripts/validate_dataset.py --help`
- **Fine-tuning** : `python scripts/finetune.py --help`

## 🔄 Intégration avec TradePulse

Le modèle fine-tuné peut être intégré dans le système principal :

1. **Push vers HuggingFace Hub** avec `--push --hub_id`
2. **Configurer le modèle custom** dans `fmp_news_updater.py`:
   ```python
   _FINBERT_MODEL = "Bencode92/tradepulse-finbert-custom"
   USE_CUSTOM_FINBERT = True
   ```

## 🎯 Prochaines étapes

- [ ] Intégration WANDB pour tracking avancé
- [ ] Tests A/B automatiques entre modèles  
- [ ] Pipeline de data augmentation
- [ ] Interface web pour annotation
- [ ] Monitoring de drift des données

## 📞 Support

Pour toute question ou problème :
- 🐛 **Issues** : Ouvrir une issue dans ce repository
- 📋 **Logs** : Consulter Actions → Job logs
- 📖 **Documentation** : DATASET_WORKFLOW.md + ce README
- 🔍 **Validation** : `python scripts/validate_dataset.py --help`

---

**TradePulse ML** - Fine-tuning FinBERT avec validation automatique pour l'analyse de sentiment financier 🚀✨

*Nouveau : Dataset Quality Gate pour des modèles plus fiables !*
