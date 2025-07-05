# 🤖 TradePulse ML - FinBERT Fine-tuning

Repository privé pour le fine-tuning de modèles FinBERT pour l'analyse de sentiment financier de TradePulse.

## 🚀 Utilisation rapide

### 1. Via GitHub Actions (Recommandé)

1. **Préparez votre dataset** dans le dossier `datasets/`
2. Allez dans l'onglet **Actions** de ce repository
3. Sélectionnez "🤖 TradePulse FinBERT Fine-tuning"
4. Cliquez "Run workflow" et configurez :
   - **Dataset**: `news_20250705.csv` (ou votre fichier)
   - **Model**: `yiyanghkust/finbert-tone`
   - **Epochs**: `3`
   - **Learning rate**: `2e-5`
   - **Push to HuggingFace**: `true/false`

### 2. En local

```bash
# Cloner le repository
git clone https://github.com/Bencode92/tradepulse-ml.git
cd tradepulse-ml

# Installer les dépendances
pip install -r requirements.txt

# Lancer le fine-tuning
python scripts/finetune.py \
    --dataset datasets/news_20250705.csv \
    --output_dir models/finbert-v1 \
    --epochs 3 \
    --lr 2e-5
```

## 📁 Structure du repository

```
tradepulse-ml/
├── 📁 .github/workflows/    # GitHub Actions
│   └── finetune-model.yml   # Workflow de fine-tuning
├── 📁 datasets/             # Datasets d'entraînement
│   └── news_20250705.csv    # Exemple de dataset
├── 📁 models/               # Modèles entraînés (généré)
├── 📁 scripts/              # Scripts Python
│   └── finetune.py          # Script principal de fine-tuning
├── requirements.txt         # Dépendances Python
└── README.md               # Ce fichier
```

## 📊 Format des datasets

### Format CSV
```csv
text,label
"Apple reported strong earnings...",positive
"Market volatility increased...",negative
"Oil prices remained stable...",neutral
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

## ⚙️ Configuration

### Variables d'environnement (GitHub Secrets)

Pour utiliser les fonctionnalités avancées, configurez ces secrets dans **Settings > Secrets**:

- `HF_TOKEN`: Token HuggingFace (obligatoire pour push de modèles)
- `WANDB_API_KEY`: Token Weights & Biases (optionnel)

### Arguments du script

```bash
python scripts/finetune.py --help
```

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

## 📈 Monitoring

### Logs d'entraînement
- **TensorBoard**: `models/[model_name]/logs/`
- **Fichiers log**: `finetune.log`
- **Rapport**: `models/[model_name]/training_report.json`

### Métriques générées
- **Accuracy**: Précision globale
- **F1-Score**: Score F1 pondéré
- **Precision**: Précision par classe
- **Recall**: Rappel par classe

## 🔧 Modèles supportés

- `yiyanghkust/finbert-tone` (Recommandé)
- `ProsusAI/finbert`
- `nlptown/bert-base-multilingual-uncased-sentiment`

## 📝 Exemple d'utilisation

### 1. Préparer un dataset personnalisé

```python
import pandas as pd

# Créer un dataset depuis vos données TradePulse
data = [
    {"text": "Tesla stock surged after earnings beat", "label": "positive"},
    {"text": "Market correction continues amid uncertainty", "label": "negative"},
    {"text": "Oil prices stable following OPEC meeting", "label": "neutral"}
]

df = pd.DataFrame(data)
df.to_csv("datasets/my_dataset.csv", index=False)
```

### 2. Lancer le fine-tuning

```bash
python scripts/finetune.py \
    --dataset datasets/my_dataset.csv \
    --output_dir models/tradepulse-custom \
    --epochs 5 \
    --lr 1e-5 \
    --push \
    --hub_id Bencode92/tradepulse-finbert-custom
```

### 3. Utiliser le modèle entraîné

```python
from transformers import pipeline

# Charger le modèle fine-tuné
classifier = pipeline(
    "text-classification",
    model="models/tradepulse-custom",
    tokenizer="models/tradepulse-custom"
)

# Analyser un texte
result = classifier("Apple reported strong quarterly results")
print(result)
# [{'label': 'positive', 'score': 0.95}]
```

## 🚨 Dépannage

### Erreurs communes

1. **"Dataset file not found"**
   - Vérifiez que le fichier existe dans `datasets/`
   - Utilisez le nom exact du fichier

2. **"HF_TOKEN required"**
   - Configurez le secret `HF_TOKEN` dans GitHub
   - Ou désactivez `push_to_hub`

3. **"Out of memory"**
   - Réduisez `train_bs` et `eval_bs`
   - Réduisez `max_length`

### Logs utiles

```bash
# Voir les logs d'entraînement
tail -f finetune.log

# Vérifier TensorBoard
tensorboard --logdir models/[model_name]/logs/
```

## 🔄 Intégration avec TradePulse

Le modèle fine-tuné peut être intégré dans le système principal :

1. **Push vers HuggingFace Hub** avec `--push --hub_id`
2. **Configurer le modèle custom** dans `fmp_news_updater.py`:
   ```python
   _FINBERT_MODEL = "Bencode92/tradepulse-finbert-custom"
   USE_CUSTOM_FINBERT = True
   ```

## 📞 Support

Pour toute question ou problème :
- Ouvrir une issue dans ce repository
- Vérifier les logs d'Actions GitHub
- Consulter la documentation HuggingFace Transformers

---

**TradePulse ML** - Fine-tuning FinBERT pour l'analyse de sentiment financier 🚀
