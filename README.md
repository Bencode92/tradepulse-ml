# 🤖 TradePulse ML Pipeline

**REPOSITORY PRIVÉ** - Fine-tuning et entraînement de modèles

## 🏗️ Structure

```
tradepulse-ml/
├── notebooks/              # Jupyter notebooks pour expérimentation
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── datasets/               # Données d'entraînement
│   ├── raw/               # Données brutes collectées
│   ├── labeled/           # Données labellisées manuellement
│   └── processed/         # Données préparées pour l'entraînement
├── models/                # Modèles entraînés
│   ├── experiments/       # Modèles expérimentaux
│   ├── production/        # Modèles prêts pour la prod
│   └── benchmarks/        # Résultats de performance
├── scripts/               # Scripts d'entraînement
│   ├── finetune.py       # Script principal de fine-tuning
│   ├── evaluate.py       # Évaluation des modèles
│   └── data_prep.py      # Préparation des données
└── config/                # Configurations d'entraînement
    ├── training_config.yaml
    └── model_config.yaml
```

## 🎯 Objectifs

- **Fine-tuning FinBERT** pour l'analyse de sentiment financier
- **Données propriétaires** : Articles TradePulse labellisés
- **Amélioration continue** : Pipeline d'entraînement automatisé
- **Versioning des modèles** : Suivi des performances

## 🔒 Sécurité

- ✅ Repository **100% privé**
- ✅ Aucun code d'entraînement exposé en production
- ✅ Modèles stockés sur infrastructure privée
- ✅ Accès restreint aux données sensibles

## 🚀 Usage

### Entraînement d'un nouveau modèle
```bash
python scripts/finetune.py \
    --base_model yiyanghkust/finbert-tone \
    --train_file datasets/labeled/train.csv \
    --eval_file datasets/labeled/eval.csv \
    --output_dir models/production/v2.5.0 \
    --epochs 3 \
    --learning_rate 2e-5
```

### Déploiement en production
```bash
# Upload vers S3
aws s3 cp models/production/v2.5.0 s3://tp-models/v2.5.0 --recursive

# Ou vers Hugging Face Hub privé
huggingface-cli upload models/production/v2.5.0 tradepulse/finbert-sentiment-v2.5.0 --private
```

## 📊 Métriques de Performance

Chaque modèle est évalué sur :
- **Accuracy** : Précision globale
- **F1-Score** : Moyenne pondérée par classe
- **Confidence Distribution** : Distribution des scores de confiance
- **Financial Domain Accuracy** : Performance sur terminologie financière

---

**🔥 TradePulse Investor-Grade ML Pipeline**
