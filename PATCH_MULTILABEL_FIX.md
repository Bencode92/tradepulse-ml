# Patch pour finetune.py - Correction multi-label dimensions

## Problème
Le modèle de base FinBERT a 3 labels, mais le modèle de corrélations en a 125.
Sans `ignore_mismatched_sizes=True`, le chargement échoue.

## Solution
Ajouter `ignore_mismatched_sizes=True` dans tous les appels `from_pretrained` pour multi-label.

## Modifications à faire dans scripts/finetune.py

### 1. Dans __init__ - Mode incrémental (ligne ~340)
```python
if self.is_multi_label:
    self.model = AutoModelForSequenceClassification.from_pretrained(
        baseline_model,
        num_labels=self.num_labels,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,  # 👈 AJOUTER CETTE LIGNE
    )
```

### 2. Dans __init__ - Mode incrémental fallback (ligne ~360)
```python
if self.is_multi_label:
    self.model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=self.num_labels,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,  # 👈 AJOUTER CETTE LIGNE
    )
```

### 3. Dans __init__ - Mode classique (ligne ~380)
```python
if self.is_multi_label:
    self.model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=self.num_labels,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,  # 👈 AJOUTER CETTE LIGNE
    )
```

## Application manuelle
1. Ouvrir scripts/finetune.py
2. Rechercher "problem_type=\"multi_label_classification\""
3. Ajouter ignore_mismatched_sizes=True après chaque occurrence

## Vérification
Après modification, relancer le workflow. Le modèle de corrélations devrait s'entraîner sans erreur de dimensions.
