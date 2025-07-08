#!/usr/bin/env python3
import pandas as pd
from collections import Counter

# Charger le dataset
df = pd.read_csv('datasets/news_20250708.csv')

print("=== DEBUG SENTIMENT ===")
print(f"Colonnes: {list(df.columns)}")
print(f"Total lignes: {len(df)}")

# Vérifier labels sentiment
print("\n📊 Labels sentiment bruts:")
print(df['label'].value_counts())

print("\n📊 Labels sentiment lower():")
print(df['label'].str.lower().value_counts())

# Mapping utilisé dans le code
SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}
print(f"\n🎯 Mapping attendu: {SENTIMENT_MAP}")

# Vérifier quels labels ne matchent pas
valid_labels = set(SENTIMENT_MAP.keys())
actual_labels = set(df['label'].str.lower().str.strip())
missing = actual_labels - valid_labels
extra = valid_labels - actual_labels

print(f"\n❌ Labels non reconnus: {missing}")
print(f"⚠️ Labels attendus manquants: {extra}")

# Simuler le preprocessing
valid_rows = 0
for _, row in df.iterrows():
    label = row.get('label', '').lower().strip()
    text = row.get('text', '')
    if text and label in SENTIMENT_MAP:
        valid_rows += 1

print(f"\n✅ Lignes valides après preprocessing: {valid_rows}/{len(df)}")

# Test importance pour comparaison
IMPORTANCE_MAP = {"générale": 0, "importante": 1, "critique": 2}
print(f"\n🎯 Importance mapping: {IMPORTANCE_MAP}")
print("📊 Labels importance:")
print(df['importance'].str.lower().value_counts())
