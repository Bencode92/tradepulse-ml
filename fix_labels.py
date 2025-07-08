#!/usr/bin/env python3
# Patch pour corriger les labels sentiment

import pandas as pd

# Charger et corriger
df = pd.read_csv('datasets/news_20250708.csv')

# Nettoyage labels
df['label'] = df['label'].str.strip().str.lower()
df['importance'] = df['importance'].str.strip().str.lower()

# Normalisation accents pour importance
df['importance'] = df['importance'].replace({
    'generale': 'générale',
    'general': 'générale'
})

# Sauvegarder
df.to_csv('datasets/news_20250708_fixed.csv', index=False)
print("✅ Dataset corrigé sauvé : news_20250708_fixed.csv")
