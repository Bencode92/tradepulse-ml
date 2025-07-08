#!/usr/bin/env python3
"""
Script pour corriger les problèmes du dataset TradePulse
Résout: textes trop longs, caractères spéciaux, labels incohérents
"""

import pandas as pd
import re
import csv
from typing import Dict, List, Tuple

def clean_text(text: str, max_length: int = 512) -> str:
    """Nettoie et tronque le texte"""
    if not text:
        return ""
    
    # Supprimer caractères de contrôle et normaliser espaces
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer guillemets doubles redondants
    text = re.sub(r'""', '"', text)
    
    # Convertir majuscules excessives
    if text.isupper() and len(text) > 50:
        text = text.title()
    
    # Tronquer si trop long
    if len(text) > max_length:
        # Couper intelligemment au dernier point/espace
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_space = truncated.rfind(' ')
        
        if last_period > max_length * 0.8:
            text = truncated[:last_period + 1]
        elif last_space > max_length * 0.8:
            text = truncated[:last_space]
        else:
            text = truncated
    
    return text.strip()

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les labels sentiment et importance"""
    df = df.copy()
    
    # Mapping sentiment
    sentiment_mapping = {
        'positive': 'positive',
        'pos': 'positive',
        'bullish': 'positive',
        'negative': 'negative', 
        'neg': 'negative',
        'bearish': 'negative',
        'neutral': 'neutral',
        'neu': 'neutral',
        'mixed': 'neutral'
    }
    
    # Mapping importance avec accents normalisés
    importance_mapping = {
        'critique': 'critique',
        'importante': 'importante',
        'important': 'importante',
        'générale': 'générale',
        'generale': 'générale',
        'general': 'générale',
        'high': 'critique',
        'medium': 'importante',
        'low': 'générale'
    }
    
    # Normaliser sentiment
    if 'label' in df.columns:
        df['label'] = df['label'].str.lower().str.strip()
        df['label'] = df['label'].map(sentiment_mapping).fillna('neutral')
    
    # Normaliser importance
    if 'importance' in df.columns:
        df['importance'] = df['importance'].str.lower().str.strip()
        df['importance'] = df['importance'].map(importance_mapping).fillna('générale')
    
    return df

def validate_dataset(df: pd.DataFrame) -> Dict[str, any]:
    """Valide le dataset et retourne un rapport"""
    issues = []
    stats = {}
    
    # Vérifier colonnes requises
    required_cols = ['text', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Colonnes manquantes: {missing_cols}")
    
    # Statistiques texte
    if 'text' in df.columns:
        text_lengths = df['text'].str.len()
        stats['text_length_avg'] = text_lengths.mean()
        stats['text_length_max'] = text_lengths.max()
        stats['text_length_min'] = text_lengths.min()
        
        # Textes trop longs
        long_texts = (text_lengths > 512).sum()
        if long_texts > 0:
            issues.append(f"{long_texts} textes trop longs (>512 car)")
    
    # Distribution labels
    if 'label' in df.columns:
        label_dist = df['label'].value_counts()
        stats['label_distribution'] = label_dist.to_dict()
        
        # Vérifier labels valides
        valid_labels = {'positive', 'negative', 'neutral'}
        invalid_labels = set(df['label'].unique()) - valid_labels
        if invalid_labels:
            issues.append(f"Labels invalides: {invalid_labels}")
    
    # Distribution importance
    if 'importance' in df.columns:
        imp_dist = df['importance'].value_counts()
        stats['importance_distribution'] = imp_dist.to_dict()
        
        # Vérifier importance valide
        valid_importance = {'critique', 'importante', 'générale'}
        invalid_importance = set(df['importance'].unique()) - valid_importance
        if invalid_importance:
            issues.append(f"Importance invalide: {invalid_importance}")
    
    # Doublons
    duplicates = df.duplicated(subset=['text']).sum()
    if duplicates > 0:
        issues.append(f"{duplicates} doublons détectés")
    
    return {
        'issues': issues,
        'stats': stats,
        'is_valid': len(issues) == 0
    }

def fix_dataset(input_file: str, output_file: str = None) -> str:
    """Corrige le dataset et sauvegarde"""
    
    # Charger dataset
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Dataset chargé: {len(df)} lignes")
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return ""
    
    # Validation initiale
    initial_report = validate_dataset(df)
    print(f"\n📊 Rapport initial:")
    print(f"Issues: {len(initial_report['issues'])}")
    for issue in initial_report['issues']:
        print(f"  ⚠️ {issue}")
    
    # Corrections
    print(f"\n🔧 Application des corrections...")
    
    # 1. Nettoyer textes
    if 'text' in df.columns:
        df['text'] = df['text'].apply(clean_text)
        print(f"  ✅ Textes nettoyés et tronqués")
    
    # 2. Normaliser labels
    df = normalize_labels(df)
    print(f"  ✅ Labels normalisés")
    
    # 3. Supprimer doublons
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    after_dedup = len(df)
    if before_dedup != after_dedup:
        print(f"  ✅ {before_dedup - after_dedup} doublons supprimés")
    
    # 4. Supprimer lignes avec texte vide
    df = df[df['text'].str.strip() != '']
    print(f"  ✅ Lignes vides supprimées: {len(df)} lignes restantes")
    
    # Validation finale
    final_report = validate_dataset(df)
    print(f"\n📊 Rapport final:")
    print(f"Issues: {len(final_report['issues'])}")
    print(f"Statut: {'✅ VALIDE' if final_report['is_valid'] else '❌ ERREURS'}")
    
    # Statistiques détaillées
    stats = final_report['stats']
    if 'label_distribution' in stats:
        print(f"\n😊 Distribution sentiment:")
        for label, count in stats['label_distribution'].items():
            pct = count / len(df) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
    
    if 'importance_distribution' in stats:
        print(f"\n🎯 Distribution importance:")
        for imp, count in stats['importance_distribution'].items():
            pct = count / len(df) * 100
            print(f"  {imp}: {count} ({pct:.1f}%)")
    
    if 'text_length_avg' in stats:
        print(f"\n📏 Longueur texte:")
        print(f"  Moyenne: {stats['text_length_avg']:.1f} caractères")
        print(f"  Min/Max: {stats['text_length_min']}/{stats['text_length_max']}")
    
    # Sauvegarder
    if not output_file:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"datasets/news_fixed_{timestamp}.csv"
    
    try:
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
        print(f"\n💾 Dataset corrigé sauvé: {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
        return ""

def main():
    """Point d'entrée principal"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fix_dataset.py <input_file> [output_file]")
        print("Exemple: python fix_dataset.py datasets/news_20250708.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("🔧 TradePulse Dataset Fixer")
    print("=" * 50)
    
    result = fix_dataset(input_file, output_file)
    
    if result:
        print(f"\n🎉 Correction terminée!")
        print(f"📁 Fichier de sortie: {result}")
        print(f"\n💡 Prochaines étapes:")
        print(f"1. Valider: python scripts/validate_dataset.py {result}")
        print(f"2. Entraîner: python scripts/finetune.py --dataset {result} --output_dir models/test")
        print(f"3. Ou utiliser l'éditeur web: open news_editor.html")
    else:
        print(f"\n❌ Échec de la correction")
        sys.exit(1)

if __name__ == "__main__":
    main()
