#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration générale pour TradePulse ML
"""

# Utiliser le correlator local au lieu de chercher dans stock-analysis-platform
USE_LOCAL_CORRELATOR = True

# Modèles HuggingFace
HF_MODELS = {
    "sentiment": "Bencode92/tradepulse-finbert-sentiment",
    "importance": "Bencode92/tradepulse-finbert-importance", 
    "correlations": "Bencode92/tradepulse-finbert-correlations"
}

# Seuils ML
ML_THRESHOLDS = {
    "confidence": 0.75,
    "correlation_min": 0.22,
    "correlation_max": 0.55
}
