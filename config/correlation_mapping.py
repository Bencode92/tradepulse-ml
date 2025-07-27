#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapping des corrélations commodités pour TradePulse ML
"""

# Liste complète des codes commodités
COMMODITY_CODES = [
    "AE:DIAMONDS", "AE:PETROLEUM_CRUDE",
    "AR:CORN",
    "AU:ALUMINIUM_ORE", "AU:COAL", "AU:IRON_ORE", "AU:NATGAS", "AU:WHEAT", "AU:ZINC_ORE",
    "BE:ZINC_METAL",
    "BO:ZINC_ORE",
    "BR:COFFEE", "BR:CORN", "BR:IRON_ORE", "BR:MEAT", "BR:SOYBEAN", "BR:SUGAR",
    "CA:ALUMINIUM_METAL", "CA:NICKEL_METAL", "CA:WHEAT",
    "CD:COPPER_REFINED",
    "CH:GOLD", "CH:PHARMACEUTICALS",
    "CL:COPPER_ORE", "CL:COPPER_REFINED", "CL:COPPER_UNREFINED",
    "CN:APPAREL", "CN:CARBON", "CN:CHEMICALS_MISC", "CN:CHEMICALS_ORGANIC", "CN:ELECTRICAL_MACHINERY",
    "CN:FURNITURE", "CN:MACHINERY", "CN:NICKEL_METAL", "CN:OPTICAL_INSTRUMENTS", "CN:PAPER",
    "CN:PLASTICS", "CN:RARE_GASES", "CN:RUBBER", "CN:SHIPS", "CN:SILVER", "CN:TOYS", "CN:VEHICLES", "CN:WOOD",
    "DE:AIRCRAFT", "DE:CHEMICALS_MISC", "DE:COCOA", "DE:FINANCIAL_SERVICES", "DE:OPTICAL_INSTRUMENTS",
    "DE:PAPER", "DE:PHARMACEUTICALS", "DE:RARE_GASES", "DE:VEHICLES",
    "FI:NICKEL_ORE",
    "FR:AIRCRAFT", "FR:BEVERAGES", "FR:COSMETICS", "FR:ELECTRICITY", "FR:FINANCIAL_SERVICES",
    "GB:GOLD", "GB:PLATINUM", "GB:SILVER",
    "GN:ALUMINIUM_ORE",
    "HK:DIAMONDS", "HK:ELECTRICAL_MACHINERY", "HK:SILVER",
    "ID:COAL", "ID:FERROALLOYS", "ID:PALM_OIL", "ID:TIN",
    "IN:DIAMONDS", "IN:IT_SERVICES", "IN:RICE",
    "KR:SHIPS",
    "KZ:URANIUM",
    "LU:FINANCIAL_SERVICES",
    "MX:LEAD_ORE", "MX:PRECIOUS_METALS_ORE",
    "MY:PALM_OIL",
    "NA:URANIUM",
    "NG:URANIUM",
    "NL:COCOA",
    "NO:FISH", "NO:NATGAS", "NO:NICKEL_METAL",
    "PK:RICE",
    "PE:COPPER_ORE", "PE:LEAD_ORE", "PE:PRECIOUS_METALS_ORE", "PE:TIN", "PE:ZINC_ORE",
    "PH:NICKEL_ORE",
    "QA:NATGAS",
    "RU:COAL", "RU:PETROLEUM_CRUDE", "RU:PRECIOUS_METALS_ORE", "RU:WHEAT",
    "SE:COPPER_UNREFINED",
    "SG:FINANCIAL_SERVICES",
    "TH:RICE",
    "UA:CORN",
    "US:CHEMICALS_MISC", "US:CHEMICALS_ORGANIC", "US:CORN", "US:DIAMONDS", "US:EDIBLE_FRUITS",
    "US:LEAD_ORE", "US:MEAT", "US:NATGAS", "US:OPTICAL_INSTRUMENTS", "US:PETROLEUM_CRUDE",
    "US:PETROLEUM_REFINED", "US:PHARMACEUTICALS", "US:PLASTICS", "US:PLATINUM", "US:RARE_GASES",
    "US:SOYBEAN", "US:TRAVEL", "US:WHEAT", "US:ZINC_ORE",
    "VN:FOOTWEAR", "VN:RICE",
    "ZA:FERROALLOYS", "ZA:PLATINUM",
    "ZM:COPPER_UNREFINED"
]

def correlations_to_labels(correlations_str: str) -> list:
    """
    Convertit une chaîne de corrélations (séparées par ;) en vecteur binaire
    """
    if not correlations_str:
        return [0] * len(COMMODITY_CODES)
    
    correlations = [c.strip() for c in correlations_str.split(';') if c.strip()]
    labels = []
    
    for code in COMMODITY_CODES:
        labels.append(1 if code in correlations else 0)
    
    return labels

def labels_to_correlations(labels: list) -> str:
    """
    Convertit un vecteur binaire en chaîne de corrélations
    """
    if not labels or len(labels) != len(COMMODITY_CODES):
        return ""
    
    correlations = []
    for i, label in enumerate(labels):
        if label == 1:
            correlations.append(COMMODITY_CODES[i])
    
    return ";".join(correlations)
