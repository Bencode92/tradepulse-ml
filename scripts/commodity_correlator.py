#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stub CommodityCorrelator pour éviter les erreurs d'import
"""

class CommodityCorrelator:
    """Fallback correlator basique"""
    
    def __init__(self):
        self.country_exports = {}
    
    def _is_company_article(self, text: str) -> bool:
        """Vérifie si l'article parle d'une entreprise spécifique"""
        company_keywords = ['inc.', 'corp.', 'ltd.', 'llc', 'earnings', 'ceo', 'stock price']
        text_lower = text.lower()
        return any(kw in text_lower for kw in company_keywords)
    
    def _is_macro_article(self, text: str) -> bool:
        """Vérifie si l'article parle de macro-économie"""
        macro_keywords = ['gdp', 'inflation', 'interest rate', 'unemployment', 'trade', 'export', 'import']
        text_lower = text.lower()
        return any(kw in text_lower for kw in macro_keywords)
    
    def detect_countries_from_text(self, text: str) -> list:
        """Détecte les pays mentionnés dans le texte"""
        countries = []
        country_names = {
            'china': 'CN', 'united states': 'US', 'usa': 'US', 'germany': 'DE',
            'france': 'FR', 'brazil': 'BR', 'russia': 'RU', 'india': 'IN',
            'japan': 'JP', 'canada': 'CA', 'australia': 'AU', 'uk': 'GB'
        }
        
        text_lower = text.lower()
        for name, code in country_names.items():
            if name in text_lower:
                countries.append(code)
        
        return list(set(countries))
    
    def get_country_exports(self, country: str) -> list:
        """Retourne les exports d'un pays (stub)"""
        # Quelques exports de base pour le fallback
        basic_exports = {
            'CN': [{'product_code': 'MACHINERY', 'impact': 'major'}],
            'US': [{'product_code': 'CORN', 'impact': 'major'}],
            'BR': [{'product_code': 'COFFEE', 'impact': 'major'}],
            'AU': [{'product_code': 'IRON_ORE', 'impact': 'major'}]
        }
        return basic_exports.get(country, [])
    
    def _mentions_product(self, text: str, product_code: str) -> bool:
        """Vérifie si un produit est mentionné"""
        product_keywords = {
            'CORN': ['corn', 'maize'],
            'COFFEE': ['coffee'],
            'IRON_ORE': ['iron ore', 'iron'],
            'MACHINERY': ['machinery', 'equipment']
        }
        
        text_lower = text.lower()
        keywords = product_keywords.get(product_code, [product_code.lower()])
        return any(kw in text_lower for kw in keywords)
