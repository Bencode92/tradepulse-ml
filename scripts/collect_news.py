#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradePulse News Collector - Smart Dual-Model ML Labeling
========================================================

Collecte avec double labellisation automatique:
- Sentiment: positive/negative/neutral
- Importance: critique/importante/générale

Usage:
    python scripts/collect_news.py --source fmp --count 60 --days 7 --auto-label
"""

import argparse
import csv
import datetime
import json
import logging
import os
import random
import zoneinfo
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import requests
import time

# Configuration des logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("smart-collector")

# Fuseau horaire Paris
PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")

# 🎯 MODÈLES SPÉCIALISÉS PRODUITS PAR LE WORKFLOW
ML_MODELS_CONFIG = {
    "sentiment": "Bencode92/tradepulse-finbert-sentiment",    # ✅ Modèle sentiment entraîné
    "importance": "Bencode92/tradepulse-finbert-importance",  # ✅ Modèle importance entraîné  
    "fallback": "yiyanghkust/finbert-tone",                   # Fallback de base
}

# Configuration importance (reprend de fmp_news_updater.py)
KEYWORD_TIERS = {
    "high": [
        # Chocs marché & macro
        "crash", "collapse", "contagion", "default", "downgrade", "stagflation",
        "recession", "sovereign risk", "yield spike", "volatility spike",
        # Banques centrales / inflation
        "cpi", "pce", "core inflation", "rate hike", "rate cut", "qt", "qe",
        # Crédit & liquidité
        "credit spread", "cds", "insolvency", "liquidity crunch",
        # Fondamentaux entreprise
        "profit warning", "guidance cut", "eps miss", "dividend cut",
        # Géopolitique
        "sanction", "embargo", "war", "conflict"
    ],
    "medium": [
        "earnings beat", "eps beat", "revenue beat", "free cash flow",
        "buyback", "merger", "acquisition", "spin-off", "ipo", "stake sale",
        "job cuts", "strike", "production halt", "regulation", "antitrust",
        "fine", "class action", "data breach", "rating watch",
        "payrolls", "unemployment rate", "pmi", "ism", "consumer confidence",
        "ppi", "housing starts"
    ]
}

PREMIUM_SOURCES = ["bloomberg", "reuters", "financial times", "wall street journal"]

# Configuration FMP
FMP_ENDPOINTS = {
    "general_news": "https://financialmodelingprep.com/api/v3/fmp/articles",
    "stock_news": "https://financialmodelingprep.com/api/v3/stock_news",
    "crypto_news": "https://financialmodelingprep.com/api/v4/crypto_news",
    "forex_news": "https://financialmodelingprep.com/api/v4/forex_news",
    "press_releases": "https://financialmodelingprep.com/api/v3/press-releases"
}

FMP_LIMITS = {
    "general_news": 20,
    "stock_news": 50,
    "crypto_news": 15,
    "forex_news": 10,
    "press_releases": 5
}

class SmartNewsCollector:
    """Collecteur avec double ML (sentiment + importance) - MODÈLES PRODUITS"""
    
    def __init__(self, output_dir: str = "datasets", enable_cache: bool = True, 
                 auto_label: bool = False, confidence_threshold: float = 0.75):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_cache = enable_cache
        self.cache_file = self.output_dir / ".article_cache.json"
        self.seen_articles: Set[str] = set()
        
        # API Key
        self.api_key = os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY environment variable required")
        
        # ML Configuration
        self.auto_label = auto_label
        self.confidence_threshold = confidence_threshold
        self.sentiment_classifier = None
        self.importance_classifier = None
        
        self._load_cache()
        
        if self.auto_label:
            self._load_ml_models()

    def _load_cache(self):
        """Charge le cache de déduplication"""
        if not self.enable_cache or not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                self.seen_articles = set(cache_data.get("articles", []))
            logger.info(f"🗄️ Cache chargé: {len(self.seen_articles)} articles connus")
        except Exception as e:
            logger.warning(f"Erreur chargement cache: {e}")
            self.seen_articles = set()

    def _save_cache(self):
        """Sauvegarde le cache de déduplication"""
        if not self.enable_cache:
            return
            
        try:
            cache_data = {"articles": list(self.seen_articles)}
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Erreur sauvegarde cache: {e}")

    def _load_ml_models(self):
        """🎯 Charge les 2 modèles spécialisés PRODUITS par le workflow"""
        try:
            from transformers import pipeline
            import torch
            
            hf_token = os.getenv("HF_TOKEN")
            model_kwargs = {"token": hf_token} if hf_token else {}
            device = 0 if torch.cuda.is_available() else -1
            
            # 1. Modèle sentiment PRODUIT
            try:
                logger.info(f"😊 Chargement modèle sentiment PRODUIT: {ML_MODELS_CONFIG['sentiment']}")
                self.sentiment_classifier = pipeline(
                    "text-classification",
                    model=ML_MODELS_CONFIG["sentiment"],
                    return_all_scores=True,
                    device=device,
                    **model_kwargs
                )
                logger.info("✅ Modèle sentiment PRODUIT chargé")
            except Exception as e:
                logger.warning(f"⚠️ Fallback sentiment: {e}")
                self.sentiment_classifier = pipeline(
                    "text-classification",
                    model=ML_MODELS_CONFIG["fallback"],
                    return_all_scores=True,
                    device=device
                )
            
            # 2. Modèle importance PRODUIT
            try:
                logger.info(f"🎯 Chargement modèle importance PRODUIT: {ML_MODELS_CONFIG['importance']}")
                self.importance_classifier = pipeline(
                    "text-classification",
                    model=ML_MODELS_CONFIG["importance"],
                    return_all_scores=True,
                    device=device,
                    **model_kwargs
                )
                logger.info("✅ Modèle importance PRODUIT chargé")
            except Exception as e:
                logger.warning(f"⚠️ Pas de modèle importance, utilisation règles: {e}")
                self.importance_classifier = None
                
        except Exception as e:
            logger.error(f"❌ Impossible de charger les modèles: {e}")
            self.sentiment_classifier = None
            self.importance_classifier = None

    def _predict_dual_labels(self, text: str) -> Tuple[str, str, float, float]:
        """🎯 Prédit sentiment ET importance avec les modèles PRODUITS"""
        text_truncated = text[:512]
        
        # 1. Prédiction sentiment avec modèle PRODUIT
        sentiment_label = "neutral"
        sentiment_confidence = 0.5
        
        if self.sentiment_classifier:
            try:
                results = self.sentiment_classifier(text_truncated)
                best_pred = max(results[0], key=lambda x: x['score'])
                
                # Normalisation labels sentiment (spécialisé PRODUIT)
                label_mapping = {
                    'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral',
                    'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral',
                    'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive',
                }
                
                sentiment_label = label_mapping.get(best_pred['label'], 'neutral')
                sentiment_confidence = best_pred['score']
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur prédiction sentiment: {e}")
                sentiment_label = self._basic_sentiment_analysis(text)
        else:
            sentiment_label = self._basic_sentiment_analysis(text)
        
        # 2. Prédiction importance avec modèle PRODUIT
        importance_label = "générale"
        importance_confidence = 0.5
        
        if self.importance_classifier:
            try:
                results = self.importance_classifier(text_truncated)
                best_pred = max(results[0], key=lambda x: x['score'])
                
                # Normalisation labels importance (spécialisé PRODUIT)
                importance_mapping = {
                    'critique': 'critique', 'importante': 'importante', 'générale': 'générale',
                    'LABEL_0': 'générale', 'LABEL_1': 'importante', 'LABEL_2': 'critique',
                }
                
                importance_label = importance_mapping.get(best_pred['label'], 'générale')
                importance_confidence = best_pred['score']
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur prédiction importance: {e}")
                importance_label = self._basic_importance_analysis(text)
        else:
            importance_label = self._basic_importance_analysis(text)
        
        return sentiment_label, importance_label, sentiment_confidence, importance_confidence

    def _basic_sentiment_analysis(self, text: str) -> str:
        """Analyse de sentiment basique (fallback)"""
        text_lower = text.lower()
        
        positive_words = ["gain", "rise", "surge", "rally", "beat", "growth", "strong", "bullish"]
        negative_words = ["drop", "fall", "decline", "crash", "loss", "weak", "bearish", "concern"]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    def _basic_importance_analysis(self, text: str) -> str:
        """🎯 Analyse d'importance basique avec mots-clés (fallback)"""
        text_lower = text.lower()
        
        high_score = sum(1 for kw in KEYWORD_TIERS["high"] if kw in text_lower)
        medium_score = sum(1 for kw in KEYWORD_TIERS["medium"] if kw in text_lower)
        
        if high_score >= 2:
            return "critique"
        elif high_score >= 1 or medium_score >= 3:
            return "importante"
        else:
            return "générale"

    def _article_hash(self, text: str) -> str:
        """Génère un hash unique pour un article"""
        import hashlib
        normalized = text.lower().strip()
        normalized = "".join(c for c in normalized if c.isalnum() or c.isspace())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _is_duplicate(self, text: str) -> bool:
        """Vérifie si un article est un doublon"""
        if not self.enable_cache:
            return False
        return self._article_hash(text) in self.seen_articles

    def _add_to_cache(self, text: str):
        """Ajoute un article au cache"""
        if self.enable_cache:
            self.seen_articles.add(self._article_hash(text))

    def fetch_fmp_data(self, endpoint: str, params: Dict = None) -> List[Dict]:
        """Récupère des données depuis l'API FMP"""
        if params is None:
            params = {}
        
        params["apikey"] = self.api_key
        
        try:
            logger.info(f"📡 Requête FMP: {endpoint}")
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                logger.info(f"✅ {len(data)} articles récupérés de FMP")
                return data
            else:
                logger.warning(f"⚠️ Format de réponse inattendu: {type(data)}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Erreur API FMP {endpoint}: {e}")
            return []

    def fetch_articles_by_period(self, endpoint: str, start_date: str, end_date: str, 
                                limit: int = 50, days_interval: int = 7) -> List[Dict]:
        """Récupère des articles sur une période donnée avec pagination"""
        from datetime import datetime, timedelta
        
        logger.info(f"📅 Collecte FMP: {start_date} à {end_date} (limite: {limit})")
        
        from_date = datetime.strptime(start_date, "%Y-%m-%d")
        to_date = datetime.strptime(end_date, "%Y-%m-%d")
        all_articles = []
        
        # Traitement par intervalles
        current_from = from_date
        while current_from < to_date and len(all_articles) < limit:
            current_to = min(current_from + timedelta(days=days_interval), to_date)
            
            # Pagination par intervalle
            for page in range(3):  # Max 3 pages par intervalle
                params = {
                    "from": current_from.strftime("%Y-%m-%d"),
                    "to": current_to.strftime("%Y-%m-%d"),
                    "page": page,
                    "limit": min(50, limit - len(all_articles))
                }
                
                articles = self.fetch_fmp_data(endpoint, params)
                
                if not articles:
                    break
                
                # Filtrage et ajout
                for article in articles:
                    title = article.get("title", "")
                    content = article.get("text", "") or article.get("content", "")
                    
                    if len(title) >= 20 and len(content) >= 100:
                        if not self._is_duplicate(title):
                            all_articles.append(article)
                            self._add_to_cache(title)
                            
                            if len(all_articles) >= limit:
                                break
                
                if len(articles) < params["limit"] or len(all_articles) >= limit:
                    break
            
            current_from = current_to
        
        logger.info(f"📊 Articles collectés: {len(all_articles)}")
        return all_articles

    def collect_fmp_news(self, count: int = 40, days: int = 7) -> List[Dict]:
        """Collecte des actualités depuis FMP avec répartition intelligente"""
        today = datetime.datetime.now(PARIS_TZ).date()
        start_date = (today - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        
        all_articles = []
        
        # Répartition par endpoint
        for endpoint_name, endpoint_url in FMP_ENDPOINTS.items():
            limit = FMP_LIMITS.get(endpoint_name, 10)
            # Ajuster selon le count total demandé
            adjusted_limit = int(limit * (count / 100))  # Proportion du total
            
            if adjusted_limit < 1:
                continue
                
            logger.info(f"🔍 Collecte {endpoint_name}: {adjusted_limit} articles max")
            
            articles = self.fetch_articles_by_period(
                endpoint_url, start_date, end_date, adjusted_limit
            )
            
            # Normalisation et enrichissement
            for article in articles:
                enriched = self._enrich_article(article, endpoint_name)
                if enriched:
                    all_articles.append(enriched)
        
        # Tri par qualité
        all_articles.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        
        return all_articles[:count]

    def _enrich_article(self, article: Dict, source_type: str) -> Optional[Dict]:
        """🎯 Enrichit un article avec double labellisation des MODÈLES PRODUITS"""
        try:
            title = article.get("title", "")
            content = article.get("text", "") or article.get("content", "")
            
            if not title or not content:
                return None
            
            # Structure normalisée
            enriched = {
                "text": f"{title}. {content}".strip(),
                "title": title,
                "summary": content[:200] + "..." if len(content) > 200 else content,
                "source": article.get("site", "") or article.get("publisher", "FMP"),
                "url": article.get("url", ""),
                "published_date": article.get("publishedDate", ""),
                "collected_at": datetime.datetime.now(PARIS_TZ).isoformat(),
                "source_type": source_type
            }
            
            # 🎯 Double prédiction avec MODÈLES PRODUITS
            if self.auto_label:
                sentiment_label, importance_label, sent_conf, imp_conf = self._predict_dual_labels(enriched["text"])
                
                enriched.update({
                    "label": sentiment_label,
                    "importance": importance_label,
                    "sentiment_confidence": sent_conf,
                    "importance_confidence": imp_conf,
                    "needs_review": sent_conf < self.confidence_threshold or imp_conf < self.confidence_threshold,
                    "sentiment_model": ML_MODELS_CONFIG["sentiment"] if self.sentiment_classifier else "rule_based",
                    "importance_model": ML_MODELS_CONFIG["importance"] if self.importance_classifier else "rule_based",
                    "labeling_method": "dual_ml_produit"
                })
            else:
                sentiment_label = self._basic_sentiment_analysis(enriched["text"])
                importance_label = self._basic_importance_analysis(enriched["text"])
                
                enriched.update({
                    "label": sentiment_label,
                    "importance": importance_label,
                    "sentiment_confidence": None,
                    "importance_confidence": None,
                    "needs_review": False,
                    "labeling_method": "rule_based_dual"
                })
            
            # Score qualité global
            enriched["quality_score"] = self._calculate_quality_score(enriched)
            
            return enriched
            
        except Exception as e:
            logger.warning(f"Erreur enrichissement article: {e}")
            return None

    def _calculate_quality_score(self, article: Dict) -> float:
        """Calcule un score de qualité"""
        # Base: longueur
        title_len = len(article.get("title", ""))
        text_len = len(article.get("text", ""))
        
        score = min(20, title_len / 5) + min(30, text_len / 100)
        
        # Bonus confiance ML des MODÈLES PRODUITS
        if article.get("sentiment_confidence"):
            score += article["sentiment_confidence"] * 10
        if article.get("importance_confidence"):
            score += article["importance_confidence"] * 10
        
        # Bonus source premium
        source = article.get("source", "").lower()
        if any(premium in source for premium in PREMIUM_SOURCES):
            score += 25
        
        # Bonus mots-clés importants
        text_lower = article.get("text", "").lower()
        high_kw = sum(1 for kw in KEYWORD_TIERS["high"] if kw in text_lower)
        score += high_kw * 5
        
        return min(100, score)

    def save_dataset(self, articles: List[Dict], output_file: Optional[Path] = None) -> Path:
        """🎯 Sauvegarde avec 3 colonnes: text, label, importance (MODÈLES PRODUITS)"""
        if output_file is None:
            today = datetime.datetime.now(PARIS_TZ).strftime("%Y%m%d")
            output_file = self.output_dir / f"news_{today}.csv"

        # Sauvegarde CSV avec 3 colonnes
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label", "importance"])  # 3 colonnes
            
            for article in articles:
                writer.writerow([
                    article["text"], 
                    article["label"], 
                    article["importance"]
                ])

        # Métadonnées JSON
        labels = [article["label"] for article in articles]
        importance_labels = [article["importance"] for article in articles]
        
        label_counts = {label: labels.count(label) for label in set(labels)}
        importance_counts = {label: importance_labels.count(label) for label in set(importance_labels)}
        
        needs_review_count = sum(1 for article in articles if article.get("needs_review", False))
        
        metadata = {
            "filename": output_file.name,
            "created_at": datetime.datetime.now(PARIS_TZ).isoformat(),
            "source": "fmp_smart_produit",
            "article_count": len(articles),
            "label_distribution": label_counts,
            "importance_distribution": importance_counts,
            "deduplication_enabled": self.enable_cache,
            "cache_size": len(self.seen_articles),
            "dual_ml_enabled": self.auto_label,
            "sentiment_model": ML_MODELS_CONFIG["sentiment"] if self.auto_label else None,
            "importance_model": ML_MODELS_CONFIG["importance"] if self.auto_label else None,
            "confidence_threshold": self.confidence_threshold if self.auto_label else None,
            "high_confidence_articles": len(articles) - needs_review_count,
            "needs_review_articles": needs_review_count,
            "models_source": "workflow_produit"  # Indique que les modèles viennent du workflow
        }

        json_file = output_file.with_suffix('.json')
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Sauvegarde cache
        self._save_cache()

        logger.info(f"✅ Dataset Smart PRODUIT: {output_file} ({len(articles)} échantillons)")
        return output_file

    def collect_and_save(self, count: int = 40, days: int = 7, output_file: Optional[Path] = None) -> Path:
        """🎯 Pipeline complet avec MODÈLES PRODUITS (sentiment + importance)"""
        logger.info(f"🚀 Collecte Smart PRODUIT: {count} articles, {days} jours")
        
        if self.auto_label:
            logger.info(f"🎯 Double ML PRODUIT activé: sentiment + importance")
            logger.info(f"😊 Modèle sentiment: {ML_MODELS_CONFIG['sentiment']}")
            logger.info(f"🎯 Modèle importance: {ML_MODELS_CONFIG['importance']}")

        # Collecte
        articles = self.collect_fmp_news(count, days)
        
        if not articles:
            raise RuntimeError("Aucun article FMP collecté")

        # Statistiques
        labels = [article["label"] for article in articles]
        importance_labels = [article["importance"] for article in articles]
        
        label_counts = {label: labels.count(label) for label in set(labels)}
        importance_counts = {label: importance_labels.count(label) for label in set(importance_labels)}
        
        logger.info(f"📊 Distribution sentiment: {label_counts}")
        logger.info(f"🎯 Distribution importance: {importance_counts}")
        logger.info(f"🗄️ Cache: {len(self.seen_articles)} articles connus")
        
        if self.auto_label:
            needs_review_count = sum(1 for article in articles if article.get("needs_review", False))
            high_confidence_count = len(articles) - needs_review_count
            logger.info(f"🎯 Articles haute confiance PRODUIT: {high_confidence_count}/{len(articles)}")

        return self.save_dataset(articles, output_file)


def main():
    parser = argparse.ArgumentParser(description="Smart News Collector with PRODUCTION Dual ML Models")
    
    parser.add_argument("--source", choices=["fmp"], default="fmp", help="Source FMP")
    parser.add_argument("--count", type=int, default=40, help="Nombre d'articles")
    parser.add_argument("--days", type=int, default=7, help="Fenêtre temporelle en jours")
    parser.add_argument("--output", type=Path, help="Fichier de sortie")
    parser.add_argument("--output-dir", default="datasets", help="Répertoire de sortie")
    parser.add_argument("--no-cache", action="store_true", help="Désactiver déduplication")
    
    # Arguments ML
    parser.add_argument("--auto-label", action="store_true", help="Activer double ML labeling PRODUIT")
    parser.add_argument("--confidence-threshold", type=float, default=0.75, 
                       help="Seuil de confiance ML")

    args = parser.parse_args()

    # Vérifier la clé API
    if not os.getenv("FMP_API_KEY"):
        logger.error("❌ FMP_API_KEY environment variable required")
        return 1

    try:
        collector = SmartNewsCollector(
            output_dir=args.output_dir,
            enable_cache=not args.no_cache,
            auto_label=args.auto_label,
            confidence_threshold=args.confidence_threshold
        )

        output_file = collector.collect_and_save(
            count=args.count,
            days=args.days,
            output_file=args.output
        )

        print(f"✅ Dataset Smart PRODUIT généré: {output_file}")
        print(f"🎯 Colonnes: text, label (sentiment), importance")
        
        if args.auto_label:
            print(f"🤖 Double ML PRODUIT:")
            print(f"  😊 Sentiment: {ML_MODELS_CONFIG['sentiment']}")
            print(f"  🎯 Importance: {ML_MODELS_CONFIG['importance']}")
        
        print("\n🚀 Prochaines étapes:")
        print(f"  1. Éditer: open news_editor.html")
        print(f"  2. Commit: réentraîne les modèles avec nouvelles données")

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
