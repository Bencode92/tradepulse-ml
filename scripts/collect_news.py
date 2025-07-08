#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradePulse News Collector - FMP API Edition with ML Labeling
==========================================================

Premium financial news collection using Financial Modeling Prep API
with FinBERT ML labeling for sentiment analysis.

Usage:
    python scripts/collect_news.py --source fmp --count 60 --days 7
    python scripts/collect_news.py --source fmp --count 40 --auto-label --ml-model fallback
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
logger = logging.getLogger("fmp-collector")

# Fuseau horaire Paris
PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")

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

# ML Models configuration - FIXED: Utilise le modèle fixe maintenant
ML_MODELS_CONFIG = {
    "production": "Bencode92/tradepulse-finbert-prod",      # ✅ MODÈLE FIXE!
    "development": "Bencode92/tradepulse-finbert-dev", 
    "fallback": "yiyanghkust/finbert-tone",
}

class FMPNewsCollector:
    def __init__(self, output_dir: str = "datasets", enable_cache: bool = True, 
                 auto_label: bool = False, ml_model: str = "fallback", 
                 confidence_threshold: float = 0.75):
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
        self.ml_model_name = ML_MODELS_CONFIG.get(ml_model, ml_model)
        self.confidence_threshold = confidence_threshold
        self.ml_classifier = None
        
        self._load_cache()
        
        if self.auto_label:
            self._load_ml_model()

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

    def _load_ml_model(self):
        """Charge le modèle ML pour labelling automatique"""
        try:
            from transformers import pipeline
            import torch
            
            logger.info(f"🤖 Chargement du modèle ML: {self.ml_model_name}")
            
            # Add HF token if available for custom models
            model_kwargs = {}
            if "Bencode92/" in self.ml_model_name and os.getenv("HF_TOKEN"):
                model_kwargs["token"] = os.getenv("HF_TOKEN")
                logger.info("🔑 Using HF token for custom model")
            
            self.ml_classifier = pipeline(
                "text-classification",
                model=self.ml_model_name,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1,
                **model_kwargs
            )
            
            logger.info(f"✅ Modèle ML chargé: {self.ml_model_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Échec chargement {self.ml_model_name}: {e}")
            
            # Fallback sur FinBERT de base
            try:
                fallback_model = ML_MODELS_CONFIG["fallback"]
                logger.info(f"🔄 Fallback sur: {fallback_model}")
                
                from transformers import pipeline
                import torch
                
                self.ml_classifier = pipeline(
                    "text-classification",
                    model=fallback_model,
                    return_all_scores=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                
                self.ml_model_name = fallback_model
                logger.info(f"✅ Modèle fallback chargé: {fallback_model}")
                
            except Exception as e2:
                logger.error(f"❌ Impossible de charger le modèle fallback: {e2}")
                self.ml_classifier = None

    def _predict_sentiment_ml(self, text: str) -> Tuple[str, float, bool]:
        """Prédit le sentiment avec le modèle ML"""
        if not self.ml_classifier:
            return self._basic_sentiment_analysis(text), 0.5, True
        
        try:
            text_truncated = text[:512]
            results = self.ml_classifier(text_truncated)
            best_pred = max(results[0], key=lambda x: x['score'])
            
            # Normalisation des labels
            label_mapping = {
                'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral',
                'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral',
                'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive',
            }
            
            raw_label = best_pred['label']
            normalized_label = label_mapping.get(raw_label, 'neutral')
            confidence = best_pred['score']
            needs_review = confidence < self.confidence_threshold
            
            return normalized_label, confidence, needs_review
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur prédiction ML: {e}")
            return self._basic_sentiment_analysis(text), 0.5, True

    def _basic_sentiment_analysis(self, text: str) -> str:
        """Analyse de sentiment basique"""
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
        
        # Tri par qualité/importance
        all_articles.sort(key=lambda x: x.get("importance_score", 0), reverse=True)
        
        return all_articles[:count]

    def _enrich_article(self, article: Dict, source_type: str) -> Optional[Dict]:
        """Enrichit un article avec métadonnées et sentiment"""
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
            
            # Analyse de sentiment
            if self.auto_label:
                label, confidence, needs_review = self._predict_sentiment_ml(enriched["text"])
                enriched.update({
                    "label": label,
                    "ml_confidence": confidence,
                    "needs_review": needs_review,
                    "ml_model_used": self.ml_model_name,
                    "labeling_method": "ml_auto"
                })
            else:
                label = self._basic_sentiment_analysis(enriched["text"])
                enriched.update({
                    "label": label,
                    "ml_confidence": None,
                    "needs_review": False,
                    "ml_model_used": None,
                    "labeling_method": "rule_based"
                })
            
            # Score d'importance
            enriched["importance_score"] = self._calculate_importance(enriched, source_type)
            
            return enriched
            
        except Exception as e:
            logger.warning(f"Erreur enrichissement article: {e}")
            return None

    def _calculate_importance(self, article: Dict, source_type: str) -> float:
        """Calcule un score d'importance pour l'article"""
        content = article.get("text", "").lower()
        title = article.get("title", "").lower()
        
        # Mots-clés d'impact élevé
        high_impact = ["earnings", "fed", "inflation", "recession", "merger", "acquisition"]
        medium_impact = ["revenue", "guidance", "analyst", "upgrade", "downgrade"]
        
        score = 0
        
        # Score basé sur les mots-clés
        for keyword in high_impact:
            if keyword in content:
                score += 15
        
        for keyword in medium_impact:
            if keyword in content:
                score += 8
        
        # Bonus pour source premium
        source = article.get("source", "").lower()
        if any(premium in source for premium in ["bloomberg", "reuters", "wsj"]):
            score += 10
        
        # Bonus pour longueur de contenu
        content_length = len(article.get("text", ""))
        score += min(10, content_length / 200)
        
        # Bonus pour confiance ML élevée
        if article.get("ml_confidence", 0) > 0.8:
            score += 5
        
        return min(100, score)

    def save_dataset(self, articles: List[Dict], output_file: Optional[Path] = None) -> Path:
        """Sauvegarde le dataset avec métadonnées"""
        if output_file is None:
            today = datetime.datetime.now(PARIS_TZ).strftime("%Y%m%d")
            output_file = self.output_dir / f"news_{today}.csv"

        # Sauvegarde CSV simple
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            
            for article in articles:
                writer.writerow([article["text"], article["label"]])

        # Métadonnées JSON
        labels = [article["label"] for article in articles]
        label_counts = {label: labels.count(label) for label in set(labels)}
        needs_review_count = sum(1 for article in articles if article.get("needs_review", False))
        
        metadata = {
            "filename": output_file.name,
            "created_at": datetime.datetime.now(PARIS_TZ).isoformat(),
            "source": "fmp",
            "article_count": len(articles),
            "label_distribution": label_counts,
            "deduplication_enabled": self.enable_cache,
            "cache_size": len(self.seen_articles),
            "auto_labeling_enabled": self.auto_label,
            "ml_model_used": self.ml_model_name if self.auto_label else None,
            "confidence_threshold": self.confidence_threshold if self.auto_label else None,
            "high_confidence_articles": len(articles) - needs_review_count,
            "needs_review_articles": needs_review_count
        }

        json_file = output_file.with_suffix('.json')
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Sauvegarde cache
        self._save_cache()

        logger.info(f"✅ Dataset FMP: {output_file} ({len(articles)} échantillons)")
        return output_file

    def collect_and_save(self, count: int = 40, days: int = 7, output_file: Optional[Path] = None) -> Path:
        """Pipeline complet de collecte FMP"""
        logger.info(f"🚀 Collecte FMP: {count} articles, {days} jours")
        
        if self.auto_label:
            logger.info(f"🤖 ML labeling activé: {self.ml_model_name}")

        # Collecte
        articles = self.collect_fmp_news(count, days)
        
        if not articles:
            raise RuntimeError("Aucun article FMP collecté")

        # Statistiques
        labels = [article["label"] for article in articles]
        label_counts = {label: labels.count(label) for label in set(labels)}
        
        logger.info(f"📊 Distribution: {label_counts}")
        logger.info(f"🗄️ Cache: {len(self.seen_articles)} articles connus")
        
        if self.auto_label:
            needs_review_count = sum(1 for article in articles if article.get("needs_review", False))
            high_confidence_count = len(articles) - needs_review_count
            logger.info(f"🎯 Articles haute confiance: {high_confidence_count}/{len(articles)}")

        return self.save_dataset(articles, output_file)


def main():
    parser = argparse.ArgumentParser(description="FMP News Collector with ML Labeling")
    
    parser.add_argument("--source", choices=["fmp"], default="fmp", help="Source FMP (défaut)")
    parser.add_argument("--count", type=int, default=40, help="Nombre d'articles")
    parser.add_argument("--days", type=int, default=7, help="Fenêtre temporelle en jours")
    parser.add_argument("--output", type=Path, help="Fichier de sortie")
    parser.add_argument("--output-dir", default="datasets", help="Répertoire de sortie")
    parser.add_argument("--no-cache", action="store_true", help="Désactiver déduplication")
    
    # Arguments ML
    parser.add_argument("--auto-label", action="store_true", help="Activer ML labeling")
    parser.add_argument("--ml-model", choices=["production", "development", "fallback"], 
                       default="fallback", help="Modèle ML")
    parser.add_argument("--confidence-threshold", type=float, default=0.75, 
                       help="Seuil de confiance ML")

    args = parser.parse_args()

    # Vérifier la clé API
    if not os.getenv("FMP_API_KEY"):
        logger.error("❌ FMP_API_KEY environment variable required")
        return 1

    try:
        collector = FMPNewsCollector(
            output_dir=args.output_dir,
            enable_cache=not args.no_cache,
            auto_label=args.auto_label,
            ml_model=args.ml_model,
            confidence_threshold=args.confidence_threshold
        )

        output_file = collector.collect_and_save(
            count=args.count,
            days=args.days,
            output_file=args.output
        )

        print(f"✅ Dataset FMP généré: {output_file}")
        print(f"🔄 Déduplication: {'activée' if not args.no_cache else 'désactivée'}")
        
        if args.auto_label:
            print(f"🤖 ML Labeling: {args.ml_model}")
        
        print("\n🚀 Prochaines étapes:")
        print(f"  1. Valider: python scripts/validate_dataset.py")
        print(f"  2. Pipeline: python unified_pipeline.py")

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
