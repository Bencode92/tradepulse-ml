#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradePulse News Collector - Version Avancée avec Auto-Labelling ML
================================================================

🚀 NOUVEAU : Labelling automatique avec votre modèle ML !

Version améliorée qui résout le problème "toujours les mêmes articles" 
ET ajoute le labelling automatique :
- Fenêtre temporelle élargie (1-7 jours)
- Cache de déduplication automatique
- Sources RSS multiples avec rotation
- Pagination NewsAPI intelligente
- Mode mixte optimisé (70% RSS + 30% NewsAPI)
- 🤖 NOUVEAU : Labelling ML automatique avec confiance

Usage:
    # Collecte classique (comme avant)
    python scripts/collect_news.py --source mixed --count 60 --days 3
    
    # 🚀 NOUVEAU : Collecte + Labelling automatique
    python scripts/collect_news.py --source mixed --count 50 --auto-label --ml-model production
    python scripts/collect_news.py --source rss --count 40 --auto-label --ml-model Bencode92/tradepulse-finbert-prod
    python scripts/collect_news.py --source mixed --count 30 --auto-label --confidence-threshold 0.8

Nouveautés:
- --auto-label : Active le labelling ML automatique
- --ml-model : Modèle à utiliser (production/development/custom)
- --confidence-threshold : Seuil de confiance (défaut: 0.75)
- --review-low-confidence : Marque les articles à faible confiance pour révision
"""

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import random
import zoneinfo
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configuration des logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("news-collector-advanced")

# Fuseau horaire Paris
PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")

# 🚀 NOUVEAU : Configuration des modèles ML
ML_MODELS_CONFIG = {
    "production": "Bencode92/tradepulse-finbert-prod",      # Modèle stable de production
    "development": "Bencode92/tradepulse-finbert-dev",      # Modèle de développement
    "fallback": "yiyanghkust/finbert-tone",                 # FinBERT de base
}

class AdvancedNewsCollector:
    def __init__(self, output_dir: str = "datasets", enable_cache: bool = True, auto_label: bool = False, ml_model: str = "fallback", confidence_threshold: float = 0.75):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_cache = enable_cache
        self.cache_file = self.output_dir / ".article_cache.json"
        self.seen_articles: Set[str] = set()
        
        # 🚀 NOUVEAU : Configuration ML
        self.auto_label = auto_label
        self.ml_model_name = ML_MODELS_CONFIG.get(ml_model, ml_model)  # Résolution des alias
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

    # 🚀 NOUVEAU : Chargement du modèle ML
    def _load_ml_model(self):
        """Charge le modèle ML pour labelling automatique"""
        try:
            from transformers import pipeline
            import torch
            
            logger.info(f"🤖 Chargement du modèle ML: {self.ml_model_name}")
            
            # Chargement avec gestion des erreurs
            self.ml_classifier = pipeline(
                "text-classification",
                model=self.ml_model_name,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
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
                logger.warning("🔄 Utilisation de l'analyse de sentiment basique")
                self.ml_classifier = None

    # 🚀 NOUVEAU : Prédiction ML avec confiance
    def _predict_sentiment_ml(self, text: str) -> Tuple[str, float, bool]:
        """
        Prédit le sentiment avec le modèle ML
        Returns: (label, confidence, needs_review)
        """
        if not self.ml_classifier:
            # Fallback sur l'analyse basique
            label = self._advanced_sentiment_analysis(text)
            return label, 0.5, True  # Confiance faible = révision recommandée
        
        try:
            # Limiter la longueur du texte
            text_truncated = text[:512]
            
            # Prédiction avec tous les scores
            results = self.ml_classifier(text_truncated)
            
            # Trouver la prédiction avec le score le plus élevé
            best_pred = max(results[0], key=lambda x: x['score'])
            
            # Normalisation des labels (selon le modèle utilisé)
            label_mapping = {
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative', 
                'NEUTRAL': 'neutral',
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'LABEL_0': 'negative',  # FinBERT mapping
                'LABEL_1': 'neutral',
                'LABEL_2': 'positive',
            }
            
            raw_label = best_pred['label']
            normalized_label = label_mapping.get(raw_label, 'neutral')
            confidence = best_pred['score']
            needs_review = confidence < self.confidence_threshold
            
            return normalized_label, confidence, needs_review
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur prédiction ML: {e}")
            # Fallback sur analyse basique
            label = self._advanced_sentiment_analysis(text)
            return label, 0.5, True

    def _article_hash(self, text: str) -> str:
        """Génère un hash unique pour un article - AMÉLIORÉ"""
        # Normaliser le texte pour la déduplication
        normalized = text.lower().strip()
        # Retirer "Sample X:" pour éviter doublons artificiels
        if normalized.startswith("sample ") and ":" in normalized:
            normalized = normalized.split(":", 1)[1].strip()
        # Retirer la ponctuation pour une meilleure détection des doublons
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

    def get_date_range(self, days_back: int = 3) -> Tuple[str, str]:
        """Génère une plage de dates dynamique"""
        end_date = datetime.datetime.now(PARIS_TZ).date()
        start_date = end_date - datetime.timedelta(days=days_back)
        return start_date.isoformat(), end_date.isoformat()

    def collect_from_rss_extended(self, count: int = 40, days: int = 3) -> List[Dict]:
        """Collecte RSS avec sources étendues et fenêtre temporelle - AMÉLIORÉ"""
        try:
            import feedparser
        except ImportError:
            logger.error("❌ feedparser non installé. pip install feedparser")
            # Ne pas utiliser placeholder en cas d'erreur, retourner vide
            return []

        # Sources RSS étendues (12 au lieu de 4)
        rss_feeds = [
            # Primary sources
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html", 
            "https://feeds.reuters.com/reuters/businessNews",
            "https://rss.cnn.com/rss/money_latest.rss",
            # Secondary sources
            "https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://www.sec.gov/news/pressreleases.rss",
            "https://home.treasury.gov/rss/press-releases",
            "https://www.cnbc.com/id/100727362/device/rss/rss.html",
            # Alternative sources  
            "https://feeds.businesswire.com/news/home/20120928006494/en",
            "https://finance.yahoo.com/news/rssindex",
            "https://seekingalpha.com/feed.xml",
            "https://feeds.efinancialcareers.com/news"
        ]

        start_date, end_date = self.get_date_range(days)
        start_dt = datetime.datetime.fromisoformat(start_date).replace(tzinfo=PARIS_TZ)
        end_dt = datetime.datetime.fromisoformat(end_date).replace(tzinfo=PARIS_TZ)
        
        logger.info(f"🔍 Collecte RSS: {start_date} à {end_date} ({days} jours)")

        all_articles = []
        sources_used = {}

        # Rotation aléatoire des sources pour diversité
        random.shuffle(rss_feeds)

        for feed_url in rss_feeds:
            try:
                logger.info(f"📰 Source: {feed_url}")
                feed = feedparser.parse(feed_url)
                feed_articles = 0

                for entry in feed.entries:
                    # Parse de la date de publication
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime.datetime(*entry.published_parsed[:6]).replace(tzinfo=PARIS_TZ)
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime.datetime(*entry.updated_parsed[:6]).replace(tzinfo=PARIS_TZ)
                    
                    # Filtrer par date (fenêtre de X jours au lieu de aujourd'hui seulement)
                    if pub_date and (pub_date < start_dt or pub_date > end_dt):
                        continue

                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title}. {summary}".strip()

                    # Vérification qualité et déduplication AMÉLIORÉE
                    if len(text) > 50 and not self._is_duplicate(text):
                        # 🚀 NOUVEAU : Structure enrichie avec métadonnées
                        article_data = {
                            "text": text,
                            "title": title,
                            "summary": summary,
                            "source": feed_url,
                            "url": entry.get("link", ""),
                            "published_date": pub_date.isoformat() if pub_date else None,
                            "collected_at": datetime.datetime.now(PARIS_TZ).isoformat()
                        }
                        
                        all_articles.append(article_data)
                        self._add_to_cache(text)
                        feed_articles += 1

                    # Limiter par source pour équilibrage
                    if feed_articles >= count // 4:  # Max 1/4 des articles par source
                        break

                if feed_articles > 0:
                    domain = feed_url.split('/')[2] if '//' in feed_url else feed_url
                    sources_used[domain] = feed_articles

                # Arrêter si on a assez d'articles
                if len(all_articles) >= count * 1.5:
                    break

            except Exception as e:
                logger.warning(f"Erreur RSS {feed_url}: {e}")

        if sources_used:
            logger.info(f"ℹ️ Sources utilisées: {dict(list(sources_used.items())[:5])}")

        # CHANGEMENT : Ne pas utiliser placeholder automatiquement
        if not all_articles:
            logger.warning("❌ Aucun article RSS collecté")
            return []

        # Sélection équilibrée et labellisation
        selected = all_articles[:count]
        return self._label_articles(selected)

    def collect_from_newsapi_paginated(self, count: int = 30, days: int = 3, api_key: Optional[str] = None) -> List[Dict]:
        """NewsAPI avec pagination et fenêtre temporelle - AMÉLIORÉ"""
        if not api_key:
            api_key = os.getenv("NEWSAPI_KEY")

        if not api_key:
            logger.warning("⚠️ Clé NewsAPI manquante")
            return []

        try:
            import requests
        except ImportError:
            logger.error("❌ requests non installé. pip install requests")
            return []

        start_date, end_date = self.get_date_range(days)
        logger.info(f"🔍 Collecte NewsAPI: {start_date} à {end_date} ({days} jours)")

        all_articles = []
        
        # Mots-clés variés pour diversification
        query_sets = [
            "stock market OR earnings OR revenue",
            "Federal Reserve OR interest rates OR inflation", 
            "cryptocurrency OR bitcoin OR blockchain",
            "merger OR acquisition OR IPO",
            "oil prices OR commodities OR energy"
        ]

        for query in query_sets:
            for page in range(1, 4):  # 3 pages par requête
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 20,  # Max par page
                    "page": page,
                    "from": start_date,
                    "to": end_date,
                    "apiKey": api_key,
                }

                try:
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()

                    articles = data.get("articles", [])
                    if not articles:
                        break  # Pas d'articles sur cette page

                    for article in articles:
                        title = article.get("title", "")
                        description = article.get("description", "")
                        text = f"{title}. {description}".strip() if description else title

                        if len(text) > 50 and not self._is_duplicate(text):
                            # 🚀 NOUVEAU : Structure enrichie avec métadonnées NewsAPI
                            article_data = {
                                "text": text,
                                "title": title,
                                "summary": description,
                                "source": "newsapi.org",
                                "url": article.get("url", ""),
                                "published_date": article.get("publishedAt"),
                                "collected_at": datetime.datetime.now(PARIS_TZ).isoformat(),
                                "newsapi_source": article.get("source", {}).get("name", "Unknown")
                            }
                            
                            all_articles.append(article_data)
                            self._add_to_cache(text)

                        # Arrêter si on a assez
                        if len(all_articles) >= count:
                            break

                    if len(all_articles) >= count:
                        break

                except Exception as e:
                    logger.warning(f"Erreur NewsAPI page {page}: {e}")
                    break

            if len(all_articles) >= count:
                break

        # Labellisation
        selected = all_articles[:count]
        return self._label_articles(selected)

    def collect_mixed_sources(self, count: int = 50, days: int = 3, api_key: Optional[str] = None) -> List[Dict]:
        """Mode mixte optimisé: 70% RSS + 30% NewsAPI - AMÉLIORÉ"""
        rss_count = int(count * 0.7)  # 70% RSS
        newsapi_count = count - rss_count  # 30% NewsAPI

        logger.info(f"🔄 Mode mixte: {rss_count} RSS + {newsapi_count} NewsAPI")

        all_articles = []
        
        # Collecte RSS (prioritaire pour diversité)
        rss_articles = self.collect_from_rss_extended(rss_count, days)
        all_articles.extend(rss_articles)

        # Collecte NewsAPI (complément) - seulement si clé disponible
        if newsapi_count > 0 and api_key:
            newsapi_articles = self.collect_from_newsapi_paginated(newsapi_count, days, api_key)
            all_articles.extend(newsapi_articles)
        elif newsapi_count > 0:
            logger.warning("🔑 Clé NewsAPI manquante, utilisation RSS seulement")

        # Mélanger pour diversité
        random.shuffle(all_articles)
        
        # CHANGEMENT : Utiliser placeholder SEULEMENT si vraiment aucun article
        if not all_articles:
            logger.warning("❌ Aucun article de sources réelles - utilisation placeholder")
            return self.get_placeholder_samples(count)
        
        logger.info(f"✅ Mode mixte: {len(all_articles)} articles uniques collectés")
        return all_articles[:count]

    # 🚀 NOUVEAU : Labellisation des articles (ML ou basique)
    def _label_articles(self, articles: List[Dict]) -> List[Dict]:
        """Labellise une liste d'articles avec ML ou analyse basique"""
        labeled_articles = []
        stats = {"positive": 0, "negative": 0, "neutral": 0, "needs_review": 0}
        
        logger.info(f"🤖 Labellisation de {len(articles)} articles...")
        if self.auto_label and self.ml_classifier:
            logger.info(f"   Modèle: {self.ml_model_name}")
            logger.info(f"   Seuil confiance: {self.confidence_threshold}")
        
        for i, article in enumerate(articles):
            text = article["text"]
            
            if self.auto_label:
                # Prédiction ML avec confiance
                label, confidence, needs_review = self._predict_sentiment_ml(text)
                
                # Ajouter les métadonnées ML
                article.update({
                    "label": label,
                    "ml_confidence": confidence,
                    "needs_review": needs_review,
                    "ml_model_used": self.ml_model_name,
                    "labeling_method": "ml_auto"
                })
                
                if needs_review:
                    stats["needs_review"] += 1
            else:
                # Analyse de sentiment basique
                label = self._advanced_sentiment_analysis(text)
                article.update({
                    "label": label,
                    "ml_confidence": None,
                    "needs_review": False,
                    "ml_model_used": None,
                    "labeling_method": "rule_based"
                })
            
            stats[label] += 1
            labeled_articles.append(article)
            
            # Log de progression
            if (i + 1) % 10 == 0:
                logger.info(f"   📊 Progression: {i + 1}/{len(articles)} labellisés")
        
        logger.info(f"✅ Labellisation terminée:")
        logger.info(f"   📊 Distribution: {stats}")
        
        if self.auto_label and stats["needs_review"] > 0:
            logger.info(f"   ⚠️ Articles à réviser (confiance < {self.confidence_threshold}): {stats['needs_review']}")
        
        return labeled_articles

    def _advanced_sentiment_analysis(self, text: str) -> str:
        """Analyse de sentiment pondérée et nuancée (méthode existante conservée)"""
        text_lower = text.lower()

        # Mots-clés avec poids (impact plus réaliste)
        high_impact_positive = ["surge", "rally", "breakthrough", "record", "beat expectations", "strong growth"]
        medium_impact_positive = ["gain", "rise", "increase", "boost", "positive", "strong"]
        low_impact_positive = ["up", "higher", "improved", "better"]

        high_impact_negative = ["crash", "plunge", "crisis", "recession", "collapse", "massive loss"]
        medium_impact_negative = ["drop", "fall", "decline", "weak", "concern", "risk"]
        low_impact_negative = ["down", "lower", "pressure", "challenge"]

        # Calcul du score pondéré
        positive_score = (
            sum(3 for kw in high_impact_positive if kw in text_lower) +
            sum(2 for kw in medium_impact_positive if kw in text_lower) +
            sum(1 for kw in low_impact_positive if kw in text_lower)
        )
        
        negative_score = (
            sum(3 for kw in high_impact_negative if kw in text_lower) +
            sum(2 for kw in medium_impact_negative if kw in text_lower) +
            sum(1 for kw in low_impact_negative if kw in text_lower)
        )

        # Logique de décision améliorée
        if positive_score > negative_score + 1:  # Seuil pour éviter trop de positifs
            return "positive"
        elif negative_score > positive_score + 1:
            return "negative"
        else:
            return "neutral"

    def get_placeholder_samples(self, count: int = 20) -> List[Dict]:
        """Échantillons placeholder pour tests - AMÉLIORÉ pour éviter doublons"""
        base_samples = [
            {
                "text": "Apple Inc. reported record quarterly earnings beating analyst expectations with strong iPhone sales and robust services revenue growth.",
                "title": "Apple Reports Record Earnings",
                "summary": "Strong quarterly performance beats expectations",
                "source": "placeholder",
                "url": "https://example.com/apple-earnings",
                "published_date": datetime.datetime.now(PARIS_TZ).isoformat(),
                "collected_at": datetime.datetime.now(PARIS_TZ).isoformat()
            },
            {
                "text": "Federal Reserve announced unexpected interest rate hike of 75 basis points citing persistent inflation concerns and tight labor markets.",
                "title": "Fed Raises Interest Rates",
                "summary": "Unexpected 75bp hike amid inflation concerns",
                "source": "placeholder",
                "url": "https://example.com/fed-rate-hike",
                "published_date": datetime.datetime.now(PARIS_TZ).isoformat(),
                "collected_at": datetime.datetime.now(PARIS_TZ).isoformat()
            },
            {
                "text": "The S&P 500 index closed unchanged at 4150 points with mixed sector performance as investors awaited key economic data releases.",
                "title": "S&P 500 Unchanged",
                "summary": "Mixed sector performance awaiting data",
                "source": "placeholder",
                "url": "https://example.com/sp500-flat",
                "published_date": datetime.datetime.now(PARIS_TZ).isoformat(),
                "collected_at": datetime.datetime.now(PARIS_TZ).isoformat()
            },
            {
                "text": "Tesla stock surged 12% in after-hours trading following better-than-expected delivery numbers and strong guidance for next quarter.",
                "title": "Tesla Stock Surges",
                "summary": "Strong delivery numbers drive stock higher",
                "source": "placeholder",
                "url": "https://example.com/tesla-surge",
                "published_date": datetime.datetime.now(PARIS_TZ).isoformat(),
                "collected_at": datetime.datetime.now(PARIS_TZ).isoformat()
            },
            {
                "text": "Oil prices dropped to $72 per barrel amid concerns about global economic slowdown and increased supply from OPEC+ members.",
                "title": "Oil Prices Decline",
                "summary": "Economic concerns drive prices lower",
                "source": "placeholder",
                "url": "https://example.com/oil-decline",
                "published_date": datetime.datetime.now(PARIS_TZ).isoformat(),
                "collected_at": datetime.datetime.now(PARIS_TZ).isoformat()
            }
        ]
        
        # CHANGEMENT : Créer de vraies variations au lieu de doublons avec "Sample X:"
        placeholder_data = []
        for i in range(count):
            base_idx = i % len(base_samples)
            sample = base_samples[base_idx].copy()
            
            # Variations légères pour éviter doublons identiques
            if i >= len(base_samples):
                variations = [
                    ("reported", "announced"),
                    ("strong", "robust"), 
                    ("increased", "rose"),
                    ("expectations", "forecasts"),
                    ("performance", "results")
                ]
                
                for old, new in variations:
                    if old in sample["text"]:
                        sample["text"] = sample["text"].replace(old, new, 1)
                        break
                        
                sample["title"] = f"Update: {sample['title']}"
            
            placeholder_data.append(sample)
        
        return self._label_articles(placeholder_data)

    def save_dataset_with_metadata(self, articles: List[Dict], output_file: Optional[Path] = None, metadata: Dict = None) -> Path:
        """Sauvegarde avec métadonnées enrichies - SIMPLIFIÉ CSV"""
        if output_file is None:
            today = datetime.datetime.now(PARIS_TZ).strftime("%Y%m%d")
            output_file = self.output_dir / f"news_{today}.csv"

        # SIMPLIFIÉ : Toujours CSV minimal (text, label)
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            
            for article in articles:
                writer.writerow([article["text"], article["label"]])

        # Sauvegarde métadonnées JSON complètes (optionnel)
        if metadata:
            json_file = output_file.with_suffix('.json')
            
            # 🚀 NOUVEAU : Métadonnées enrichies avec infos ML
            enhanced_metadata = metadata.copy()
            enhanced_metadata.update({
                "auto_labeling_enabled": self.auto_label,
                "ml_model_used": self.ml_model_name if self.auto_label else None,
                "confidence_threshold": self.confidence_threshold if self.auto_label else None,
                "articles_metadata": articles  # Données complètes pour debug
            })
            
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"📁 Métadonnées: {json_file}")

        # Sauvegarde cache
        self._save_cache()

        logger.info(f"✅ Dataset: {output_file} ({len(articles)} échantillons)")
        return output_file

    def collect_and_save(self, source: str = "mixed", count: int = 40, days: int = 3, output_file: Optional[Path] = None, **kwargs) -> Path:
        """Collecte avancée avec déduplication et métadonnées"""
        logger.info(f"🔄 Collecte avancée: {source}, {count} articles, {days} jours")
        
        if self.auto_label:
            logger.info(f"🤖 Labelling automatique activé (modèle: {self.ml_model_name})")

        if source == "placeholder":
            articles = self.get_placeholder_samples(count)
        elif source == "rss":
            articles = self.collect_from_rss_extended(count, days)
            if not articles:  # Fallback si RSS échoue
                logger.warning("🔄 RSS échoué, fallback sur placeholder")
                articles = self.get_placeholder_samples(count)
        elif source == "newsapi":
            articles = self.collect_from_newsapi_paginated(count, days, kwargs.get("api_key"))
            if not articles:  # Fallback si NewsAPI échoue
                logger.warning("🔄 NewsAPI échoué, fallback sur placeholder")
                articles = self.get_placeholder_samples(count)
        elif source == "mixed":
            articles = self.collect_mixed_sources(count, days, kwargs.get("api_key"))
        else:
            raise ValueError(f"Source non supportée: {source}")

        if not articles:
            raise RuntimeError("Aucun échantillon collecté")

        # Statistiques
        labels = [article["label"] for article in articles]
        label_counts = {label: labels.count(label) for label in set(labels)}
        
        # 🚀 NOUVEAU : Statistiques ML
        needs_review_count = sum(1 for article in articles if article.get("needs_review", False))
        high_confidence_count = len(articles) - needs_review_count
        
        # Métadonnées enrichies
        metadata = {
            "filename": output_file.name if output_file else f"news_{datetime.datetime.now(PARIS_TZ).strftime('%Y%m%d')}.csv",
            "created_at": datetime.datetime.now(PARIS_TZ).isoformat(),
            "source": source,
            "article_count": len(articles),
            "days_range": days,
            "label_distribution": label_counts,
            "deduplication_enabled": self.enable_cache,
            "cache_size": len(self.seen_articles) if self.enable_cache else 0,
            # 🚀 NOUVEAU : Métadonnées ML
            "auto_labeling_enabled": self.auto_label,
            "ml_model_used": self.ml_model_name if self.auto_label else None,
            "confidence_threshold": self.confidence_threshold if self.auto_label else None,
            "high_confidence_articles": high_confidence_count,
            "needs_review_articles": needs_review_count
        }

        logger.info(f"📊 Distribution: {label_counts}")
        if self.enable_cache:
            logger.info(f"🗄️ Cache: {len(self.seen_articles)} articles connus (doublons évités)")
        
        if self.auto_label:
            logger.info(f"🎯 Articles haute confiance: {high_confidence_count}/{len(articles)}")
            if needs_review_count > 0:
                logger.info(f"⚠️ Articles à réviser: {needs_review_count}")

        return self.save_dataset_with_metadata(articles, output_file, metadata)


def main():
    parser = argparse.ArgumentParser(description="TradePulse News Collector - Version Avancée avec Auto-Labelling ML")
    
    # Arguments existants (conservés)
    parser.add_argument("--source", choices=["placeholder", "rss", "newsapi", "mixed"], default="mixed", help="Source (défaut: mixed)")
    parser.add_argument("--count", type=int, default=40, help="Nombre d'articles (défaut: 40)")
    parser.add_argument("--days", type=int, default=3, help="Fenêtre temporelle en jours (défaut: 3)")
    parser.add_argument("--output", type=Path, help="Fichier de sortie")
    parser.add_argument("--newsapi-key", help="Clé NewsAPI")
    parser.add_argument("--output-dir", default="datasets", help="Répertoire de sortie")
    parser.add_argument("--no-cache", action="store_true", help="Désactiver déduplication")
    parser.add_argument("--seed", type=int, help="Seed pour reproductibilité")
    
    # 🚀 NOUVEAUX arguments pour ML
    parser.add_argument("--auto-label", action="store_true", help="Activer le labelling ML automatique")
    parser.add_argument("--ml-model", choices=["production", "development", "fallback"], default="fallback",
                       help="Modèle ML à utiliser (défaut: fallback)")
    parser.add_argument("--custom-model", type=str, help="Modèle personnalisé (format HuggingFace)")
    parser.add_argument("--confidence-threshold", type=float, default=0.75, 
                       help="Seuil de confiance pour ML (défaut: 0.75)")
    parser.add_argument("--review-low-confidence", action="store_true",
                       help="Marquer les articles à faible confiance pour révision")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"🎲 Seed: {args.seed}")

    # NOUVEAU : Activer auto-label par défaut pour mixed/rss
    if not args.auto_label and args.source in ["mixed", "rss"]:
        logger.info("🤖 Auto-labelling ML activé par défaut pour source réelle")
        args.auto_label = True

    # Résolution du modèle ML
    ml_model = args.custom_model if args.custom_model else args.ml_model
    
    collector = AdvancedNewsCollector(
        output_dir=args.output_dir, 
        enable_cache=not args.no_cache,
        auto_label=args.auto_label,
        ml_model=ml_model,
        confidence_threshold=args.confidence_threshold
    )

    try:
        output_file = collector.collect_and_save(
            source=args.source,
            count=args.count,
            days=args.days,
            output_file=args.output,
            api_key=args.newsapi_key,
        )

        print(f"✅ Dataset généré: {output_file}")
        print(f"🔄 Déduplication: {'activée' if not args.no_cache else 'désactivée'}")
        
        if args.auto_label:
            print(f"🤖 Labelling ML: {ml_model} (seuil: {args.confidence_threshold})")
        
        print("\n🚀 Prochaines étapes:")
        if args.auto_label:
            print(f"  1. Vérifier: open news_editor.html (réviser les articles marqués)")
            print(f"  2. Valider: python scripts/validate_dataset.py")
            print(f"  3. Pipeline: python scripts/finetune.py --incremental --dataset {output_file}")
        else:
            print(f"  1. Valider: python scripts/validate_dataset.py")
            print(f"  2. Pipeline: python unified_pipeline.py")

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
