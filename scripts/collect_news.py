#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradePulse News Collector - Version Avancée avec Déduplication
==============================================================

Version améliorée qui résout le problème "toujours les mêmes articles" :
- Fenêtre temporelle élargie (1-7 jours)
- Cache de déduplication automatique
- Sources RSS multiples avec rotation
- Pagination NewsAPI intelligente
- Mode mixte optimisé (70% RSS + 30% NewsAPI)

Usage:
    python scripts/collect_news.py --source mixed --count 60 --days 3
    python scripts/collect_news.py --source rss --count 50 --days 5
    python scripts/collect_news.py --source newsapi --count 40 --days 2 --newsapi-key YOUR_KEY

Nouveautés:
- --days : Fenêtre temporelle en jours (défaut: 3)
- --no-cache : Désactiver la déduplication
- mode 'mixed' : Sources combinées intelligemment
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

class AdvancedNewsCollector:
    def __init__(self, output_dir: str = "datasets", enable_cache: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_cache = enable_cache
        self.cache_file = self.output_dir / ".article_cache.json"
        self.seen_articles: Set[str] = set()
        self._load_cache()

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

    def _article_hash(self, text: str) -> str:
        """Génère un hash unique pour un article"""
        # Normaliser le texte pour la déduplication
        normalized = text.lower().strip()
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

    def collect_from_rss_extended(self, count: int = 40, days: int = 3) -> List[Tuple[str, str]]:
        """Collecte RSS avec sources étendues et fenêtre temporelle"""
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser non installé. pip install feedparser")
            return self.get_placeholder_samples(count)

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

                    # Vérification qualité et déduplication
                    if len(text) > 50 and not self._is_duplicate(text):
                        all_articles.append(text)
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

        if not all_articles:
            logger.warning("Aucun article RSS collecté, utilisation placeholder")
            return self.get_placeholder_samples(count)

        # Sélection équilibrée et labellisation
        selected = all_articles[:count]
        labeled_samples = [(text, self._advanced_sentiment_analysis(text)) for text in selected]
        
        logger.info(f"✅ {len(labeled_samples)} articles RSS uniques collectés")
        return labeled_samples

    def collect_from_newsapi_paginated(self, count: int = 30, days: int = 3, api_key: Optional[str] = None) -> List[Tuple[str, str]]:
        """NewsAPI avec pagination et fenêtre temporelle"""
        if not api_key:
            api_key = os.getenv("NEWSAPI_KEY")

        if not api_key:
            logger.warning("Clé NewsAPI manquante, utilisation placeholder")
            return self.get_placeholder_samples(count)

        try:
            import requests
        except ImportError:
            logger.warning("requests non installé. pip install requests")
            return self.get_placeholder_samples(count)

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
                            all_articles.append(text)
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
        labeled_samples = [(text, self._advanced_sentiment_analysis(text)) for text in selected]
        
        logger.info(f"✅ {len(labeled_samples)} articles NewsAPI uniques collectés")
        return labeled_samples

    def collect_mixed_sources(self, count: int = 50, days: int = 3, api_key: Optional[str] = None) -> List[Tuple[str, str]]:
        """Mode mixte optimisé: 70% RSS + 30% NewsAPI"""
        rss_count = int(count * 0.7)  # 70% RSS
        newsapi_count = count - rss_count  # 30% NewsAPI

        logger.info(f"🔄 Mode mixte: {rss_count} RSS + {newsapi_count} NewsAPI")

        all_samples = []
        
        # Collecte RSS (prioritaire pour diversité)
        rss_samples = self.collect_from_rss_extended(rss_count, days)
        all_samples.extend(rss_samples)

        # Collecte NewsAPI (complément)
        if newsapi_count > 0:
            newsapi_samples = self.collect_from_newsapi_paginated(newsapi_count, days, api_key)
            all_samples.extend(newsapi_samples)

        # Mélanger pour diversité
        random.shuffle(all_samples)
        
        logger.info(f"✅ Mode mixte: {len(all_samples)} articles uniques collectés")
        return all_samples[:count]

    def _advanced_sentiment_analysis(self, text: str) -> str:
        """Analyse de sentiment pondérée et nuancée"""
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

    def get_placeholder_samples(self, count: int = 20) -> List[Tuple[str, str]]:
        """Échantillons placeholder pour tests (conservé de l'original)"""
        positive_samples = [
            ("Apple Inc. reported record quarterly earnings beating analyst expectations with strong iPhone sales and robust services revenue growth.", "positive"),
            ("Tesla stock surged after announcing breakthrough in battery technology that could extend vehicle range by 40% while reducing costs.", "positive"),
            ("Microsoft Azure cloud services revenue grew 35% year-over-year driven by increased enterprise digital transformation investments.", "positive"),
            ("Amazon Web Services announced expansion into three new regions boosting global infrastructure and attracting major enterprise clients.", "positive"),
            ("NVIDIA shares rallied on strong AI chip demand with data center revenue jumping 200% as artificial intelligence adoption accelerates.", "positive"),
        ]
        
        negative_samples = [
            ("Federal Reserve announced unexpected interest rate hike of 75 basis points citing persistent inflation concerns and tight labor markets.", "negative"),
            ("Meta Platforms stock declined following announcement of additional workforce reductions affecting 15000 employees across multiple divisions.", "negative"),
            ("Cryptocurrency markets experienced severe volatility with Bitcoin dropping 18% following regulatory crackdowns in major Asian economies.", "negative"),
            ("Oil prices fell 12% amid concerns about global demand slowdown and unexpected inventory build-up in major consuming nations.", "negative"),
        ]
        
        neutral_samples = [
            ("The S&P 500 index closed unchanged at 4150 points with mixed sector performance as investors awaited key economic data releases.", "neutral"),
            ("European Central Bank maintained benchmark interest rate at 4.0% in line with market expectations during monthly policy meeting.", "neutral"),
            ("Oil prices traded sideways around $75 per barrel following OPEC+ production announcement and mixed economic signals from major economies.", "neutral"),
            ("Treasury yields remained stable ahead of upcoming inflation data release with investors positioning for potential market volatility.", "neutral"),
        ]

        # Distribution équilibrée
        samples_per_class = count // 3
        remaining = count % 3

        selected = []
        selected.extend(positive_samples[:samples_per_class])
        selected.extend(negative_samples[:samples_per_class])
        selected.extend(neutral_samples[:samples_per_class + remaining])

        random.shuffle(selected)
        return selected[:count]

    def save_dataset_with_metadata(self, samples: List[Tuple[str, str]], output_file: Optional[Path] = None, metadata: Dict = None) -> Path:
        """Sauvegarde avec métadonnées enrichies"""
        if output_file is None:
            today = datetime.datetime.now(PARIS_TZ).strftime("%Y%m%d")
            output_file = self.output_dir / f"news_{today}.csv"

        # Sauvegarde CSV
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerows(samples)

        # Sauvegarde métadonnées JSON
        if metadata:
            json_file = output_file.with_suffix('.json')
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"📁 Métadonnées: {json_file}")

        # Sauvegarde cache
        self._save_cache()

        logger.info(f"✅ Dataset: {output_file} ({len(samples)} échantillons)")
        return output_file

    def collect_and_save(self, source: str = "mixed", count: int = 40, days: int = 3, output_file: Optional[Path] = None, **kwargs) -> Path:
        """Collecte avancée avec déduplication et métadonnées"""
        logger.info(f"🔄 Collecte avancée: {source}, {count} articles, {days} jours")

        if source == "placeholder":
            samples = self.get_placeholder_samples(count)
        elif source == "rss":
            samples = self.collect_from_rss_extended(count, days)
        elif source == "newsapi":
            samples = self.collect_from_newsapi_paginated(count, days, kwargs.get("api_key"))
        elif source == "mixed":
            samples = self.collect_mixed_sources(count, days, kwargs.get("api_key"))
        else:
            raise ValueError(f"Source non supportée: {source}")

        if not samples:
            raise RuntimeError("Aucun échantillon collecté")

        # Statistiques
        labels = [label for _, label in samples]
        label_counts = {label: labels.count(label) for label in set(labels)}
        
        # Métadonnées enrichies
        metadata = {
            "filename": output_file.name if output_file else f"news_{datetime.datetime.now(PARIS_TZ).strftime('%Y%m%d')}.csv",
            "created_at": datetime.datetime.now(PARIS_TZ).isoformat(),
            "source": source,
            "article_count": len(samples),
            "days_range": days,
            "label_distribution": label_counts,
            "deduplication_enabled": self.enable_cache,
            "cache_size": len(self.seen_articles) if self.enable_cache else 0
        }

        logger.info(f"📊 Distribution: {label_counts}")
        if self.enable_cache:
            logger.info(f"🗄️ Cache: {len(self.seen_articles)} articles connus (doublons évités)")

        return self.save_dataset_with_metadata(samples, output_file, metadata)


def main():
    parser = argparse.ArgumentParser(description="TradePulse News Collector - Version Avancée")
    parser.add_argument("--source", choices=["placeholder", "rss", "newsapi", "mixed"], default="mixed", help="Source (défaut: mixed)")
    parser.add_argument("--count", type=int, default=40, help="Nombre d'articles (défaut: 40)")
    parser.add_argument("--days", type=int, default=3, help="Fenêtre temporelle en jours (défaut: 3)")
    parser.add_argument("--output", type=Path, help="Fichier de sortie")
    parser.add_argument("--newsapi-key", help="Clé NewsAPI")
    parser.add_argument("--output-dir", default="datasets", help="Répertoire de sortie")
    parser.add_argument("--no-cache", action="store_true", help="Désactiver déduplication")
    parser.add_argument("--seed", type=int, help="Seed pour reproductibilité")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"🎲 Seed: {args.seed}")

    collector = AdvancedNewsCollector(args.output_dir, enable_cache=not args.no_cache)

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
        print("\n🚀 Prochaines étapes:")
        print(f"  1. Valider: python scripts/validate_dataset.py")
        print(f"  2. Pipeline: ./scripts/auto-pipeline.sh pipeline")

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
