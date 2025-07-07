#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradePulse News Collector - Advanced Version with Deduplication
===============================================================

Collecte automatiquement des actualités financières avec gestion des doublons
et fenêtre temporelle élargie pour maximiser la diversité.

Nouvelles fonctionnalités:
- Fenêtre temporelle configurable (1-7 jours)
- Déduplication automatique des articles 
- Cache local pour éviter les re-collectes
- Sources RSS étendues avec rotation
- Pagination améliorée pour NewsAPI
- Gestion des articles similaires

Usage:
    python scripts/collect_news.py --source rss --days 3 --count 50
    python scripts/collect_news.py --source newsapi --newsapi-key KEY --days 2
    python scripts/collect_news.py --source mixed --count 60 --dedupe-cache
"""

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import random
import re
import zoneinfo
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

# Configuration des logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("news-collector")

# Fuseau horaire Paris pour les timestamps
PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")


class ArticleDeduplicator:
    """Gestionnaire de déduplication des articles"""
    
    def __init__(self, cache_file: str = "datasets/.article_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(exist_ok=True)
        self.seen_articles: Set[str] = set()
        self.load_cache()
    
    def load_cache(self):
        """Charge le cache des articles déjà vus"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.seen_articles = set(cache_data.get('articles', []))
                    logger.info(f"Cache chargé: {len(self.seen_articles)} articles connus")
            except Exception as e:
                logger.warning(f"Erreur chargement cache: {e}")
    
    def save_cache(self):
        """Sauvegarde le cache"""
        try:
            cache_data = {
                'articles': list(self.seen_articles),
                'last_updated': datetime.datetime.now(PARIS_TZ).isoformat()
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cache sauvegardé: {len(self.seen_articles)} articles")
        except Exception as e:
            logger.warning(f"Erreur sauvegarde cache: {e}")
    
    def get_article_hash(self, text: str) -> str:
        """Génère un hash unique pour un article"""
        # Normaliser le texte : minuscules, espaces, ponctuation
        normalized = re.sub(r'[^\w\s]', '', text.lower().strip())
        normalized = re.sub(r'\s+', ' ', normalized)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, text: str, similarity_threshold: float = 0.8) -> bool:
        """Vérifie si un article est un doublon"""
        article_hash = self.get_article_hash(text)
        
        # Vérification hash exact
        if article_hash in self.seen_articles:
            return True
        
        # Vérification similarité (pour articles très similaires)
        if len(text) < 50:  # Trop court pour être fiable
            return False
            
        words = set(text.lower().split())
        if len(words) < 10:
            return False
            
        # Vérification basique de similarité avec articles récents
        # (implémentation simplifiée - pour production, utiliser difflib ou similar)
        for existing_hash in list(self.seen_articles)[-100:]:  # Check only recent articles
            # Pour simplicité, on se base sur le hash exact uniquement
            pass
            
        return False
    
    def add_article(self, text: str):
        """Ajoute un article au cache"""
        article_hash = self.get_article_hash(text)
        self.seen_articles.add(article_hash)
    
    def cleanup_old_articles(self, days_to_keep: int = 30):
        """Nettoie les anciens articles du cache"""
        # Implémentation simplifiée - garde les N derniers articles
        if len(self.seen_articles) > days_to_keep * 50:  # ~50 articles/jour max
            articles_list = list(self.seen_articles)
            self.seen_articles = set(articles_list[-days_to_keep * 50:])
            logger.info(f"Cache nettoyé: gardé {len(self.seen_articles)} articles récents")


class AdvancedNewsCollector:
    """Collecteur d'actualités avancé avec déduplication"""
    
    def __init__(self, output_dir: str = "datasets", use_cache: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.deduplicator = ArticleDeduplicator() if use_cache else None
        
        # Sources RSS étendues avec rotation
        self.rss_sources = {
            'primary': [
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://www.cnbc.com/id/100003114/device/rss/rss.html", 
                "https://feeds.reuters.com/reuters/businessNews",
                "https://rss.cnn.com/rss/money_latest.rss",
            ],
            'secondary': [
                "https://feeds.marketwatch.com/marketwatch/marketpulse/",
                "https://www.sec.gov/news/pressreleases.rss",
                "https://www.treasury.gov/resource-center/data-chart-center/Pages/rss.aspx",
                "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
            ],
            'alternative': [
                "https://www.businesswire.com/portal/site/home/news/rss/",
                "https://finance.yahoo.com/news/rssindex",
                "https://seekingalpha.com/market_currents.xml",
            ]
        }

    def get_date_range(self, days: int = 1) -> Tuple[datetime.date, datetime.date]:
        """Génère une plage de dates pour la collecte"""
        end_date = datetime.datetime.now(PARIS_TZ).date()
        start_date = end_date - datetime.timedelta(days=days-1)
        return start_date, end_date

    def collect_from_rss_extended(self, count: int = 50, days: int = 3) -> List[Tuple[str, str]]:
        """Collecte RSS avec sources multiples et déduplication"""
        try:
            import feedparser
        except ImportError:
            logger.warning(
                "feedparser non installé. Installez avec: pip install feedparser"
            )
            return self.get_placeholder_samples(count)

        start_date, end_date = self.get_date_range(days)
        logger.info(f"Collecte RSS du {start_date} au {end_date} ({days} jours)")
        
        all_articles = []
        sources_used = []
        
        # Utiliser toutes les sources disponibles
        all_feeds = []
        for category, feeds in self.rss_sources.items():
            all_feeds.extend(feeds)
            
        # Mélanger les sources pour éviter les biais
        random.shuffle(all_feeds)
        
        for feed_url in all_feeds:
            try:
                logger.info(f"Collecte depuis {urlparse(feed_url).netloc}")
                feed = feedparser.parse(feed_url)
                
                articles_from_feed = 0
                
                for entry in feed.entries:
                    # Filtrage par date élargi
                    pub = entry.get("published_parsed")
                    if pub:
                        pub_date = datetime.date(*pub[:3])
                        if not (start_date <= pub_date <= end_date):
                            continue
                    
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    
                    # Combiner titre et résumé
                    text = f"{title}. {summary}".strip()
                    
                    # Filtres de qualité
                    if len(text) < 50 or len(text) > 2000:
                        continue
                        
                    # Filtrage financier
                    if not self._is_financial_content(text):
                        continue
                    
                    # Déduplication
                    if self.deduplicator and self.deduplicator.is_duplicate(text):
                        continue
                    
                    all_articles.append(text)
                    articles_from_feed += 1
                    
                    if self.deduplicator:
                        self.deduplicator.add_article(text)
                    
                    # Limiter par feed pour diversité
                    if articles_from_feed >= 15:
                        break
                
                if articles_from_feed > 0:
                    sources_used.append(f"{urlparse(feed_url).netloc} ({articles_from_feed})")
                
                # Arrêter si on a assez d'articles
                if len(all_articles) >= count * 1.5:
                    break
                    
            except Exception as e:
                logger.warning(f"Erreur RSS {feed_url}: {e}")

        logger.info(f"Sources utilisées: {', '.join(sources_used)}")
        
        if not all_articles:
            logger.warning("Aucun article RSS collecté, utilisation des placeholders")
            return self.get_placeholder_samples(count)

        # Labellisation et sélection finale
        labeled_samples = []
        for text in all_articles[:count]:
            label = self._advanced_sentiment_analysis(text)
            labeled_samples.append((text, label))

        logger.info(f"✅ {len(labeled_samples)} articles RSS collectés avec déduplication")
        return labeled_samples

    def collect_from_newsapi_paginated(
        self, count: int = 50, days: int = 2, api_key: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Collecte NewsAPI avec pagination et fenêtre temporelle"""
        if not api_key:
            api_key = os.getenv("NEWSAPI_KEY")

        if not api_key:
            logger.warning("Clé NewsAPI manquante")
            return self.get_placeholder_samples(count)

        try:
            import requests
        except ImportError:
            logger.warning("requests non installé. Installez avec: pip install requests")
            return self.get_placeholder_samples(count)

        start_date, end_date = self.get_date_range(days)
        logger.info(f"Collecte NewsAPI du {start_date} au {end_date}")
        
        all_articles = []
        
        # Requêtes multiples avec mots-clés variés
        queries = [
            "stock market OR finance OR earnings",
            "Federal Reserve OR interest rates OR inflation", 
            "cryptocurrency OR bitcoin OR blockchain",
            "IPO OR merger OR acquisition",
            "GDP OR employment OR economic data"
        ]
        
        for query in queries:
            try:
                # Pagination NewsAPI
                for page in range(1, 4):  # 3 pages max par requête
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        "q": query,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 50,  # Maximum par page
                        "page": page,
                        "from": start_date.isoformat(),
                        "to": end_date.isoformat(),
                        "apiKey": api_key,
                    }

                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()

                    articles = data.get("articles", [])
                    if not articles:
                        break  # Plus d'articles pour cette requête
                    
                    for article in articles:
                        title = article.get("title", "")
                        description = article.get("description", "")

                        text = f"{title}. {description}".strip() if description else title
                        
                        if len(text) < 50 or len(text) > 2000:
                            continue
                            
                        if not self._is_financial_content(text):
                            continue

                        # Déduplication
                        if self.deduplicator and self.deduplicator.is_duplicate(text):
                            continue

                        all_articles.append(text)
                        
                        if self.deduplicator:
                            self.deduplicator.add_article(text)
                        
                        if len(all_articles) >= count:
                            break
                    
                    if len(all_articles) >= count:
                        break
                        
            except Exception as e:
                logger.warning(f"Erreur NewsAPI query '{query}': {e}")

        if not all_articles:
            logger.warning("Aucun article NewsAPI collecté")
            return self.get_placeholder_samples(count)

        # Labellisation
        labeled_samples = []
        for text in all_articles[:count]:
            label = self._advanced_sentiment_analysis(text)
            labeled_samples.append((text, label))

        logger.info(f"✅ {len(labeled_samples)} articles NewsAPI collectés avec pagination")
        return labeled_samples

    def collect_mixed_sources(self, count: int = 60, days: int = 3, **kwargs) -> List[Tuple[str, str]]:
        """Collecte depuis sources multiples pour maximiser la diversité"""
        logger.info(f"Collecte mixte: RSS + NewsAPI ({count} articles, {days} jours)")
        
        # Répartition 70% RSS, 30% NewsAPI
        rss_count = int(count * 0.7)
        api_count = count - rss_count
        
        all_samples = []
        
        # RSS
        try:
            rss_samples = self.collect_from_rss_extended(rss_count, days)
            all_samples.extend(rss_samples)
            logger.info(f"RSS: {len(rss_samples)} articles")
        except Exception as e:
            logger.warning(f"Erreur collecte RSS: {e}")
        
        # NewsAPI (si clé disponible)
        api_key = kwargs.get('api_key') or os.getenv("NEWSAPI_KEY")
        if api_key:
            try:
                api_samples = self.collect_from_newsapi_paginated(api_count, days, api_key)
                all_samples.extend(api_samples)
                logger.info(f"NewsAPI: {len(api_samples)} articles")
            except Exception as e:
                logger.warning(f"Erreur collecte NewsAPI: {e}")
        
        # Compléter avec placeholders si nécessaire
        if len(all_samples) < count // 2:
            placeholder_needed = count - len(all_samples)
            placeholders = self.get_placeholder_samples(placeholder_needed)
            all_samples.extend(placeholders)
            logger.info(f"Placeholders: {len(placeholders)} articles")
        
        # Mélanger et limiter
        random.shuffle(all_samples)
        final_samples = all_samples[:count]
        
        logger.info(f"✅ Collecte mixte terminée: {len(final_samples)} articles")
        return final_samples

    def _is_financial_content(self, text: str) -> bool:
        """Filtre le contenu financier"""
        financial_keywords = [
            'stock', 'market', 'finance', 'earnings', 'revenue', 'profit',
            'investment', 'trading', 'economy', 'federal reserve', 'fed',
            'interest rate', 'inflation', 'gdp', 'nasdaq', 'dow jones',
            'sp500', 's&p', 'cryptocurrency', 'bitcoin', 'blockchain',
            'ipo', 'merger', 'acquisition', 'dividend', 'bond', 'treasury'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in financial_keywords)

    def _advanced_sentiment_analysis(self, text: str) -> str:
        """Analyse de sentiment améliorée"""
        text_lower = text.lower()

        # Mots-clés étendus avec pondération
        positive_patterns = {
            'high_impact': ['surge', 'rally', 'soar', 'jump', 'record high', 'all-time high', 'breakthrough'],
            'medium_impact': ['gain', 'rise', 'growth', 'beat', 'exceed', 'strong', 'robust'],
            'low_impact': ['increase', 'up', 'positive', 'bullish', 'optimistic']
        }
        
        negative_patterns = {
            'high_impact': ['crash', 'plummet', 'collapse', 'crisis', 'recession', 'bear market'],
            'medium_impact': ['fall', 'drop', 'decline', 'loss', 'weak', 'concern'],
            'low_impact': ['down', 'pressure', 'worry', 'risk', 'negative']
        }

        # Calcul pondéré
        positive_score = 0
        negative_score = 0
        
        for impact, keywords in positive_patterns.items():
            weight = {'high_impact': 3, 'medium_impact': 2, 'low_impact': 1}[impact]
            for keyword in keywords:
                if keyword in text_lower:
                    positive_score += weight

        for impact, keywords in negative_patterns.items():
            weight = {'high_impact': 3, 'medium_impact': 2, 'low_impact': 1}[impact]
            for keyword in keywords:
                if keyword in text_lower:
                    negative_score += weight

        # Contexte nuancé
        if 'despite' in text_lower or 'however' in text_lower:
            # Inverser le sentiment si mots de contraste
            positive_score, negative_score = negative_score, positive_score

        if positive_score > negative_score + 1:
            return "positive"
        elif negative_score > positive_score + 1:
            return "negative"
        else:
            return "neutral"

    def get_placeholder_samples(self, count: int = 20) -> List[Tuple[str, str]]:
        """Échantillons placeholder (version originale)"""
        # Code original maintenu pour compatibilité
        positive_samples = [
            ("Apple Inc. reported record quarterly earnings beating analyst expectations with strong iPhone sales and robust services revenue growth.", "positive"),
            ("Tesla stock surged after announcing breakthrough in battery technology that could extend vehicle range by 40% while reducing costs.", "positive"),
            ("Microsoft Azure cloud services revenue grew 35% year-over-year driven by increased enterprise digital transformation investments.", "positive"),
        ]
        
        negative_samples = [
            ("Federal Reserve announced unexpected interest rate hike of 75 basis points citing persistent inflation concerns and tight labor markets.", "negative"),
            ("Meta Platforms stock declined following announcement of additional workforce reductions affecting 15000 employees across multiple divisions.", "negative"),
        ]
        
        neutral_samples = [
            ("The S&P 500 index closed unchanged at 4150 points with mixed sector performance as investors awaited key economic data releases.", "neutral"),
            ("European Central Bank maintained benchmark interest rate at 4.0% in line with market expectations during monthly policy meeting.", "neutral"),
        ]
        
        # Sélection équilibrée
        samples_per_class = count // 3
        all_samples = []
        all_samples.extend(positive_samples[:samples_per_class])
        all_samples.extend(negative_samples[:samples_per_class])
        all_samples.extend(neutral_samples[:samples_per_class])
        
        # Compléter si nécessaire
        remaining = count - len(all_samples)
        if remaining > 0:
            extra_samples = (positive_samples + negative_samples + neutral_samples)[:remaining]
            all_samples.extend(extra_samples)
        
        random.shuffle(all_samples)
        return all_samples[:count]

    def save_dataset(self, samples: List[Tuple[str, str]], output_file: Optional[Path] = None) -> Path:
        """Sauvegarde avec métadonnées enrichies"""
        if output_file is None:
            today = datetime.datetime.now(PARIS_TZ).strftime("%Y%m%d")
            output_file = self.output_dir / f"news_{today}.csv"

        # Sauvegarde CSV
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerows(samples)

        # Métadonnées
        metadata = {
            'filename': output_file.name,
            'created_at': datetime.datetime.now(PARIS_TZ).isoformat(),
            'article_count': len(samples),
            'label_distribution': {
                label: sum(1 for _, l in samples if l == label)
                for label in ['positive', 'negative', 'neutral']
            },
            'deduplication_enabled': self.deduplicator is not None,
            'cache_size': len(self.deduplicator.seen_articles) if self.deduplicator else 0
        }
        
        metadata_file = output_file.with_suffix('.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Dataset: {output_file} ({len(samples)} articles)")
        logger.info(f"📊 Distribution: {metadata['label_distribution']}")
        
        return output_file

    def collect_and_save(self, source: str = "mixed", count: int = 40, days: int = 3, 
                        output_file: Optional[Path] = None, **kwargs) -> Path:
        """Collecte avancée avec toutes les améliorations"""
        logger.info(f"🔄 Collecte avancée: {source}, {count} articles, {days} jours")

        # Collecte selon la source
        if source == "placeholder":
            samples = self.get_placeholder_samples(count)
        elif source == "rss":
            samples = self.collect_from_rss_extended(count, days)
        elif source == "newsapi":
            samples = self.collect_from_newsapi_paginated(count, days, kwargs.get("api_key"))
        elif source == "mixed":
            samples = self.collect_mixed_sources(count, days, **kwargs)
        else:
            raise ValueError(f"Source non supportée: {source}")

        if not samples:
            raise RuntimeError("Aucun échantillon collecté")

        # Sauvegarde cache
        if self.deduplicator:
            self.deduplicator.cleanup_old_articles()
            self.deduplicator.save_cache()

        return self.save_dataset(samples, output_file)


def main():
    parser = argparse.ArgumentParser(description="Collecteur d'actualités TradePulse avancé")
    
    parser.add_argument("--source", choices=["placeholder", "rss", "newsapi", "mixed"],
                       default="mixed", help="Source de données (défaut: mixed)")
    parser.add_argument("--count", type=int, default=40,
                       help="Nombre d'articles (défaut: 40)")
    parser.add_argument("--days", type=int, default=3, 
                       help="Fenêtre temporelle en jours (défaut: 3)")
    parser.add_argument("--output", type=Path, 
                       help="Fichier de sortie (défaut: auto)")
    parser.add_argument("--newsapi-key", help="Clé NewsAPI")
    parser.add_argument("--output-dir", default="datasets", 
                       help="Répertoire de sortie")
    parser.add_argument("--no-cache", action="store_true",
                       help="Désactiver la déduplication")
    parser.add_argument("--seed", type=int, help="Seed optionnel")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"🎲 Seed fixé: {args.seed}")

    collector = AdvancedNewsCollector(args.output_dir, use_cache=not args.no_cache)

    try:
        output_file = collector.collect_and_save(
            source=args.source,
            count=args.count, 
            days=args.days,
            output_file=args.output,
            api_key=args.newsapi_key
        )

        print(f"✅ Dataset généré: {output_file}")
        print(f"📁 Métadonnées: {output_file.with_suffix('.json')}")
        
        print("\n🚀 Prochaines étapes:")
        print(f"  1. Valider: python scripts/validate_dataset.py {output_file}")
        print(f"  2. Éditer: open news_editor.html")
        print(f"  3. Commit: git add {output_file} && git commit -m 'Add diversified dataset'")

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())