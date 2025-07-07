#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradePulse News Collector - Automated Dataset Generation
========================================================

Collecte automatiquement des actualités financières et génère un dataset
labellisé pour l'entraînement FinBERT.

Usage:
    python scripts/collect_news.py --source rss
    python scripts/collect_news.py --source newsapi --newsapi-key YOUR_API_KEY
    python scripts/collect_news.py --source rss --output custom_news.csv

Sources supportées:
- RSS feeds financiers (recommandé)
- NewsAPI (si clé API disponible)
- Placeholder samples (tests uniquement)

Output:
    datasets/news_YYYYMMDD.csv avec colonnes text,label
"""

import argparse
import csv
import datetime
import logging
import os
import random
import zoneinfo
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuration des logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("news-collector")

# Fuseau horaire Paris pour les timestamps
PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")


class NewsCollector:
    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def get_placeholder_samples(self, count: int = 20) -> List[Tuple[str, str]]:
        """Génère des échantillons placeholder pour tests"""

        positive_samples = [
            (
                "Apple Inc. reported record quarterly earnings beating analyst "
                "expectations with strong iPhone sales and robust services "
                "revenue growth."
            ),
            (
                "Tesla stock surged after announcing breakthrough in battery "
                "technology that could extend vehicle range by 40% while "
                "reducing costs."
            ),
            (
                "Microsoft Azure cloud services revenue grew 35% year-over-year "
                "driven by increased enterprise digital transformation "
                "investments."
            ),
            (
                "Amazon Web Services announced expansion into three new regions "
                "boosting global infrastructure and attracting major enterprise "
                "clients."
            ),
            (
                "NVIDIA shares rallied on strong AI chip demand with data center "
                "revenue jumping 200% as artificial intelligence adoption "
                "accelerates."
            ),
            (
                "JPMorgan Chase reported better-than-expected quarterly results "
                "with net interest income growing 25% due to rising interest "
                "rates."
            ),
            (
                "Berkshire Hathaway increased its cash position to record levels "
                "while Warren Buffett identified several attractive investment "
                "opportunities."
            ),
            (
                "Bank of America raised price targets for renewable energy "
                "stocks citing accelerating green transition and supportive "
                "policy environment."
            ),
        ]

        negative_samples = [
            (
                "Federal Reserve announced unexpected interest rate hike of 75 "
                "basis points citing persistent inflation concerns and tight "
                "labor markets."
            ),
            (
                "Meta Platforms stock declined following announcement of "
                "additional workforce reductions affecting 15000 employees "
                "across multiple divisions."
            ),
            (
                "Cryptocurrency markets experienced severe volatility with "
                "Bitcoin dropping 18% following regulatory crackdowns in major "
                "Asian economies."
            ),
            (
                "Oil prices fell 12% amid concerns about global demand slowdown "
                "and unexpected inventory build-up in major consuming nations."
            ),
            (
                "Supply chain disruptions continue to impact automotive "
                "production with Ford reducing Q4 guidance due to ongoing "
                "semiconductor shortages."
            ),
            (
                "Goldman Sachs downgraded several retail stocks citing concerns "
                "over consumer spending patterns and rising inventory levels."
            ),
            (
                "European markets declined sharply following ECB warning about "
                "potential recession risks and tightening monetary policy "
                "stance."
            ),
            (
                "Netflix subscriber growth slowed significantly missing analyst "
                "estimates while facing increased competition from streaming "
                "rivals."
            ),
        ]

        neutral_samples = [
            (
                "The S&P 500 index closed unchanged at 4150 points with mixed "
                "sector performance as investors awaited key economic data "
                "releases."
            ),
            (
                "European Central Bank maintained benchmark interest rate at "
                "4.0% in line with market expectations during monthly policy "
                "meeting."
            ),
            (
                "Oil prices traded sideways around $75 per barrel following "
                "OPEC+ production announcement and mixed economic signals from "
                "major economies."
            ),
            (
                "Berkshire Hathaway's quarterly filing revealed minimal "
                "changes to portfolio holdings with cash position remaining "
                "near record levels."
            ),
            (
                "Federal Reserve officials indicated data-dependent approach "
                "to future policy decisions while monitoring inflation and "
                "employment trends carefully."
            ),
            (
                "Treasury yields remained stable ahead of upcoming inflation "
                "data release with investors positioning for potential market "
                "volatility."
            ),
            (
                "Currency markets showed limited movement with dollar index "
                "hovering near recent ranges against major trading partners."
            ),
            (
                "Commodity prices displayed mixed performance with agricultural "
                "products gaining while industrial metals remained under "
                "pressure."
            ),
        ]

        # Sélectionner un échantillon équilibré
        samples_per_class = count // 3
        remaining = count % 3

        selected_samples = []
        selected_samples.extend(
            random.sample(
                positive_samples, min(samples_per_class, len(positive_samples))
            )
        )
        selected_samples.extend(
            random.sample(
                negative_samples, min(samples_per_class, len(negative_samples))
            )
        )
        selected_samples.extend(
            random.sample(
                neutral_samples,
                min(samples_per_class + remaining, len(neutral_samples)),
            )
        )

        # Ajouter les labels
        labeled_samples = []
        labeled_samples.extend(
            [(text, "positive") for text in selected_samples[:samples_per_class]]
        )
        labeled_samples.extend(
            [
                (text, "negative")
                for text in selected_samples[
                    samples_per_class : 2 * samples_per_class
                ]
            ]
        )
        labeled_samples.extend(
            [
                (text, "neutral")
                for text in selected_samples[2 * samples_per_class :]
            ]
        )

        # Mélanger l'ordre
        random.shuffle(labeled_samples)

        return labeled_samples

    def collect_from_rss(self, count: int = 20) -> List[Tuple[str, str]]:
        """Collecte depuis des flux RSS financiers (nécessite feedparser)"""
        try:
            import feedparser
        except ImportError:
            logger.warning(
                "feedparser non installé, utilisation des échantillons "
                "placeholder. Installez avec: pip install feedparser"
            )
            return self.get_placeholder_samples(count)

        rss_feeds = [
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://feeds.reuters.com/reuters/businessNews",
            "https://rss.cnn.com/rss/money_latest.rss",
        ]

        all_articles = []
        today = datetime.datetime.now(PARIS_TZ).date()

        for feed_url in rss_feeds:
            try:
                logger.info(f"Collecte depuis {feed_url}")
                feed = feedparser.parse(feed_url)

                # CORRECTIF 3: Filtrer les articles publiés aujourd'hui
                for entry in feed.entries:
                    pub = entry.get("published_parsed")
                    if pub:
                        pub_date = datetime.date(*pub[:3])
                        if pub_date != today:
                            continue  # Sauter les articles non publiés aujourd'hui
                    
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")

                    # Combiner titre et résumé
                    text = f"{title}. {summary}".strip()
                    if len(text) > 50:  # Filtrer les textes trop courts
                        all_articles.append(text)
                        
                    # Limiter pour éviter trop d'articles d'un seul feed
                    if len(all_articles) >= count * 2:
                        break

            except Exception as e:
                logger.warning(f"Erreur collecte RSS {feed_url}: {e}")

        if not all_articles:
            logger.warning(
                "Aucun article collecté depuis RSS aujourd'hui, "
                "utilisation des échantillons placeholder"
            )
            return self.get_placeholder_samples(count)

        # Limiter au nombre demandé et labelliser
        selected_articles = all_articles[:count]
        labeled_samples = []

        for text in selected_articles:
            # Labellisation basique par mots-clés (à améliorer)
            label = self._simple_sentiment_analysis(text)
            labeled_samples.append((text, label))

        logger.info(f"✅ Collecté {len(labeled_samples)} articles d'aujourd'hui depuis RSS")
        return labeled_samples

    def collect_from_newsapi(
        self, count: int = 20, api_key: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Collecte depuis NewsAPI (nécessite clé API)"""
        if not api_key:
            api_key = os.getenv("NEWSAPI_KEY")

        if not api_key:
            logger.warning(
                "Clé NewsAPI non trouvée (variable NEWSAPI_KEY ou --newsapi-key), "
                "utilisation des échantillons placeholder"
            )
            return self.get_placeholder_samples(count)

        try:
            import requests
        except ImportError:
            logger.warning(
                "requests non installé, utilisation des échantillons "
                "placeholder. Installez avec: pip install requests"
            )
            return self.get_placeholder_samples(count)

        # CORRECTIF 3: Ajouter filtre temporel pour aujourd'hui uniquement
        today = datetime.datetime.now(PARIS_TZ).date().isoformat()
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "stock market OR finance OR earnings OR Federal Reserve",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": count,
            "from": today,
            "to": today,
            "apiKey": api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            labeled_samples = []

            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")

                # Combiner titre et description
                text = (
                    f"{title}. {description}".strip() if description else title
                )

                if len(text) > 50:  # Filtrer les textes trop courts
                    label = self._simple_sentiment_analysis(text)
                    labeled_samples.append((text, label))

            logger.info(f"✅ Collecté {len(labeled_samples)} articles d'aujourd'hui depuis NewsAPI")
            return labeled_samples[:count]

        except Exception as e:
            logger.error(f"Erreur NewsAPI: {e}")
            return self.get_placeholder_samples(count)

    def _simple_sentiment_analysis(self, text: str) -> str:
        """Analyse de sentiment basique par mots-clés"""
        text_lower = text.lower()

        positive_keywords = [
            "surge",
            "rally",
            "gain",
            "rise",
            "growth",
            "beat",
            "exceed",
            "strong",
            "record",
            "high",
            "boost",
            "increase",
            "up",
            "bullish",
            "positive",
            "earnings beat",
            "revenue growth",
            "profit",
            "expansion",
        ]

        negative_keywords = [
            "fall",
            "drop",
            "decline",
            "crash",
            "loss",
            "miss",
            "weak",
            "down",
            "concern",
            "worry",
            "risk",
            "pressure",
            "bearish",
            "negative",
            "layoff",
            "recession",
            "inflation",
            "crisis",
            "volatility",
        ]

        positive_score = sum(
            1 for keyword in positive_keywords if keyword in text_lower
        )
        negative_score = sum(
            1 for keyword in negative_keywords if keyword in text_lower
        )

        if positive_score > negative_score:
            return "positive"
        elif negative_score > positive_score:
            return "negative"
        else:
            return "neutral"

    def save_dataset(
        self,
        samples: List[Tuple[str, str]],
        output_file: Optional[Path] = None,
    ) -> Path:
        """Sauvegarde le dataset au format CSV"""
        if output_file is None:
            today = datetime.datetime.now(PARIS_TZ).strftime("%Y%m%d")
            output_file = self.output_dir / f"news_{today}.csv"

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerows(samples)

        logger.info(
            f"✅ Dataset sauvegardé: {output_file} ({len(samples)} "
            "échantillons)"
        )
        return output_file

    def collect_and_save(
        self,
        source: str = "rss",  # CORRECTIF 1: Changer défaut de placeholder à rss
        count: int = 20,
        output_file: Optional[Path] = None,
        **kwargs,
    ) -> Path:
        """Collecte les news et sauvegarde le dataset"""
        logger.info(
            f"🔄 Collecte de {count} échantillons depuis source: {source}"
        )

        if source == "placeholder":
            samples = self.get_placeholder_samples(count)
        elif source == "rss":
            samples = self.collect_from_rss(count)
        elif source == "newsapi":
            samples = self.collect_from_newsapi(count, kwargs.get("api_key"))
        else:
            raise ValueError(f"Source non supportée: {source}")

        if not samples:
            raise RuntimeError("Aucun échantillon collecté")

        # Statistiques de distribution
        labels = [label for _, label in samples]
        label_counts = {label: labels.count(label) for label in set(labels)}
        logger.info(f"📊 Distribution: {label_counts}")

        return self.save_dataset(samples, output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Collecte automatique d'actualités financières pour "
        "TradePulse"
    )
    parser.add_argument(
        "--source",
        choices=["placeholder", "rss", "newsapi"],
        default="rss",  # CORRECTIF 1: Changer défaut de placeholder à rss
        help="Source de données (défaut: rss)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Nombre d'échantillons à collecter (défaut: 20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Fichier de sortie (défaut: datasets/news_YYYYMMDD.csv)",
    )
    parser.add_argument(
        "--newsapi-key",
        help="Clé API NewsAPI (ou utiliser variable NEWSAPI_KEY)",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Répertoire de sortie (défaut: datasets)",
    )
    # CORRECTIF 2: Rendre le seed optionnel
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed optionnel pour reproductibilité (utile seulement pour les placeholders)",
    )

    args = parser.parse_args()

    # CORRECTIF 2: Appliquer le seed seulement s'il est fourni
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"🎲 Seed fixé à {args.seed} pour reproductibilité")

    collector = NewsCollector(args.output_dir)

    try:
        output_file = collector.collect_and_save(
            source=args.source,
            count=args.count,
            output_file=args.output,
            api_key=args.newsapi_key,
        )

        print(f"✅ Dataset généré avec succès: {output_file}")

        # Suggestion pour la suite
        print("\n🚀 Prochaines étapes:")
        print(f"  1. Valider: python scripts/validate_dataset.py {output_file}")
        print(
            f"  2. Commit: git add {output_file} && "
            "git commit -m 'Add daily dataset'"
        )
        print(
            "  3. Push: git push (déclenche validation + fine-tuning "
            "automatique)"
        )

    except Exception as e:
        logger.error(f"❌ Erreur lors de la collecte: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
