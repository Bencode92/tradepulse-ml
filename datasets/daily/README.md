# Daily datasets directory
This directory contains daily JSONL files for incremental learning.

Format: `YYYY-MM-DD.jsonl`

Each file contains news articles collected for that specific day with:
- `text`: Article content
- `label_sentiment`: Sentiment classification (positive/negative/neutral)
- `label_importance`: Importance level (general/important/critical)
- Additional metadata (url, title, source, date)
