# 🔄 TradePulse ML - Incremental Learning System

## 📋 Overview

This document describes the incremental learning system for TradePulse ML, which enables daily model updates using LoRA (Low-Rank Adaptation) for efficient fine-tuning.

## 🎯 Key Features

- **Daily Updates**: Automatic daily model updates with new financial news
- **Replay Buffer**: Stratified sampling from historical data to prevent catastrophic forgetting
- **LoRA Fine-tuning**: Efficient training using only 10% of parameters
- **Quality Gating**: Automatic rollback if model performance degrades
- **Dual Tasks**: Supports both sentiment and importance classification

## 🏗️ Architecture

```
Daily News Collection
        ↓
CSV → JSONL Conversion
        ↓
Replay Buffer Creation
        ↓
LoRA Fine-tuning
        ↓
Quality Gate Check
        ↓
HuggingFace Push (if passed)
```

## 🚀 Quick Start

### 1. Manual Incremental Training

```bash
# Collect today's news
python scripts/collect_news.py --source mixed --count 60 --days 1

# Convert to daily format
python scripts/convert_to_daily.py

# Prepare replay buffer
python scripts/prepare_replay.py \
  --daily datasets/daily/$(date +%F).jsonl \
  --history datasets/history.jsonl \
  --out datasets/combined.jsonl \
  --replay_size 800

# Train sentiment model with LoRA
python scripts/finetune_incremental.py \
  --task sentiment \
  --incremental \
  --model yiyanghkust/finbert-tone \
  --dataset datasets/combined.jsonl \
  --epochs 2 \
  --batch_size 8 \
  --hf_repo Bencode92/tradepulse-finbert-sentiment

# Train importance model with LoRA
python scripts/finetune_incremental.py \
  --task importance \
  --incremental \
  --model distilbert-base-uncased \
  --dataset datasets/combined.jsonl \
  --epochs 2 \
  --batch_size 8 \
  --hf_repo Bencode92/tradepulse-finbert-importance
```

### 2. Automatic Daily Training

The system runs automatically every day at 4:30 AM UTC via GitHub Actions.

To trigger manually:
1. Go to Actions tab
2. Select "🔄 Incremental Training"
3. Click "Run workflow"
4. Configure parameters (optional)

### 3. Convert Existing Datasets

```bash
# Convert single file
python scripts/convert_to_daily.py datasets/news_20250107.csv

# Batch convert all CSV files
python scripts/convert_to_daily.py --batch

# Auto-select latest
python scripts/convert_to_daily.py
```

## 📊 Data Format

### Daily JSONL Format
```json
{
  "text": "Apple reported strong earnings...",
  "label_sentiment": "positive",
  "label_importance": "important",
  "url": "https://...",
  "title": "Apple Q4 Earnings",
  "source": "Reuters",
  "date": "2025-01-07"
}
```

### Labels

**Sentiment Labels:**
- `positive`: Bullish/favorable news
- `negative`: Bearish/unfavorable news
- `neutral`: Neutral/informational content

**Importance Labels:**
- `general`: Regular market news
- `important`: Significant market events
- `critical`: Major market-moving news

## ⚙️ Configuration

### Environment Variables

Set these in GitHub Secrets:

```bash
HF_TOKEN=hf_xxx...           # HuggingFace token with write access
NEWSAPI_KEY=xxx...           # NewsAPI key for news collection
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 2 | Number of training epochs |
| `--batch_size` | 8 | Training batch size |
| `--learning_rate` | 3e-5 | Learning rate |
| `--replay_size` | 800 | Historical samples in replay buffer |
| `--gate_drop` | 0.01 | Max allowed F1 score drop |
| `--lora_r` | 8 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha parameter |
| `--lora_dropout` | 0.05 | LoRA dropout rate |

## 📈 Monitoring

### Metrics Tracking

Metrics are saved after each training run:
- `outputs/last_metrics.json.sentiment` - Latest sentiment model metrics
- `outputs/last_metrics.json.importance` - Latest importance model metrics
- `outputs/incremental/*/metrics.json` - Detailed training metrics

### Quality Gate

The system automatically checks F1 score before pushing:
- If F1 drops more than `gate_drop` (default 1%), the push is skipped
- This prevents degraded models from reaching production

### GitHub Actions Artifacts

Training artifacts are saved for 30 days:
- Model checkpoints
- Training metrics
- Evaluation results

## 🔍 Troubleshooting

### Common Issues

1. **No daily data found**
   ```bash
   # Manually create today's data
   python scripts/collect_news.py --source mixed --count 60
   python scripts/convert_to_daily.py
   ```

2. **Gate check failing**
   ```bash
   # Increase gate tolerance
   python scripts/finetune_incremental.py --gate_drop 0.02
   ```

3. **Out of memory**
   ```bash
   # Reduce batch size
   python scripts/finetune_incremental.py --batch_size 4
   ```

4. **HuggingFace push failing**
   ```bash
   # Check token permissions
   echo $HF_TOKEN | huggingface-cli login --token
   ```

## 🎯 Performance Expectations

With incremental learning:

| Metric | Baseline | After 7 days | After 30 days |
|--------|----------|--------------|---------------|
| F1 Score | 0.82 | 0.84-0.86 | 0.86-0.88 |
| Training Time | 30 min | 5 min | 5 min |
| GPU Memory | 8 GB | 3 GB | 3 GB |
| Model Size | 440 MB | 445 MB | 445 MB |

## 🔮 Future Improvements

- [ ] Multi-GPU training support
- [ ] Advanced replay strategies (prioritized experience replay)
- [ ] Continuous learning with online updates
- [ ] A/B testing between incremental and full retrain
- [ ] Automatic hyperparameter tuning
- [ ] Model ensemble with voting

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Catastrophic Forgetting in Neural Networks](https://arxiv.org/abs/1312.6211)

## 🤝 Contributing

To contribute to the incremental learning system:

1. Test changes locally with small datasets
2. Ensure quality gate passes
3. Update this documentation
4. Submit PR with metrics comparison

---

**Note**: The incremental learning system is designed to complement, not replace, periodic full retraining. Consider full retraining monthly for best results.
