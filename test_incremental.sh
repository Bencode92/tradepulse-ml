#!/bin/bash
# Quick start script for incremental learning system
# Usage: ./test_incremental.sh

set -e

echo "🔄 TradePulse ML - Incremental Learning Test"
echo "============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo "🐍 Checking Python environment..."
python --version
pip --version

# Install dependencies if needed
echo ""
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt
pip install -q -r requirements-ml.txt
echo -e "${GREEN}✅ Dependencies ready${NC}"

# Create directories
echo ""
echo "📁 Creating directory structure..."
mkdir -p datasets/daily
mkdir -p outputs/incremental
echo -e "${GREEN}✅ Directories created${NC}"

# Step 1: Convert existing datasets to daily format
echo ""
echo "📊 Step 1: Converting existing datasets to daily format..."
if ls datasets/news_*.csv 1> /dev/null 2>&1; then
    python scripts/convert_to_daily.py --batch
    echo -e "${GREEN}✅ Datasets converted${NC}"
else
    echo -e "${YELLOW}⚠️ No existing CSV files found, skipping conversion${NC}"
fi

# Step 2: Collect sample news (if no daily data exists)
echo ""
echo "📰 Step 2: Checking for daily data..."
TODAY=$(date +%F)
DAILY_FILE="datasets/daily/${TODAY}.jsonl"

if [ ! -f "$DAILY_FILE" ]; then
    echo "Collecting sample news for today..."
    python scripts/collect_news.py --source placeholder --count 20 --output "datasets/news_test.csv" || true
    python scripts/convert_to_daily.py datasets/news_test.csv --date "$TODAY" || true
    
    if [ -f "$DAILY_FILE" ]; then
        echo -e "${GREEN}✅ Sample data created${NC}"
    else
        # Create minimal test data
        echo "Creating minimal test data..."
        cat > "$DAILY_FILE" << EOF
{"text": "Apple reported strong quarterly earnings beating analyst expectations", "label_sentiment": "positive", "label_importance": "important", "date": "${TODAY}"}
{"text": "Market volatility increased amid economic uncertainty", "label_sentiment": "negative", "label_importance": "critical", "date": "${TODAY}"}
{"text": "Oil prices remained stable following OPEC meeting", "label_sentiment": "neutral", "label_importance": "general", "date": "${TODAY}"}
{"text": "Tech stocks surge as AI investments pay off", "label_sentiment": "positive", "label_importance": "important", "date": "${TODAY}"}
{"text": "Banking sector faces regulatory challenges", "label_sentiment": "negative", "label_importance": "important", "date": "${TODAY}"}
EOF
        echo -e "${GREEN}✅ Test data created${NC}"
    fi
else
    echo -e "${GREEN}✅ Daily data already exists${NC}"
fi

# Step 3: Prepare replay buffer
echo ""
echo "🔄 Step 3: Preparing replay buffer..."
python scripts/prepare_replay.py \
    --daily "$DAILY_FILE" \
    --history "datasets/history.jsonl" \
    --out "datasets/combined.jsonl" \
    --replay_size 100
echo -e "${GREEN}✅ Replay buffer prepared${NC}"

# Step 4: Run incremental training (small test)
echo ""
echo "🤖 Step 4: Running incremental training test..."
echo "This is a quick test with minimal settings..."
echo ""

# Test sentiment model
echo "Training sentiment model (test mode)..."
python scripts/finetune_incremental.py \
    --task sentiment \
    --incremental \
    --model "distilbert-base-uncased" \
    --dataset "datasets/combined.jsonl" \
    --output_dir "outputs/incremental/test" \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --gate_drop 0.5

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Sentiment model training successful${NC}"
else
    echo -e "${RED}❌ Sentiment model training failed${NC}"
fi

# Step 5: Check outputs
echo ""
echo "📊 Step 5: Checking outputs..."
if [ -d "outputs/incremental" ]; then
    echo "Output files:"
    find outputs/incremental -name "*.json" -type f | head -5
    
    # Display metrics if available
    if [ -f "outputs/last_metrics.json.sentiment" ]; then
        echo ""
        echo "Latest metrics:"
        python -c "
import json
with open('outputs/last_metrics.json.sentiment') as f:
    data = json.load(f)
    print(f'  F1 Score: {data.get(\"f1_macro\", 0):.4f}')
    print(f'  Timestamp: {data.get(\"timestamp\", \"N/A\")}')
"
    fi
    echo -e "${GREEN}✅ Outputs verified${NC}"
else
    echo -e "${YELLOW}⚠️ No outputs found${NC}"
fi

# Summary
echo ""
echo "============================================="
echo -e "${GREEN}🎉 Incremental Learning Test Complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Configure HuggingFace token in GitHub Secrets (HF_TOKEN)"
echo "2. Configure NewsAPI key in GitHub Secrets (NEWSAPI_KEY)"
echo "3. Update HF repository names in workflow file"
echo "4. Enable GitHub Actions schedule for daily training"
echo ""
echo "For full training, run:"
echo "  python scripts/finetune_incremental.py --help"
echo ""
echo "For manual workflow trigger:"
echo "  Go to Actions tab → Select '🔄 Incremental Training' → Run workflow"
echo ""
echo "Documentation: INCREMENTAL_LEARNING.md"
