#!/bin/bash
# initial_correlations_training.sh - Create the initial correlations model

echo "🚀 Initial training for correlations model (non-incremental)"

# Variables
DATASET="${1:-datasets/news_20250724.csv}"
OUTPUT_DIR="models/finbert-correlations-initial"
MODEL_NAME="yiyanghkust/finbert-tone"

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN not set. Define it with: export HF_TOKEN='your_token'"
    exit 1
fi

echo "📊 Dataset: $DATASET"
echo "📁 Output: $OUTPUT_DIR"
echo "🤖 Base model: $MODEL_NAME"

# Training WITHOUT --incremental to create initial model
python scripts/finetune.py \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --target-column correlations \
    --epochs 3 \
    --train_bs 8 \
    --eval_bs 16 \
    --max_length 256 \
    --push \
    --mode production

echo "✅ Initial training complete!"
echo "📤 Model should be available at https://huggingface.co/Bencode92/tradepulse-finbert-correlations"
