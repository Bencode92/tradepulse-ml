#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare replay buffer for incremental learning
Combines new daily data with historical samples for balanced training
"""
import argparse
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


def load_jsonl(filepath):
    """Load JSONL file"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return data


def save_jsonl(data, filepath):
    """Save data to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_historical_datasets(date_str, days_back=30):
    """Get list of historical dataset files"""
    datasets_dir = Path("datasets")
    historical_files = []
    
    # Parse the target date
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        target_date = datetime.now()
    
    # Look for files in the past N days
    for i in range(1, days_back + 1):
        past_date = target_date - timedelta(days=i)
        date_pattern = past_date.strftime("%Y%m%d")
        
        # Look for files matching the date pattern
        for filepath in datasets_dir.glob(f"*{date_pattern}*.jsonl"):
            if filepath.exists():
                historical_files.append(filepath)
    
    # Also include any labeled datasets
    for filepath in datasets_dir.glob("labeled_*.jsonl"):
        historical_files.append(filepath)
    
    return list(set(historical_files))  # Remove duplicates


def balance_by_label(data, target_field="label_sentiment"):
    """Balance dataset by label distribution"""
    by_label = defaultdict(list)
    
    for item in data:
        # Try different label field names
        label = item.get(target_field) or item.get("label")
        if label:
            by_label[label].append(item)
    
    # Find minimum class size
    if not by_label:
        return data
    
    min_size = min(len(items) for items in by_label.values())
    
    # Sample equally from each class
    balanced = []
    for label, items in by_label.items():
        sampled = random.sample(items, min(min_size, len(items)))
        balanced.extend(sampled)
    
    random.shuffle(balanced)
    return balanced


def main():
    parser = argparse.ArgumentParser(description="Prepare replay buffer for incremental learning")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Target date (YYYY-MM-DD)")
    parser.add_argument("--replay-size", type=int, default=800,
                        help="Maximum size of replay buffer")
    parser.add_argument("--days-back", type=int, default=30,
                        help="Number of days to look back for historical data")
    parser.add_argument("--output", default="datasets/combined.jsonl",
                        help="Output file path")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Balance dataset by label distribution")
    
    args = parser.parse_args()
    
    print(f"📅 Preparing replay buffer for {args.date}")
    
    # Load today's data
    today_pattern = args.date.replace("-", "")
    today_files = list(Path("datasets").glob(f"*{today_pattern}*.jsonl"))
    
    today_data = []
    for filepath in today_files:
        print(f"  Loading today's data: {filepath}")
        today_data.extend(load_jsonl(filepath))
    
    print(f"  Found {len(today_data)} samples from today")
    
    # Load historical data
    historical_files = get_historical_datasets(args.date, args.days_back)
    historical_data = []
    
    for filepath in historical_files[:10]:  # Limit to 10 most recent files
        print(f"  Loading historical: {filepath.name}")
        historical_data.extend(load_jsonl(filepath))
    
    print(f"  Found {len(historical_data)} historical samples")
    
    # Combine data
    all_data = today_data + historical_data
    
    # Remove duplicates based on text
    seen_texts = set()
    unique_data = []
    for item in all_data:
        text = item.get("text", "")
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_data.append(item)
    
    print(f"  After deduplication: {len(unique_data)} samples")
    
    # Balance if requested
    if args.balance:
        unique_data = balance_by_label(unique_data)
        print(f"  After balancing: {len(unique_data)} samples")
    
    # Limit to replay size
    if len(unique_data) > args.replay_size:
        unique_data = random.sample(unique_data, args.replay_size)
        print(f"  Limited to replay size: {args.replay_size} samples")
    
    # Save combined dataset
    save_jsonl(unique_data, args.output)
    print(f"✅ Saved combined dataset to {args.output}")
    
    # Print statistics
    label_counts = defaultdict(int)
    for item in unique_data:
        label = item.get("label_sentiment") or item.get("label")
        if label:
            label_counts[label] += 1
    
    print(f"\n📊 Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count} ({count*100/len(unique_data):.1f}%)")


if __name__ == "__main__":
    main()
