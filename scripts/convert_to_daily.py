#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert collected news to daily JSONL format for incremental learning
This script bridges the existing collect_news.py with the incremental pipeline
"""
import os
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path


def calculate_importance(text):
    """
    Calculate importance level based on keyword impact
    Using the same weighted keywords from collect_news.py
    """
    if not text:
        return "general"
    
    text_lower = text.lower()
    
    # High impact keywords (×3 weight in original)
    high_impact = [
        'crash', 'surge', 'crisis', 'breakthrough', 'bankruptcy',
        'collapse', 'soar', 'plunge', 'boom', 'bust', 'scandal',
        'fraud', 'investigation', 'bailout', 'recession'
    ]
    
    # Medium impact keywords (×2 weight in original)
    medium_impact = [
        'gain', 'drop', 'strong', 'weak', 'rise', 'fall',
        'growth', 'decline', 'increase', 'decrease', 'profit',
        'loss', 'beat', 'miss', 'upgrade', 'downgrade'
    ]
    
    # Check for high impact
    if any(keyword in text_lower for keyword in high_impact):
        return "critical"
    
    # Check for medium impact
    if any(keyword in text_lower for keyword in medium_impact):
        return "important"
    
    return "general"


def csv_to_jsonl(csv_path, jsonl_path=None, date=None):
    """
    Convert CSV from collect_news.py to JSONL format for incremental training
    
    Args:
        csv_path: Path to input CSV file
        jsonl_path: Path to output JSONL file (auto-generated if None)
        date: Date string for the data (auto-detected if None)
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return None
    
    # Auto-generate output path if not provided
    if jsonl_path is None:
        # Try to extract date from filename (e.g., news_20250107.csv)
        if date is None:
            filename = csv_path.stem
            if filename.startswith("news_") and len(filename) >= 13:
                date = filename[5:13]  # Extract YYYYMMDD
                # Convert to YYYY-MM-DD format
                try:
                    date_obj = datetime.strptime(date, "%Y%m%d")
                    date = date_obj.strftime("%Y-%m-%d")
                except:
                    date = datetime.now().strftime("%Y-%m-%d")
            else:
                date = datetime.now().strftime("%Y-%m-%d")
        
        # Create output path in daily directory
        jsonl_path = Path(f"datasets/daily/{date}.jsonl")
    else:
        jsonl_path = Path(jsonl_path)
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
    
    # Create output directory
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read CSV and convert to JSONL
    rows_written = 0
    label_counts = {"positive": 0, "negative": 0, "neutral": 0}
    importance_counts = {"general": 0, "important": 0, "critical": 0}
    
    with open(csv_path, 'r', encoding='utf-8') as csv_file, \
         open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            text = row.get('text', '').strip()
            label = row.get('label', '').lower()
            
            if not text or label not in ['positive', 'negative', 'neutral']:
                continue
            
            # Calculate importance
            importance = calculate_importance(text)
            
            # Create JSONL entry
            json_entry = {
                'text': text,
                'label_sentiment': label,
                'label_importance': importance,
                'url': row.get('url', ''),
                'title': row.get('title', ''),
                'source': row.get('source', ''),
                'published_at': row.get('published_at', ''),
                'date': date
            }
            
            # Remove empty fields
            json_entry = {k: v for k, v in json_entry.items() if v}
            
            # Write to JSONL
            jsonl_file.write(json.dumps(json_entry, ensure_ascii=False) + '\n')
            rows_written += 1
            
            # Update counts
            label_counts[label] += 1
            importance_counts[importance] += 1
    
    print(f"✅ Conversion complete!")
    print(f"   Input:  {csv_path}")
    print(f"   Output: {jsonl_path}")
    print(f"   Rows:   {rows_written}")
    print(f"   Date:   {date}")
    print(f"📊 Sentiment distribution: {label_counts}")
    print(f"📊 Importance distribution: {importance_counts}")
    
    return jsonl_path


def batch_convert(input_dir="datasets", output_dir="datasets/daily"):
    """
    Convert all CSV files in a directory to JSONL format
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(input_path.glob("news_*.csv")) + list(input_path.glob("financial_news_*.csv"))
    
    if not csv_files:
        print(f"⚠️ No CSV files found in {input_dir}")
        return
    
    print(f"📁 Found {len(csv_files)} CSV files to convert")
    
    for csv_file in sorted(csv_files):
        print(f"\n📄 Processing: {csv_file.name}")
        csv_to_jsonl(csv_file)
    
    print(f"\n✅ Batch conversion complete! Processed {len(csv_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Convert collected news CSV to JSONL format for incremental learning"
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        help="Input CSV file or directory (default: latest news_*.csv)"
    )
    
    parser.add_argument(
        "--output",
        help="Output JSONL file path (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--date",
        help="Date for the data in YYYY-MM-DD format (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Convert all CSV files in the datasets directory"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        batch_convert()
    else:
        # Find input file
        if args.input:
            input_path = args.input
        else:
            # Auto-detect latest news CSV
            from scripts.utils import latest_dataset
            try:
                input_path = latest_dataset()
                print(f"📂 Auto-selected: {input_path}")
            except:
                print("❌ No dataset found. Please specify input file.")
                return
        
        # Convert
        csv_to_jsonl(input_path, args.output, args.date)


if __name__ == "__main__":
    main()
