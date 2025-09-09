#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay Buffer for Incremental Learning
Stratified sampling from history + daily data deduplication
"""
import os
import json
import argparse
import random
import hashlib
from datetime import datetime

SENT = ["negative", "neutral", "positive"]
IMP = ["general", "important", "critical"]


def load_jsonl(p):
    """Load JSONL file"""
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def dump_jsonl(p, rows):
    """Save JSONL file"""
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def hash_key(r):
    """Generate unique hash for deduplication"""
    base = (r.get("url") or "") + "||" + (r.get("title") or "") + "||" + (r.get("text") or "")
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def dedup(rows):
    """Remove duplicates based on content hash"""
    seen, out = set(), []
    for r in rows:
        k = hash_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def stratified_sample(history, replay_size=800):
    """Stratified sampling by (sentiment, importance) pairs"""
    # Create buckets by (sentiment, importance)
    buckets = {}
    for r in history:
        s = r.get("label_sentiment")
        i = r.get("label_importance")
        if s in SENT and (i in IMP or i is None):
            buckets.setdefault((s, i), []).append(r)
    
    items = []
    if not buckets:
        random.shuffle(history)
        return history[:replay_size]
    
    # Sample equally from each bucket
    per = max(1, replay_size // len(buckets))
    for _, arr in buckets.items():
        random.shuffle(arr)
        items.extend(arr[:per])
    
    return items[:replay_size]


def main():
    ap = argparse.ArgumentParser(description="Prepare replay buffer for incremental learning")
    ap.add_argument("--daily", default="datasets/daily/latest.jsonl",
                    help="Path to daily data file")
    ap.add_argument("--history", default="datasets/history.jsonl",
                    help="Path to historical data file")
    ap.add_argument("--out", default="datasets/combined.jsonl",
                    help="Output combined dataset path")
    ap.add_argument("--replay_size", type=int, default=800,
                    help="Number of historical samples to include")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility")
    args = ap.parse_args()
    
    if args.seed:
        random.seed(args.seed)
    
    # Load data
    daily = load_jsonl(args.daily)
    history = load_jsonl(args.history)
    
    print(f"📥 Loaded: daily={len(daily)}, history={len(history)}")
    
    # Filter valid samples
    def valid(r):
        t = r.get("text", "").strip()
        s = r.get("label_sentiment") or r.get("label")  # Support both formats
        return bool(t and s in SENT)
    
    daily = [r for r in daily if valid(r)]
    history = [r for r in history if valid(r)]
    
    # Normalize label field
    for r in daily + history:
        if "label" in r and "label_sentiment" not in r:
            r["label_sentiment"] = r["label"]
    
    # Deduplicate
    daily = dedup(daily)
    history = dedup(history)
    
    # Sample from history
    replay = stratified_sample(history, args.replay_size) if history else []
    combined = daily + replay
    random.shuffle(combined)
    
    # Write combined dataset
    dump_jsonl(args.out, combined)
    
    # Update history (append-only with deduplication)
    hist_map = {hash_key(r): r for r in history}
    for r in daily:
        hist_map[hash_key(r)] = r
    dump_jsonl(args.history, list(hist_map.values()))
    
    print(f"✅ Combined dataset created:")
    print(f"   - Total: {len(combined)} samples")
    print(f"   - Daily: {len(daily)} samples")
    print(f"   - Replay: {len(replay)} samples")
    print(f"   - History updated: {len(hist_map)} total samples")
    
    # Statistics
    if combined:
        sent_dist = {}
        imp_dist = {}
        for r in combined:
            s = r.get("label_sentiment")
            i = r.get("label_importance", "N/A")
            sent_dist[s] = sent_dist.get(s, 0) + 1
            imp_dist[i] = imp_dist.get(i, 0) + 1
        
        print(f"📊 Sentiment distribution: {sent_dist}")
        print(f"📊 Importance distribution: {imp_dist}")


if __name__ == "__main__":
    main()
