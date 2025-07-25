"""
Multi-label data collator for TradePulse ML
Ensures labels are float32 for BCEWithLogitsLoss
"""
from transformers import DataCollatorWithPadding
import torch


class MultiLabelCollator(DataCollatorWithPadding):
    """Data collator that ensures labels are float32 for multi-label classification"""
    
    def __call__(self, features):
        batch = super().__call__(features)
        
        # Force labels to float32 for BCEWithLogitsLoss compatibility
        if "labels" in batch:
            batch["labels"] = batch["labels"].to(torch.float32)
            
        return batch
