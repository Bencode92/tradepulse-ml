#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TradePulse FinBERT Fine-tuning Script
Fine-tuning FinBERT pour l'analyse de sentiment financier avec données propriétaires
"""

import os
import json
import pandas as pd
import yaml
from datetime import datetime
import argparse
import logging
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradePulseFineTuner:
    def __init__(self, config_path="config/training_config.yaml"):
        """Initialize le fine-tuner avec la configuration"""
        self.config = self.load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.datasets = None
        
    def load_config(self, config_path):
        """Charge la configuration depuis le fichier YAML"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_data(self):
        """Charge et prépare les données d'entraînement"""
        logger.info("📊 Chargement des données d'entraînement...")
        
        # Chargement des fichiers CSV
        train_df = pd.read_csv(self.config['data']['train_file'])
        eval_df = pd.read_csv(self.config['data']['eval_file'])
        
        # Nettoyage des données
        train_df = train_df.dropna().reset_index(drop=True)
        eval_df = eval_df.dropna().reset_index(drop=True)
        
        # Mapping des labels
        label2id = self.config['labels']['label2id']
        train_df['labels'] = train_df[self.config['data']['label_column']].map(label2id)
        eval_df['labels'] = eval_df[self.config['data']['label_column']].map(label2id)
        
        # Conversion en datasets Hugging Face
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        
        self.datasets = DatasetDict({
            'train': train_dataset,
            'eval': eval_dataset
        })
        
        logger.info(f"✅ Données chargées: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        # Affichage de la distribution des classes
        train_dist = train_df['labels'].value_counts().sort_index()
        logger.info(f"📈 Distribution train: {dict(train_dist)}")
        
        return self.datasets
    
    def load_model_and_tokenizer(self):
        """Charge le modèle de base et le tokenizer"""
        logger.info(f"🤖 Chargement du modèle: {self.config['model']['base_model']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['base_model'])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model']['base_model'],
            num_labels=self.config['model']['num_labels'],
            problem_type=self.config['model']['problem_type'],
            id2label=self.config['labels']['id2label'],
            label2id=self.config['labels']['label2id']
        )
        
        logger.info("✅ Modèle et tokenizer chargés")
        
    def tokenize_data(self):
        """Tokenise les données"""
        logger.info("🔤 Tokenisation des données...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.config['data']['text_column']],
                truncation=True,
                padding=False,
                max_length=self.config['data']['max_length']
            )
        
        self.datasets = self.datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=self.datasets['train'].column_names
        )
        
        logger.info("✅ Tokenisation terminée")
        
    def compute_metrics(self, eval_pred):
        """Calcule les métriques d'évaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1
        }
    
    def train(self, output_dir=None, run_name=None):
        """Lance l'entraînement"""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{self.config['output']['output_dir']}/finbert-sentiment-{timestamp}"
        
        if run_name is None:
            run_name = f"{self.config['output']['run_name']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🚀 Début de l'entraînement: {run_name}")
        logger.info(f"📁 Dossier de sortie: {output_dir}")
        
        # Configuration de l'entraînement
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config['training']['learning_rate'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            evaluation_strategy=self.config['training']['evaluation_strategy'],
            eval_steps=self.config['training']['eval_steps'],
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            logging_dir=f"{self.config['output']['logging_dir']}/{run_name}",
            run_name=run_name,
            report_to=None,  # Désactiver wandb par défaut
            save_total_limit=2,
            dataloader_pin_memory=False
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets['eval'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Entraînement
        train_result = trainer.train()
        
        # Sauvegarde du modèle final
        trainer.save_model()
        trainer.save_state()
        
        # Évaluation finale
        eval_result = trainer.evaluate()
        
        # Sauvegarde des métriques
        self.save_training_results(output_dir, train_result, eval_result)
        
        logger.info("✅ Entraînement terminé!")
        logger.info(f"📊 Résultats finaux: {eval_result}")
        
        return output_dir, eval_result
    
    def save_training_results(self, output_dir, train_result, eval_result):
        """Sauvegarde les résultats d'entraînement"""
        results = {
            "training_config": self.config,
            "train_result": {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics['train_runtime'],
                "train_samples_per_second": train_result.metrics['train_samples_per_second']
            },
            "eval_result": eval_result,
            "model_info": {
                "base_model": self.config['model']['base_model'],
                "num_labels": self.config['model']['num_labels'],
                "training_date": datetime.now().isoformat()
            }
        }
        
        # Sauvegarde JSON
        with open(f"{output_dir}/training_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Sauvegarde des métriques séparément
        with open(f"{output_dir}/metrics.json", 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, indent=2)
        
        logger.info(f"💾 Résultats sauvegardés dans {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="TradePulse FinBERT Fine-tuning")
    parser.add_argument("--config", default="config/training_config.yaml", help="Fichier de configuration")
    parser.add_argument("--output_dir", help="Dossier de sortie")
    parser.add_argument("--run_name", help="Nom du run")
    parser.add_argument("--train_file", help="Fichier d'entraînement CSV")
    parser.add_argument("--eval_file", help="Fichier d'évaluation CSV")
    
    args = parser.parse_args()
    
    # Initialisation
    fine_tuner = TradePulseFineTuner(args.config)
    
    # Override des fichiers si spécifiés
    if args.train_file:
        fine_tuner.config['data']['train_file'] = args.train_file
    if args.eval_file:
        fine_tuner.config['data']['eval_file'] = args.eval_file
    
    # Pipeline complet
    try:
        # 1. Chargement des données
        fine_tuner.load_data()
        
        # 2. Chargement du modèle
        fine_tuner.load_model_and_tokenizer()
        
        # 3. Tokenisation
        fine_tuner.tokenize_data()
        
        # 4. Entraînement
        output_dir, results = fine_tuner.train(
            output_dir=args.output_dir,
            run_name=args.run_name
        )
        
        logger.info(f"🎉 Fine-tuning terminé! Modèle sauvegardé dans: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()
