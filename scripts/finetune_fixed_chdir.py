#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correction du problème d'import correlation_mapping
===================================================

Problème : os.chdir(model_dir) fait perdre l'accès aux modules config/

Solution : Supprimer os.chdir() et utiliser des chemins absolus
"""

# À remplacer dans finetune.py autour de la ligne 300-320 :

def __init__(self, model_name: str, max_length: int, incremental_mode: bool = False, 
             baseline_model: str = None, target_column: str = "label", 
             class_balancing: str = None, mode: str = "production"):
    # ... code existant ...
    
    if self.hub_id and self.mode in ["production", "development"]:
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logger.error("❌ HF_TOKEN non trouvé dans l'environnement")
                self.repo = None
                self.repo_dir = None
                return
            
            # Clone (ou update) le repo dans un dossier local
            self.repo_dir = Path(f"./hf-{self.task_type}-{self.mode}")
            
            # CORRECTION : Ne PAS faire os.chdir() !
            # os.chdir(self.repo_dir)  # <-- SUPPRIMER CETTE LIGNE
            
            # Vérifier si le repo existe déjà sur HF
            hf_api = HfApi()
            try:
                hf_api.repo_info(self.hub_id)
                repo_exists = True
                logger.info(f"📦 Repo HuggingFace existant: {self.hub_id}")
            except:
                repo_exists = False
                logger.info(f"📦 Création du repo HuggingFace: {self.hub_id}")
                
            if not repo_exists:
                create_repo(
                    self.hub_id,
                    token=hf_token,
                    private=False,
                    exist_ok=True
                )
            
            # Cloner ou pull le repo
            if self.repo_dir.exists():
                self.repo = Repository(
                    local_dir=self.repo_dir,
                    clone_from=self.hub_id,
                    token=hf_token,
                    skip_lfs_files=True
                )
                try:
                    self.repo.git_pull()
                    logger.info(f"🔄 Repo mis à jour: {self.repo_dir.resolve()}")
                except Exception as e:
                    logger.warning(f"⚠️ Pull impossible (pas grave): {e}")
            else:
                self.repo = Repository(
                    local_dir=self.repo_dir,
                    clone_from=self.hub_id,
                    token=hf_token,
                    skip_lfs_files=True
                )
                logger.info(f"📥 Repo cloné: {self.repo_dir.resolve()}")
            
            self._setup_gitignore()
            
        except Exception as e:
            logger.warning(f"⚠️ Impossible de cloner le repo {self.hub_id}: {e}")
            logger.info("🔧 Le modèle sera sauvé localement uniquement")
            self.repo = None
            self.repo_dir = None


# Et aussi dans la méthode train(), remplacer :

def train(self, ds: DatasetDict, args: argparse.Namespace, test_ds: Dataset = None):
    # ... code existant ...
    
    # CORRECTION : Utiliser repo_dir directement sans chdir
    output_dir = str(self.repo_dir) if self.repo_dir else args.output_dir
    logger.info(f"📂 Répertoire d'entraînement: {output_dir}")
    
    targs = TrainingArguments(
        output_dir=output_dir,
        # ... reste des arguments ...
        logging_dir=os.path.join(output_dir, "logs"),
        # ... reste des arguments ...
    )
    
    # ... reste du code ...
    
    # CORRECTION : Sauvegarder avec chemin complet
    save_dir = self.repo_dir if self.repo_dir else Path(args.output_dir)
    trainer.save_model(str(save_dir))  # Convertir en string
    self.tokenizer.save_pretrained(str(save_dir))
    logger.info(f"💾 Modèle sauvé dans: {save_dir}")
