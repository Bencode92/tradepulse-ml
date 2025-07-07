#!/bin/bash
# setup-fixed-model.sh
# Configure TradePulse pour utiliser UN SEUL modèle avec nom fixe

set -euo pipefail

echo "🔧 Configuration Modèle Fixe TradePulse"
echo "======================================"

WORKFLOW_FILE=".github/workflows/finetune-model.yml"
FIXED_MODEL_NAME="tradepulse-finbert-production"
HF_MODEL_ID="Bencode92/$FIXED_MODEL_NAME"

echo "📦 Nom de modèle fixe: $FIXED_MODEL_NAME"
echo "🤗 HuggingFace ID: $HF_MODEL_ID"

# 1. Backup du workflow
cp "$WORKFLOW_FILE" "${WORKFLOW_FILE}.backup"
echo "💾 Backup créé: ${WORKFLOW_FILE}.backup"

# 2. Modification pour nom fixe
echo "🔧 Modification du workflow..."

# Remplacer la génération dynamique par nom fixe
sed -i 's/TIMESTAMP=$(date +%Y%m%d_%H%M%S)/#TIMESTAMP=$(date +%Y%m%d_%H%M%S) # Désactivé pour nom fixe/' "$WORKFLOW_FILE"
sed -i 's/MODEL_NAME_CLEAN=$(echo "$MODEL_NAME" | sed '\''s\/[^a-zA-Z0-9]\/-\/g'\'')/#MODEL_NAME_CLEAN=$(echo "$MODEL_NAME" | sed '\''s\/[^a-zA-Z0-9]\/-\/g'\'') # Désactivé/' "$WORKFLOW_FILE"
sed -i "s/UNIQUE_MODEL_NAME=\"finbert-\${MODEL_NAME_CLEAN}-\${TIMESTAMP}\"/UNIQUE_MODEL_NAME=\"$FIXED_MODEL_NAME\"/" "$WORKFLOW_FILE"

# Remplacer HF_MODEL_ID pour nom fixe
sed -i "s/HF_MODEL_ID=\"Bencode92\/tradepulse-\$UNIQUE_MODEL_NAME\"/HF_MODEL_ID=\"$HF_MODEL_ID\"/" "$WORKFLOW_FILE"

echo "✅ Workflow modifié pour nom fixe"

# 3. Créer fichier de configuration pour votre plateforme
mkdir -p config

cat > config/stock_platform_config.py << EOF
"""
Configuration Stock Analysis Platform - TradePulse
Modèle fixe pour connexion automatique permanente
"""

# 🤖 MODÈLE FIXE - Plus besoin de changer !
MODEL_ID = "$HF_MODEL_ID"
MODEL_NAME = "$FIXED_MODEL_NAME"

# Configuration automatique
AUTO_UPDATE_ENABLED = True
FALLBACK_MODEL = "yiyanghkust/finbert-tone"

def get_model_config():
    """Retourne la configuration du modèle"""
    return {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "auto_update": AUTO_UPDATE_ENABLED,
        "fallback": FALLBACK_MODEL
    }

print(f"🤖 Modèle configuré: {MODEL_ID}")
EOF

# 4. Créer script d'intégration simple
cat > scripts/connect_stock_platform.py << 'EOF'
#!/usr/bin/env python3
"""
Script simple pour connecter Stock Analysis Platform
Nom de modèle FIXE = connexion automatique permanente
"""

import sys
import os
sys.path.append('config')

try:
    from stock_platform_config import MODEL_ID, MODEL_NAME
    from transformers import pipeline
    
    print(f"🤖 Connexion au modèle: {MODEL_ID}")
    
    # Charger le modèle (toujours le même nom)
    classifier = pipeline("text-classification", model=MODEL_ID)
    print("✅ Modèle connecté avec succès !")
    
    # Test rapide
    test_text = "Apple reported strong quarterly earnings with record revenue"
    result = classifier(test_text)
    print(f"🧪 Test: {result[0]['label']} ({result[0]['score']:.3f})")
    
    # Sauvegarder info connexion
    import json
    from datetime import datetime
    
    connection_info = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "connected_at": datetime.now().isoformat(),
        "status": "connected",
        "test_result": result[0]
    }
    
    with open("config/last_connection.json", "w") as f:
        json.dump(connection_info, f, indent=2)
    
    print("📁 Info connexion sauvée: config/last_connection.json")
    
except Exception as e:
    print(f"❌ Erreur connexion: {e}")
    print("🔄 Tentative avec modèle fallback...")
    
    try:
        from stock_platform_config import FALLBACK_MODEL
        classifier = pipeline("text-classification", model=FALLBACK_MODEL)
        print(f"✅ Connecté au modèle fallback: {FALLBACK_MODEL}")
    except Exception as e2:
        print(f"❌ Erreur fallback: {e2}")
        sys.exit(1)
EOF

chmod +x scripts/connect_stock_platform.py

# 5. Créer template pour votre plateforme
cat > template_stock_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
Template Stock Analysis Platform avec TradePulse
MODÈLE FIXE = Connexion automatique permanente !
"""

import sys
import os
sys.path.append('config')

from stock_platform_config import MODEL_ID
from transformers import pipeline
import pandas as pd
from datetime import datetime

class StockAnalysisPlatform:
    """Plateforme d'analyse avec TradePulse intégré"""
    
    def __init__(self):
        # 🤖 MODÈLE TOUJOURS LE MÊME = Connexion automatique !
        self.model_id = MODEL_ID
        self.classifier = None
        self.connect_model()
    
    def connect_model(self):
        """Connecte au modèle TradePulse (nom fixe)"""
        try:
            print(f"🤖 Connexion modèle: {self.model_id}")
            self.classifier = pipeline("text-classification", model=self.model_id)
            print("✅ TradePulse connecté automatiquement !")
        except Exception as e:
            print(f"❌ Erreur: {e}")
            # Fallback
            print("🔄 Fallback...")
            self.classifier = pipeline("text-classification", model="yiyanghkust/finbert-tone")
    
    def analyze_stock_news(self, symbol, news_list):
        """Analyse sentiment pour un stock"""
        if not self.classifier:
            return {"error": "Modèle non connecté"}
        
        results = []
        for news in news_list:
            prediction = self.classifier(news)
            results.append({
                "text": news,
                "sentiment": prediction[0]["label"],
                "confidence": prediction[0]["score"]
            })
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model_id,
            "results": results,
            "summary": self._calculate_summary(results)
        }
    
    def _calculate_summary(self, results):
        """Calcule résumé global"""
        sentiments = [r["sentiment"] for r in results]
        confidences = [r["confidence"] for r in results]
        
        # Sentiment majoritaire
        from collections import Counter
        sentiment_counts = Counter(sentiments)
        overall_sentiment = sentiment_counts.most_common(1)[0][0]
        
        return {
            "overall_sentiment": overall_sentiment,
            "avg_confidence": sum(confidences) / len(confidences),
            "distribution": dict(sentiment_counts),
            "total_analyzed": len(results)
        }

# Usage
if __name__ == "__main__":
    # Initialiser plateforme
    platform = StockAnalysisPlatform()
    
    # Test avec AAPL
    test_news = [
        "Apple reports record quarterly earnings beating analyst expectations",
        "iPhone sales surge in China market showing strong demand",
        "Apple stock rises on positive earnings guidance"
    ]
    
    result = platform.analyze_stock_news("AAPL", test_news)
    
    print("\n📊 Résultat Analyse:")
    print(f"Stock: {result['symbol']}")
    print(f"Sentiment global: {result['summary']['overall_sentiment']}")
    print(f"Confiance moyenne: {result['summary']['avg_confidence']:.3f}")
    print(f"Distribution: {result['summary']['distribution']}")
    print(f"Modèle utilisé: {result['model_used']}")
EOF

echo ""
echo "🎉 Configuration terminée !"
echo "========================="
echo "📦 Nom fixe: $FIXED_MODEL_NAME"
echo "🤗 HuggingFace: $HF_MODEL_ID"
echo "📁 Config: config/stock_platform_config.py"
echo "🐍 Script test: scripts/connect_stock_platform.py"
echo "📄 Template: template_stock_analysis.py"
echo ""
echo "🔄 Prochaines étapes:"
echo "1. Commitez ces changements"
echo "2. Relancez un réentraînement"
echo "3. Le modèle aura le nom fixe sur HuggingFace"
echo "4. Votre plateforme se connecte automatiquement !"
echo ""
echo "💡 AVANTAGE: Plus besoin de changer le nom dans votre code !"
echo "   Le modèle s'auto-met à jour à chaque réentraînement"
