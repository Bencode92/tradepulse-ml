#!/bin/bash
# Patch pour corriger l'erreur de dimensions multi-label dans finetune.py

echo "🔧 Application du patch ignore_mismatched_sizes..."

# Localiser les lignes à modifier dans finetune.py
sed -i '
# Pour la branche incremental multi-label
/self\.is_multi_label:/{
    n
    /self\.model = AutoModelForSequenceClassification\.from_pretrained(/{
        :a
        /)/!{
            N
            ba
        }
        s/problem_type="multi_label_classification"/problem_type="multi_label_classification",\n                        ignore_mismatched_sizes=True/
    }
}

# Pour la branche modèle de base multi-label
/# Mode classique/,/logger\.info.*Model & tokenizer loaded/{
    /self\.is_multi_label:/{
        n
        /self\.model = AutoModelForSequenceClassification\.from_pretrained(/{
            :b
            /)/!{
                N
                bb
            }
            s/problem_type="multi_label_classification"/problem_type="multi_label_classification",\n                    ignore_mismatched_sizes=True/
        }
    }
}
' scripts/finetune.py

echo "✅ Patch appliqué!"
echo ""
echo "Vérification:"
grep -A5 "problem_type=\"multi_label_classification\"" scripts/finetune.py | head -20
