pip install -r requirements.txt

# Export Embeddings From Pretrained Models
python export_original_embeddings.py

# Align Spanish Embeddings onto English Embeddings
python align_embeddings.py

# English Performance Baseline
python eval_xnli.py "mono" "orig" "en"
# Spanish Upper Bound Supervised
python eval_xnli.py "mono" "orig" "es"
# Spanish Lower Bound (Zero-Shot Monolingual English -> Spanish)
python eval_xnli.py "wrong" "orig" "es"
# Multilingual Spanish Baseline (Compare to Our Proposed Method)
python eval_xnli.py "multi" "orig" "es"
# Aligned Model (English MHA Weights, Spanish Aligned Embedding Matrix)
python eval_xnli.py "align" "align" "es"

