import numpy as np
import pandas as pd
import transformers

from transformers import AutoModelForSequenceClassification


def export_embed(lang):
    if lang == "en":
        model_url = "WillHeld/en-bert-xnli"
    else:
        model_url = "WillHeld/es-bert-xnli"
    model = AutoModelForSequenceClassification.from_pretrained(model_url)
    embeddings = model.base_model.embeddings.word_embeddings.weight
    out_np = embeddings.detach().numpy()
    out_df = pd.DataFrame(out_np)
    out_df.to_csv(f"embedding_files/{lang}_embeddings.csv")


export_embed("en")
export_embed("es")
