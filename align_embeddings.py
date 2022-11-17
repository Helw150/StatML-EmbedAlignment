import numpy as np
import pandas as pd
import torch


def load_embeddings(filename):
    df = pd.read_csv(filename, sep=",")
    # Remove Column Numbers
    np_array = df.values[:, 1:]
    return torch.tensor(np_array)


def align_embeddings(start, anchor):
    # PLACEHOLDER METHOD - THIS IS WHERE WE'll IMPLEMENT ALIGNMENT

    return torch.tensor(np.eye(768))


en_embed = load_embeddings("embedding_files/en_embeddings.csv")
print(en_embed.shape)
es_embed = load_embeddings("embedding_files/es_embeddings.csv")
print(es_embed.shape)

alignment_matrix = align_embeddings(es_embed, en_embed)
aligned_es_embed = torch.mm(es_embed, alignment_matrix)
# Sanity Check on the Embedding Matrix
assert torch.all(es_embed == aligned_es_embed)
out_np = aligned_es_embed.numpy()
out_df = pd.DataFrame(out_np)
out_df.to_csv("embedding_files/aligned_es_embeddings.txt")
