
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Criando lista de parametros para serem testados

n_neighbors_values = [2, 5, 10, 15, 20]
min_dist_values = [0.01, 0.1, 0.3, 0.5]

for n_neighbors in n_neighbors_values:
    for min_dist in min_dist_values:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        embedding = reducer.fit_transform(df)

        plt.figure(figsize=(6, 4))
        plt.scatter(embedding[:, 0], embedding[:, 1])
        plt.title(f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})')
        plt.xlabel('UMAP dimensão 1')
        plt.ylabel('UMAP dimensão 2')
        plt.show()