# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#%%

df = pd.read_csv(r"C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\dados_semnulo.csv")

# %%
df.head()
# %%
df.info()
# %%
df.describe()
# %%

# Contagem de valores únicos por coluna
valores_unicos = df.nunique()

# Contagem de frequência para cada variável
frequencias = {coluna: df[coluna].value_counts() for coluna in df.columns}

valores_unicos, {col: freq.head(5) for col, freq in frequencias.items()}  # Exibindo apenas as 5 categorias mais frequentes de cada variável

# %% 

# Configuração para histogramas de todas as variáveis categóricas
fig, axes = plt.subplots(nrows=11, ncols=2, figsize=(16, 40))
axes = axes.flatten()

for i, coluna in enumerate(df.columns):
    df[coluna].value_counts().sort_index().plot(
        kind='bar', ax=axes[i], title=coluna, color='skyblue', edgecolor='black'
    )
    axes[i].set_ylabel('Frequência')
    axes[i].set_xlabel('Categorias')

# Ajustar o layout
plt.tight_layout()
plt.show()

# %% 

# Utilizando a PCA para plotagem 2D
pca = PCA(n_components=2)
components = pca.fit_transform(df)

plt.scatter(components[:, 0], components[:, 1])
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA dos Dados')
plt.show()
