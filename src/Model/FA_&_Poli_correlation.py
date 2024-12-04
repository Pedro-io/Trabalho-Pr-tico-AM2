# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import bartlett
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis # Analise fatorial 
from sklearn.preprocessing import MinMaxScaler # Normalização dos fatores 
import pingouin as pg # Para o calculo da Correlação Policórica para as variavéis ordinais 
import umap







# %%
# Configurando e Carregando nossos dados
# Permite visualizar todas as colunas do DataFrame
pd.set_option('display.max_columns', None)

# Leitura dos dados
df_original = pd.read_csv(r"C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\base_discretizada.csv")
df_original.head()








# %%
df = df_original.drop(columns= ['Unnamed: 0', 'diagnostico_hipertensao'])

# %%
df.head()







# %%
df.columns





# %% [markdown]
# ### Correlação Policórica

# %%
corr_policorica = pg.pcorr(df)
corr_policorica_matrix = corr_policorica.to_numpy()





# %%
corr_policorica





# %%
plt.figure(figsize=(10, 8)) 
sns.heatmap(corr_policorica, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

plt.title("Matriz de Correlação Policórica")
plt.show()

# %% [markdown]
# ## Aplicando a Analise Fatorial sobre a matriz de correlação policórica

# %% [markdown]
# >Primeiro vamos realizar um teste aleátorio aplicando para 2 fatores

# %%
fa = FactorAnalysis(n_components=2, random_state=42)  # Reduzindo para 2 fatores
fatores = fa.fit_transform(corr_policorica_matrix)

# %%
#Plotando os Fatores 
plt.figure(figsize=(8, 6))
plt.scatter(fatores[:, 0], fatores[:, 1], alpha=0.7)

# Configuração do gráfico
plt.title("Análise Fatorial - Componentes 1 e 2")
plt.xlabel("Fator 1")
plt.ylabel("Fator 2")
plt.grid(True)
plt.show()

# %% [markdown]
# ### Calculando o KMO e Bartlett para varificar a possibilidade de utilizar a análise de fatores

# %%
# Calcular o KMO
kmo_all, kmo_model = calculate_kmo(df)
print(f'KMO: {kmo_model}')

# %%
# Teste de Esfericidade de Bartlett
# Verificando a matriz de correlação
corr_matrix = np.corrcoef(df, rowvar=False)

# %%
# Realizando o teste de Bartlett
chi2_stat, p_value = bartlett(*[df.iloc[:, i] for i in range(df.shape[1])])
print(f"Teste de Bartlett - Chi2: {chi2_stat}, p-value: {p_value}")

# Interpretação dos resultados
if p_value < 0.05:
    print("O Teste de Bartlett indica que os dados são adequados para FA.")
else:
    print("O Teste de Bartlett não indica adequação para FA.")

# %% [markdown]
# ## Aplicando o Método do Cotovelo e o Parallel Analysis para verificar o numero ideal de fatores

# %% [markdown]
# > Método do Cotovelo
# 

# %%
# Aplicando FA com o número máximo de fatores
fa = FactorAnalyzer(n_factors=df.shape[1], rotation='varimax')
fa.fit(df)

# Obtendo os eigenvalues
eigenvalues = fa.get_eigenvalues()

# %%
# Plotando a variância explicada acumulada
explained_variance = eigenvalues / sum(eigenvalues)  # Variância explicada por cada fator
cumulative_variance = explained_variance.cumsum()  # Variância acumulada

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Análise Elbow - Variância Acumulada')
plt.xlabel('Número de Fatores')
plt.ylabel('Variância Explicada Acumulada')
plt.grid(True)
plt.show()

# %% [markdown]
# > Parallel Analysis

# %%
# Função para Parallel Analysis
def parallel_analysis(real_data, n_iterations=100):
    # Número de fatores/fatores que queremos manter (com base nos dados reais)
    n_fatores = real_data.shape[1]
    
    # Realizando a Análise PCA nos dados reais
    pca = PCA(n_components=n_fatores)
    pca.fit(real_data)
    
    # Eigenvalues dos dados reais
    eigenvalues_real = pca.explained_variance_

    # Gerando dados aleatórios com a mesma forma dos dados reais
    eigenvalues_random = np.zeros((n_iterations, n_fatores))
    for i in range(n_iterations):
        random_data = np.random.normal(size=real_data.shape)
        pca.fit(random_data)
        eigenvalues_random[i, :] = pca.explained_variance_
    
    # Calculando o valor médio dos eigenvalues dos dados aleatórios
    mean_eigenvalues_random = np.mean(eigenvalues_random, axis=0)
    
    return eigenvalues_real, mean_eigenvalues_random

# %%
# Aplicar a Parallel Analysis
eigenvalues_real, mean_eigenvalues_random = parallel_analysis(df)

# %%
# Plotando os eigenvalues reais vs aleatórios
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(eigenvalues_real) + 1), eigenvalues_real, marker='o', label="Eigenvalues Reais")
plt.plot(range(1, len(mean_eigenvalues_random) + 1), mean_eigenvalues_random, marker='x', label="Eigenvalues Aleatórios", linestyle='--')
plt.title('Análise Parallel - Eigenvalues Reais vs Aleatórios')
plt.xlabel('Número de Fatores')
plt.ylabel('Eigenvalue')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Verificando o número de fatores a serem mantidos
num_fatores_ideais = np.sum(eigenvalues_real > mean_eigenvalues_random)
print(f'O número ideal de fatores é: {num_fatores_ideais}')

# %% [markdown]
# ### Analise de Fatores
# Iremos aplicar a Analise de fatores utilizando o numero de fatores determinado pelo Parallel Analysis por ser um metodo mais robusto. 

# ### Aplicando analise de fatores utilizando resultados do método Parallel Analysis

# %%
# Aplicar a análise fatorial com 13 fatores
fa = FactorAnalyzer(n_factors=13, rotation="varimax", method="minres")
fa.fit(df)

# %%
# Mostrar a variância explicada por cada fator (eigenvalues)
print("Eigenvalues (variância explicada por cada fator):")
print(fa.get_eigenvalues())

# %%
# Obter as cargas dos fatores (fatores x variáveis)
fatores = fa.loadings_
print("Cargas dos fatores:")
print(fatores)

# %%
# Mostrar a variância explicada total
variancia_explicada = fa.get_factor_variance()
print("Variância explicada por cada fator e variância acumulada:")
print(variancia_explicada)


# %%
# Obter as cargas fatoriais
loadings = fa.loadings_

# Visualizar as cargas fatoriais
loadings_df = pd.DataFrame(loadings, columns=[f'Fator_{i+1}' for i in range(loadings.shape[1])], index=df.columns)
print(loadings_df)

# %%
# Número de fatores
num_fatores = loadings_df.shape[1]

# Gerar gráficos separados para cada fator
for i in range(num_fatores):
    plt.figure(figsize=(10, 6))
    plt.bar(loadings_df.index, loadings_df.iloc[:, i])
    plt.title(f'Cargas Fatoriais - Fator {i+1}')
    plt.ylabel('Carga Fatorial')
    plt.xlabel('Variáveis')
    plt.xticks(rotation=90)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Ajustar o layout para evitar cortes nas labels
    plt.show()

# %% [markdown]
# ## Salvando Fatores em um DF

# %%
# Obtendo as pontuações dos fatores (as variáveis latentes para cada observação)
fatores = fa.transform(df)

# %%
# Convertendo os fatores em um DataFrame para facilitar o salvamento
fatores_df = pd.DataFrame(fatores, columns=[f'Fator_{i+1}' for i in range(fatores.shape[1])])

# %%
fatores_df = fatores_df.iloc[:, :13] 
fatores_df.head()


# %% [markdown]
# #### Visualização de Clusters com UMAP


# %% normalizando 
# objeto MinMaxScaler
scaler = MinMaxScaler()

# Aplicando normalização 
fatores_df_normalizado = pd.DataFrame(scaler.fit_transform(fatores_df), columns=fatores_df.columns)

# %% Fatores sem normalização
fatores_df['diagnostico_hipertensao'] = pd.Series(dtype='object') 
fatores_df['diagnostico_hipertensao'] = df_original['diagnostico_hipertensao'] 

# %% Fatores com normalização
fatores_df_normalizado['diagnostico_hipertensao'] = pd.Series(dtype='object') 
fatores_df_normalizado['diagnostico_hipertensao'] = df_original['diagnostico_hipertensao'] 

# %% UMAP
# Instanciando UMAP  
umap_2d = umap.UMAP(n_components=2, random_state=42)
# Reduzindo os fatores originais para 2D com UMAP
fatores_umap = umap_2d.fit_transform(fatores_df.drop('diagnostico_hipertensao', axis=1))  # Remove a coluna 'diagnostico_hipertensao' antes de aplicar o UMAP

# Reduzindo os fatores Normalizados para 2D com UMAP
fatores_umap_normalizado = umap_2d.fit_transform(fatores_df_normalizado.drop('diagnostico_hipertensao', axis=1))  # Remove a coluna 'diagnostico_hipertensao' antes de aplicar o UMAP

# %% Plotando graficos 2D com o UMAP
def plot_umap_distribuicao(data, umap_data, diagnostico_column, titulo, cores=None):
    """
    Plota a distribuição dos dados em 2D após redução de dimensionalidade com UMAP, 
    colorindo os pontos de acordo com uma coluna de diagnóstico.

    Args:
        data: DataFrame com os dados originais (incluindo a coluna de diagnóstico).
        umap_data: Array NumPy com os dados reduzidos pelo UMAP (2 dimensões).
        diagnostico_column: Nome da coluna no DataFrame que contém o diagnóstico.
        titulo: Título do gráfico.
        cores: Dicionário que mapeia valores únicos da coluna de diagnóstico para cores. 
               Se None, cores serão geradas automaticamente.
    """
    
    classes = data[diagnostico_column].unique()
    if cores is None:
        cores = {classe: plt.cm.get_cmap('viridis')(i/len(classes)) for i, classe in enumerate(classes)}
    elif len(cores) != len(classes):
        raise ValueError("Número de cores deve corresponder ao número de classes únicas.")


    plt.figure(figsize=(10, 8))
    for i, classe in enumerate(classes):
        indices = data[diagnostico_column] == classe
        plt.scatter(
            umap_data[indices, 0], 
            umap_data[indices, 1], 
            c=[cores[classe]], # Usa a cor pré-definida ou gerada
            label=f'Diagnóstico {classe}', 
            s=50, alpha=0.7
        )

    plt.title(titulo, fontsize=16)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


cores_especificas = {1: 'blue', 2: 'red'} 

plot_umap_distribuicao(fatores_df, fatores_umap, 'diagnostico_hipertensao', 'Distribuição dos Dados (Não Normalizados) usando UMAP com Diagnóstico de Hipertensão', cores_especificas)
plot_umap_distribuicao(fatores_df_normalizado, fatores_umap_normalizado, 'diagnostico_hipertensao', 'Distribuição dos Dados (Normalizados) usando UMAP com Diagnóstico de Hipertensão', cores_especificas)
# %% [markdown]
# > Salvando fatores em csv

# %%
fatores_df.to_csv(r'C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\base_fatores.csv')
fatores_df_normalizado.to_csv(r'C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\base_fatores_normalizados.csv')



