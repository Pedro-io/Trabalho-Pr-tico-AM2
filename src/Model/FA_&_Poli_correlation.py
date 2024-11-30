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
import pingouin as pg # Para o calculo da Correlação Policórica para as variavéis ordinais 
import umap

# %%
# Configurando e Carregando nossos dados
# Permite visualizar todas as colunas do DataFrame
pd.set_option('display.max_columns', None)

# Leitura dos dados
df = pd.read_csv(r"C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\base_discretizada.csv")
df.head()


# %%
df = df.drop(columns= ['Unnamed: 0'])

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
# ## Aplicando analise de fatores utilizando resultados do Método do Cotovelo

# %%
# Aplicar a análise fatorial com 21 fatores
fa_el = FactorAnalyzer(n_factors=21, rotation="varimax", method="minres")
fa_el.fit(df)

# %%
# Mostrar a variância explicada por cada fator (eigenvalues)
print("Eigenvalues (variância explicada por cada fator):")
print(fa_el.get_eigenvalues())

# %%
# Obter as cargas dos fatores (fatores x variáveis)
fatores = fa_el.loadings_
print("Cargas dos fatores:")
print(fatores)

# %%
# Mostrar a variância explicada total
variancia_explicada = fa_el.get_factor_variance()
print("Variância explicada por cada fator e variância acumulada:")
print(variancia_explicada)

# %% [markdown]
# ## Análise dos Resultados da Análise Fatorial com 21 Fatores
# 
# ### Eigenvalues (Variância Explicada por Cada Fator)
# Os eigenvalues representam a variância explicada por cada fator extraído durante a análise fatorial. Para os 21 fatores extraídos, a maior parte da variância é explicada pelos primeiros fatores, com valores de eigenvalue superiores a 1, como observado nos primeiros 5 fatores (valores como 2.47, 1.93, 1.62, 1.43 e 1.17). Esses fatores representam as dimensões principais da variabilidade dos dados.
# 
# Nos fatores subsequentes, a variância explicada diminui consideravelmente, com eigenvalues abaixo de 1, indicando que esses fatores explicam menos da variação total. A partir do 19º fator, os eigenvalues se aproximam de zero, sugerindo que esses fatores são menos relevantes para a explicação dos dados.
# 
# ### Cargas dos Fatores
# As cargas fatoriais indicam o grau de associação entre cada variável original e os fatores extraídos. Em geral, as cargas maiores (em módulo) indicam que as variáveis estão mais fortemente associadas a um fator específico. A seguir, são observados alguns pontos importantes sobre as cargas fatoriais:
# 
# - **Fator 1** tem várias cargas significativas, especialmente na variável associada à carga \( 0.4233 \), sugerindo que este fator pode ser relacionado a características centrais ou comuns entre várias variáveis.
# - **Fator 2** apresenta uma forte carga positiva para algumas variáveis, como \( 0.913 \), que indica que essas variáveis têm uma relação forte com esse fator.
# - **Fatores 3 a 5** também apresentam várias variáveis associadas a altas cargas, indicando que eles podem estar refletindo diferentes dimensões subjacentes do conjunto de dados.
#   
# Observando os fatores a partir do sexto, as cargas começam a diminuir, indicando uma menor contribuição dessas variáveis na formação dos fatores. Fatores mais altos apresentam muitas cargas próximas de zero, sugerindo que essas dimensões têm pouca relevância ou estão relacionadas a ruídos ou variações menores nos dados.
# 
# ### Interpretação
# A análise fatorial revelou que a maior parte da variabilidade dos dados é explicada por um número relativamente pequeno de fatores (em torno dos 5 primeiros). A explicação da variância diminui rapidamente à medida que mais fatores são extraídos, e a partir do 19º fator, os fatores se tornam quase irrelevantes.
# 
# Esses resultados sugerem que, apesar de se ter extraído 21 fatores, apenas os primeiros fatores têm um impacto significativo na estrutura subjacente dos dados. Considerando isso, uma abordagem de redução de dimensionalidade utilizando esses primeiros fatores pode ser mais eficaz e representativa para a análise posterior ou para a aplicação de algoritmos de clusterização.

# %% [markdown]
# ## Aplicando analise de fatores utilizando resultados do método Parallel Analysis

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

# %% [markdown]
# ### **Análise dos Resultados da Parallel Analysis (13 Fatores Extraídos)**
# 
# A seguir, vamos realizar a análise dos resultados da Parallel Analysis para os 13 fatores extraídos. Para isso, vamos observar três aspectos principais: **Autovalores (Eigenvalues)**, **Cargas dos Fatores** e **Variância Explicada**.
# 
# #### 1. **Autovalores (Eigenvalues)**
# 
# Os autovalores indicam a quantidade de variância explicada por cada fator. Consideramos os fatores com autovalores superiores a 1 como significativos, de acordo com a regra de Kaiser.
# 
# Os 13 autovalores extraídos foram:
# 
# - Fatores 1 a 13: [2.469, 1.930, 1.623, 1.435, 1.174, 1.066, 1.046, 0.999, 0.927, 0.892, 0.857, 0.827, 0.790]
# 
# Os fatores 1 a 13 possuem autovalores acima de 1, indicando que todos os 13 fatores extraídos são significativos, ou seja, eles explicam uma quantidade relevante da variância nos dados. 
# 
# #### 2. **Cargas dos Fatores**
# 
# As **cargas dos fatores** representam a correlação entre as variáveis originais e os fatores extraídos. Cargas absolutas maiores indicam que a variável tem uma contribuição significativa para o fator correspondente.
# 
# Aqui estão as cargas dos fatores para os 13 fatores extraídos (valores de cada variável por fator):
# 
# **Fator 1**:
# - Variáveis: [-0.0999, 0.9135, 0.0299, 0.0159, -0.004, ...]
# 
# **Fator 2**:
# - Variáveis: [0.0914, 0.0072, 0.0763, 0.0262, -0.0437, ...]
# 
# **Fator 3**:
# - Variáveis: [0.0299, 0.0763, -0.0060, -0.0035, ...]
# 
# Cada linha aqui mostra as cargas das variáveis no fator correspondente. Observando as variáveis que possuem cargas mais altas, podemos concluir quais são mais representativas para cada fator. Por exemplo, o **Fator 2** tem uma carga de **0.9135** para a segunda variável, indicando uma forte associação entre elas.
# 
# #### 3. **Variância Explicada por Fator**
# 
# A **variância explicada** por cada fator e a **variância acumulada** são fundamentais para entender o impacto de cada fator no modelo. A variância explicada por fator mostra quanto de toda a variância nos dados é explicada por cada fator.
# 
# Aqui estão os valores:
# 
# **Variância Explicada por Fator**:
# - Fator 1: 1.7839
# - Fator 2: 0.9264
# - Fator 3: 0.8994
# - Fator 4: 0.7663
# - Fator 5: 0.7617
# - Fator 6: 0.7617
# - Fator 7: 0.6218
# - Fator 8: 0.5660
# - Fator 9: 0.5642
# - Fator 10: 0.5160
# - Fator 11: 0.3835
# - Fator 12: 0.2798
# - Fator 13: 0.2669
# 
# **Variância Acumulada** (em porcentagem):
# - Após o Fator 1: **8.49%**
# - Após o Fator 2: **12.91%**
# - Após o Fator 3: **17.19%**
# - Após o Fator 4: **20.84%**
# - Após o Fator 5: **24.47%**
# - Após o Fator 6: **28.09%**
# - Após o Fator 7: **31.05%**
# - Após o Fator 8: **33.75%**
# - Após o Fator 9: **36.44%**
# - Após o Fator 10: **38.89%**
# - Após o Fator 11: **40.72%**
# - Após o Fator 12: **42.05%**
# - Após o Fator 13: **43.32%**
# 
# **Observação**:
# - A maior parte da variância é explicada pelos primeiros fatores. Os primeiros 4 fatores explicam cerca de **20.84%** da variância, e os primeiros 6 fatores explicam **28.09%**.
# - A contribuição dos fatores diminui significativamente a partir do **Fator 7**, e os fatores a partir deste ponto explicam menos de **3%** da variância total.
# 
# ### **Conclusões**
# 
# 1. **Significância dos Fatores**:
#    - Todos os 13 fatores extraídos são significativos, com base no critério de autovalores superiores a 1.
#    
# 2. **Contribuição dos Fatores**:
#    - Os fatores mais significativos são os primeiros, com o **Fator 1** explicando a maior parte da variância.
#    - A contribuição dos fatores diminui conforme avançamos na ordem dos fatores extraídos, com os últimos fatores tendo uma explicação muito menor da variância.
# 
# 3. **Relevância dos Fatores**:
#    - A análise das **cargas dos fatores** pode ajudar a interpretar quais variáveis estão mais associadas aos fatores extraídos. Variáveis com cargas absolutas mais altas devem ser observadas mais atentamente para entender o papel de cada fator.
# 
# 4. **Escolha de Fatores**:
#    - A escolha de quantos fatores manter pode depender do objetivo da análise. Com base na variância acumulada, pode-se decidir manter os primeiros fatores que explicam a maior parte da variância, descartando os fatores subsequentes que têm uma contribuição menor.
# 

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
fatores_df = fatores_df.iloc[:, :6] 
fatores_df.head()


# %% [markdown]
# #### Visualização de Clusters com UMAP

# %%
fatores_df['diagnostico_hipertensao'] = df['diagnostico_hipertensao']  

# %%
# Reduzindo os fatores para 2D com UMAP
umap_2d = umap.UMAP(n_components=2, random_state=42)
fatores_umap = umap_2d.fit_transform(fatores_df.drop('diagnostico_hipertensao', axis=1))  # Remove a coluna 'diagnostico_hipertensao' antes de aplicar o UMAP

# %%
# Definindo cores específicas para as classes
cores = {1: 'blue', 2: 'red'}

# Plotando a distribuição com base na classe 'diagnostico_hipertensao'
plt.figure(figsize=(10, 8))
for classe in fatores_df['diagnostico_hipertensao'].unique():
    indices = fatores_df['diagnostico_hipertensao'] == classe
    plt.scatter(
        fatores_umap[indices, 0], 
        fatores_umap[indices, 1], 
        c=cores[classe], 
        label=f'Diagnóstico {classe}', 
        s=50, alpha=0.7
    )

# Adicionando título, labels e legenda
plt.title('Distribuição dos Dados usando UMAP com Diagnóstico de Hipertensão', fontsize=16)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# %% [markdown]
# > Salvando fatores_df em um csv

# %%
fatores_df.to_csv(r'C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\base_fatores.csv')


