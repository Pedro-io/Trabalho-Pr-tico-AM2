#%% 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

#%%
# Configurando e Carregando nossos dados
# Permite visualizar todas as colunas do DataFrame
pd.set_option('display.max_columns', None)

# Leitura dos dados
df = pd.read_csv(r"C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\base_formatada.csv")

#%% 
df.info(0)

#%%
df['diagnostico_hipertensao'].value_counts()
# %%
colunas_numericas = ['peso', 'altura', 'doses_por_dia_consumo_alcool', 'IMC']
df[colunas_numericas].describe()

# %%
def remover_outliers(data):
    # 1. Remover outliers grosseiros com Z-score
    outliers_zscore = detectar_outliers_zscore(data, limite_inferior=-4, limite_superior=4)
    data = data[~data.isin(outliers_zscore)]
    
    # 2. Remover outliers restantes com IQR
    outliers_iqr = detectar_outliers_iqr(data)
    data = data[~data.isin(outliers_iqr)]
    
    return data

# Função para detectar outliers usando o método IQR
def detectar_outliers_iqr(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = data[(data < limite_inferior) | (data > limite_superior)]
    return outliers

# Função para detectar outliers usando o método Zscore
def detectar_outliers_zscore(data, limite_inferior=-4, limite_superior=4):
  z_scores = abs(zscore(data))  # Calcula o Z-score para cada valor
  outliers = data[(z_scores < limite_inferior) | (z_scores > limite_superior)]
  return outliers

#%% 
# Criando uma cópia do DataFrame para não modificar o original
df_com_outliers = df.copy()

#%%
# Visualizando Outliers com Box Plots (Horizontal)
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, len(colunas_numericas), figsize=(15, 6))  # Mudança para horizontal

for i, coluna in enumerate(colunas_numericas):
    sns.boxplot(x=df[coluna], ax=axes[i], color='lightblue', showmeans=True, showfliers=False, orient='h')  # Adicionando 'orient='h'
    axes[i].set_title(f'Boxplot de {coluna}')
    axes[i].set_xlabel(coluna)

plt.tight_layout()
plt.show()

#%%
# Visualizando Outliers com Scatter Plots
# Peso x Altura
plt.figure(figsize=(8, 6))
plt.scatter(df['peso'], df['altura'], alpha=0.5)
plt.xlabel('Peso (KG)')
plt.ylabel('Altura (CM)')
plt.title('Scatter Plot: Peso x Altura')
plt.show()

# Peso x IMC
plt.figure(figsize=(8, 6))
plt.scatter(df['peso'], df['IMC'], alpha=0.5)
plt.xlabel('Peso (KG)')
plt.ylabel('IMC')
plt.title('Scatter Plot: Peso x IMC')
plt.show()

# IMC x Altura
plt.figure(figsize=(8, 6))
plt.scatter(df['IMC'], df['altura'], alpha=0.5)
plt.xlabel('IMC')
plt.ylabel('Altura(CM)')
plt.title('Scatter Plot: IMC x Altura')
plt.show()
#%%
# Visualizando Outliers com Histogramas
for coluna in colunas_numericas:
    plt.figure(figsize=(8, 6))
    plt.hist(df[coluna], bins=20, edgecolor='black')
    plt.xlabel(coluna)
    plt.ylabel('Frequência')
    plt.title(f'Histograma de {coluna}')
    plt.show()

#%%
# Removendo Outliers (Combinando Z-score e IQR)
for coluna in colunas_numericas:
    # 1. Remover outliers grosseiros com Z-score
    outliers_zscore = detectar_outliers_zscore(df[coluna])
    df = df[~df[coluna].isin(outliers_zscore)]

    # 2. Remover outliers restantes com IQR
    outliers_iqr = detectar_outliers_iqr(df[coluna])
    df = df[~df[coluna].isin(outliers_iqr)]
    
#%%
# Verificando se os outliers foram removidos
for coluna in colunas_numericas:
    plt.figure(figsize=(8, 6))
    plt.hist(df[coluna], bins=20, edgecolor='black')
    plt.xlabel(coluna)
    plt.ylabel('Frequência')
    plt.title(f'Histograma de {coluna} (Após Remoção de Outliers)')
    plt.show()
    
    
# Visualizando Outliers com Scatter Plots
# Peso x Altura
plt.figure(figsize=(8, 6))
plt.scatter(df['peso'], df['altura'], alpha=0.5)
plt.xlabel('Peso (KG)')
plt.ylabel('Altura (CM)')
plt.title('Scatter Plot: Peso x Altura')
plt.show()

# Peso x IMC
plt.figure(figsize=(8, 6))
plt.scatter(df['peso'], df['IMC'], alpha=0.5)
plt.xlabel('Peso (KG)')
plt.ylabel('IMC')
plt.title('Scatter Plot: Peso x IMC')
plt.show()

# IMC x Altura
plt.figure(figsize=(8, 6))
plt.scatter(df['IMC'], df['altura'], alpha=0.5)
plt.xlabel('IMC')
plt.ylabel('Altura(CM)')
plt.title('Scatter Plot: IMC x Altura')
plt.show()

#%%
df['diagnostico_hipertensao'].value_counts()


#%%
df.to_csv(r'C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\base_sem_outiliers.csv')
