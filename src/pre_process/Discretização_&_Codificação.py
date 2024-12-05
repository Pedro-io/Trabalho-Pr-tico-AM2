#%% 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import KBinsDiscretizer
#%%
# Configurando e Carregando nossos dados
# Permite visualizar todas as colunas do DataFrame
pd.set_option('display.max_columns', None)

# Leitura dos dados
df = pd.read_csv(r"C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\base_sem_outiliers.csv")

#%%
df = df.drop(columns= ['Unnamed: 0.1', 'Unnamed: 0'])
df.columns 
# %%
colunas_numericas = ['peso', 'altura', 'doses_por_dia_consumo_alcool', 'IMC']

# %%
# Discretização dos dados - IMC

# Definindo as faixas para discretização

faixas_imc = [0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf]  # Faixas de IMC da OMS
# Criando o objeto KBinsDiscretizer
discretizador_imc = KBinsDiscretizer(n_bins=len(faixas_imc)-1, encode='ordinal', strategy='uniform')

# Discretizando as colunas
df['imc_discretizado'] = discretizador_imc.fit_transform(df[['IMC']]).astype(int)

#%%
# Discretização dos dados - Doses de Alcool 

# Definindo as bordas dos intervalos manuais e criando os rótulos
bins = [0, 1, 2, 4, 7]
discretizador_doses = KBinsDiscretizer(n_bins=len(bins) - 1, encode='ordinal', strategy='uniform')

# Adaptando os dados aos intervalos personalizados para ajuste
valores_transformados = np.digitize(df['doses_por_dia_consumo_alcool'], bins) - 1

# Ajustando à saída no DataFrame
df['doses_por_dia_consumo_alcool_discretizada'] = discretizador_doses.fit_transform(valores_transformados.reshape(-1, 1)).astype(int)

# %%
df.drop(columns=['IMC', 'doses_por_dia_consumo_alcool', 'peso', 'altura'], inplace=True)
df.head()

df.to_csv(r'C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\base_discretizada.csv')