# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import sidetable as stb ## Plotar dados Ausentes
import umap ## Reduzir dimensionalidade dos dados e plotar em 2D 

# %%
data = pd.read_csv("pns2019.csv")


# %%
colunas_relevantes = [
    'P034',  # Quantos dias por semana o(a) Sr(a) costuma  (costumava)praticar exercício físico ou esporte?
    'P027',  # Com que frequência o(a) Sr(a) costuma consumir alguma bebida alcoólica?
    'P029',  # Em geral, no dia que o(a) Sr(a) bebe, quantas doses de bebida alcoólica o(a) Sr(a) consome?
    'J01101',  # Quando consultou um médico pela última vez
    'J014',  # Nas duas últimas semanas, procurou algum lugar, serviço ou profissional de saúde para atendimento relacionado à própria saúde
    'N010',  # Nas duas últimas semanas, com que frequência o(a) Sr(a) teve problemas no sono, como dificuldade para adormecer, acordar frequentemente à noite ou dormir mais do que de costume?
    'P050',  # Atualmente, o(a) Sr(a) fuma algum produto do tabaco?
    'Q00201',  # Quando foi a última vez que o(a) Sr(a) teve sua pressão arterial alta
    'P02602', #Em quantos dias da semana o(a) Sr(a) costuma substituir a refeição do almoço por lanches rápidos como sanduíches, salgados, pizza, cachorro quente, etc?
    'P02601', # Considerando a comida preparada na hora e os alimentos industrializados, o(a) Sr(a) acha que o seu consumo de sal é:
    'P02002', # Em quantos dias da semana o(a) Sr(a) costuma tomar refrigerante? 
    'P023',  # Em quantos dias da semana o(a) Sr(a) costuma tomar leite? (de origem animal: vaca, cabra, búfala etc.) 
    'P013', # Em quantos dias da semana o(a) Sr(a) costuma comer frango/galinha? 
    'P02001', # Em quantos dias da semana o(a) Sr(a) costuma tomar suco de caixinha/lata ou refresco em pó ? 
    'P01101', # Em quantos dias da semana o(a) Sr(a) costuma comer carne vermelha (boi, porco, cabrito, bode, ovelha etc.)? 
    'P018', # Em quantos dias da semana o(a) Sr(a) costuma comer frutas? 
    'P01601', # Em quantos dias da semana o(a) Sr(a) costuma tomar suco de fruta natural (incluída a polpa de fruta congelada)? 
    'P02501', #Em quantos dias da semana o(a) Sr(a) costuma comer alimentos doces como biscoito/bolacha recheado, chocolate, gelatina, balas e outros?
    'P00901', # Em quantos dias da semana, o(a) Sr(a) costuma comer pelo menos um tipo de verdura ou legume (sem contar batata, mandioca, cará ou inhame) como alface, tomate, couve, cenoura, chuchu, berinjela, abobrinha? 
    'P00403', #ALTURA
    'P00104',#peso
    'J001', #De um modo geral, como é o estado de saúde de_______
]



# Seleciona apenas as colunas relevantes
data = data[colunas_relevantes]

# %%
data = data[data['Q00201'].notna()]

# %%
#Excluindo linhas duplicadas
data = data.drop_duplicates(keep=False)

# %%
# Resumo dos valores nulos por coluna
data.stb.missing().round(2)

# %%
print("Contagem de Valores:" )
print(data['P027'].value_counts())

# %%
# Aplicando a lógica
for index, row in data.iterrows():
    if row['P027'] >= 2 and pd.isna(row['P029']):
        data.at[index, 'P029'] = None
    elif pd.isna(row['P029']):
        data.at[index, 'P029'] = '0'

# %%
# Resumo dos valores nulos por coluna
data.stb.missing().round(2)

# %%
data = data.dropna()

# %%
# Supondo que 'data' seja o DataFrame completo com os dados imputados
data.to_csv('dados_semnulo.csv', index=False)



