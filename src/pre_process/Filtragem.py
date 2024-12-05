# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import sidetable as stb ## Plotar dados Ausentes

# %%
data = pd.read_csv("pns2019.csv")


# %%
colunas_relevantes = [
    'P034',  # * CAT * Nos últimos doze meses, o(a) Sr(a) praticou algum tipo de exercício físico ou esporte?(não considere fisioterapia)
    'P027',  # *CAT* Com que frequência o(a) Sr(a) costuma consumir alguma bebida alcoólica?
    'P029',  # *NUM* Em geral, no dia que o(a) Sr(a) bebe, quantas doses de bebida alcoólica o(a) Sr(a) consome?
    'J01101',  # *CAT* Quando consultou um médico pela última vez
    'J014',  # *CAT* Nas duas últimas semanas, procurou algum lugar, serviço ou profissional de saúde para atendimento relacionado à própria saúde
    'N010',  # *CAT* Nas duas últimas semanas, com que frequência o(a) Sr(a) teve problemas no sono, como dificuldade para adormecer, acordar frequentemente à noite ou dormir mais do que de costume?
    'P050',  # *CAT* Atualmente, o(a) Sr(a) fuma algum produto do tabaco?
    'Q00201',  #*TARGET CAT* Algum médico já lhe deu o diagnóstico de hipertensão arterial (pressão alta)
    'P02602', # *CAT* Em quantos dias da semana o(a) Sr(a) costuma substituir a refeição do almoço por lanches rápidos como sanduíches, salgados, pizza, cachorro quente, etc?
    'P02601', # *CAT* Considerando a comida preparada na hora e os alimentos industrializados, o(a) Sr(a) acha que o seu consumo de sal é:
    'P02002', # *CAT*Em quantos dias da semana o(a) Sr(a) costuma tomar refrigerante? 
    'P023',  # *CAT* Em quantos dias da semana o(a) Sr(a) costuma tomar leite? (de origem animal: vaca, cabra, búfala etc.) 
    'P013', # *CAT* Em quantos dias da semana o(a) Sr(a) costuma comer frango/galinha? 
    'P02001', # *CAT* Em quantos dias da semana o(a) Sr(a) costuma tomar suco de caixinha/lata ou refresco em pó ? 
    'P01101', # *CAT* Em quantos dias da semana o(a) Sr(a) costuma comer carne vermelha (boi, porco, cabrito, bode, ovelha etc.)? 
    'P018', # *CAT* Em quantos dias da semana o(a) Sr(a) costuma comer frutas? 
    'P01601', # *CAT* Em quantos dias da semana o(a) Sr(a) costuma tomar suco de fruta natural (incluída a polpa de fruta congelada)? 
    'P02501', #*CAT* Em quantos dias da semana o(a) Sr(a) costuma comer alimentos doces como biscoito/bolacha recheado, chocolate, gelatina, balas e outros?
    'P00901', # *CAT* Em quantos dias da semana, o(a) Sr(a) costuma comer pelo menos um tipo de verdura ou legume (sem contar batata, mandioca, cará ou inhame) como alface, tomate, couve, cenoura, chuchu, berinjela, abobrinha? 
    'P00403', #*NUM *ALTURA
    'P00104',#*NUM* peso
    'J001', # *CAT* De um modo geral, como é o estado de saúde de_______
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



