# %%
import pandas as pd
import numpy as np

#%% 
# Configurando e Carregando nossos dados
# Permite visualizar todas as colunas do DataFrame
pd.set_option('display.max_columns', None)

# Leitura dos dados
df = pd.read_csv(r"C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\dados_semnulo.csv")

#%%
"""
Nesta etapa, faremos a padronização dos nomes dos atributos para tornar o conjunto de dados mais compreensível e facilitar sua manipulação. Além disso, calcularemos o IMC (Índice de Massa Corporal) com base nos atributos de peso e altura, considerando sua relevância identificada na análise realizada pelo método CAPTO.
"""
#%%
#Calculando o IMC
def calcular_imc(row):
    peso = row['P00104']  
    altura = row['P00403'] / 100  
    if peso > 0 and altura > 0:
        imc = peso / (altura ** 2)
        return imc
    else:
        return None  

# Aplicar a função para criar a nova coluna dz IMC
df['IMC'] = df.apply(calcular_imc, axis=1)



# %%
# Renomeando atributos

# Mapeando atributos 

atributos = {
    'P034': 'praticou_exercicio_12_meses',  # CAT - Nos últimos doze meses, praticou algum tipo de exercício físico ou esporte?
    'P027': 'frequencia_consumo_alcool',  # CAT - Com que frequência consome alguma bebida alcoólica?
    'P029': 'doses_por_dia_consumo_alcool',  # NUM - No dia que bebe, quantas doses consome?
    'J01101': 'consultou_medico_ultima_vez',  # CAT - Quando consultou um médico pela última vez?
    'J014': 'procurou_atendimento_saude_2_semanas',  # CAT - Procurou atendimento relacionado à saúde nas últimas duas semanas?
    'N010': 'frequencia_problemas_sono',  # CAT - Com que frequência teve problemas no sono nas últimas duas semanas?
    'P050': 'fuma_produto_tabaco',  # CAT - Atualmente fuma algum produto do tabaco?
    'Q00201': 'diagnostico_hipertensao',  # TARGET CAT - Algum médico já diagnosticou hipertensão arterial?
    'P02602': 'substitui_almoco_por_lanche',  # CAT - Dias por semana que substitui almoço por lanches rápidos?
    'P02601': 'percepcao_consumo_sal',  # CAT - Percepção sobre o consumo de sal?
    'P02002': 'dias_semana_consumo_refrigerante',  # CAT - Dias por semana que consome refrigerante?
    'P023': 'dias_semana_consumo_leite',  # CAT - Dias por semana que consome leite de origem animal?
    'P013': 'dias_semana_consumo_frango',  # CAT - Dias por semana que consome frango/galinha?
    'P02001': 'dias_semana_consumo_suco_industrializado',  # CAT - Dias por semana que consome suco de caixinha/lata ou refresco em pó?
    'P01101': 'dias_semana_consumo_carne_vermelha',  # CAT - Dias por semana que consome carne vermelha?
    'P018': 'dias_semana_consumo_frutas',  # CAT - Dias por semana que consome frutas?
    'P01601': 'dias_semana_consumo_suco_natural',  # CAT - Dias por semana que consome suco de fruta natural?
    'P02501': 'dias_semana_consumo_doces',  # CAT - Dias por semana que consome alimentos doces?
    'P00901': 'dias_semana_consumo_verduras_legumes',  # CAT - Dias por semana que consome verduras ou legumes?
    'P00403': 'altura',  # NUM - Altura
    'P00104': 'peso',  # NUM - Peso
    'J001': 'estado_geral_saude',  # CAT - De um modo geral, como é o estado de saúde?
}

# Renomeando as colunas
df = df.rename(columns=atributos)
# %%
#Salvando o conjunto de dados
df.to_csv('base_formatada.csv')
# %%
