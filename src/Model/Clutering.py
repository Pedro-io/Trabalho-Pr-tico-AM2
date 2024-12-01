# %% Importações necessárias
# Importando bibliotecas para manipulação de dados e visualização
import pandas as pd              # Para trabalhar com dataframes
import matplotlib.pyplot as plt  # Para gerar gráficos
import numpy as np               # Para operações matemáticas

# Importando bibliotecas específicas de clusterização
from kmodes.kmodes import KModes            # Para o algoritmo K-Modes (ideal para dados categóricos)
from sklearn.metrics import silhouette_score  # Para medir a qualidade dos clusters usando o índice de silhueta
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram  # Para clusterização hierárquica
from CLOPE import CLOPE  # Importe sua implementação do algoritmo CLOPE (usado para dados categóricos)

# %% Carregando os dados
# Lê os dados pré-processados salvos anteriormente contendo fatores calculados e resultados UMAP
fatores_df = pd.read_csv(r'C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\base_fatores.csv', index_col=0)
fatores_umap = pd.read_csv(r'C:\Users\maype\Desktop\projetos\Trabalho Prático AM2\data\fatores_umap.csv', index_col=0).to_numpy()

# %% Método do Cotovelo para K-Modes
# Função para determinar o número ideal de clusters usando o método do cotovelo
def metodo_cotovelo_kmodes(data, max_clusters=15):
    custos = []  # Lista para armazenar os custos de cada execução do K-Modes
    
    # Loop para testar diferentes números de clusters de 2 até max_clusters
    for k in range(2, max_clusters + 1):
        # Inicializa o modelo K-Modes com 'Huang' e realiza múltiplas inicializações para estabilidade
        kmodes = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0)
        kmodes.fit(data)  # Ajusta o modelo aos dados
        custos.append(kmodes.cost_)  # Armazena o custo associado a cada k
    
    # Plotando o gráfico do método do cotovelo
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), custos, marker='o')
    plt.title('Método do Cotovelo - K-Modes')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Custo')
    plt.grid(True)
    plt.show()

# Executando o método do cotovelo com o DataFrame de fatores (removendo a coluna de diagnóstico para evitar viés)
metodo_cotovelo_kmodes(fatores_df.drop('diagnostico_hipertensao', axis=1))

# %% Clusterização com K-Modes
# Inicializando o modelo K-Modes com o número de clusters ideal (ajustar com base no cotovelo)
k_modes = KModes(n_clusters=5, init='Huang', n_init=5, verbose=0)
clusters_kmodes = k_modes.fit_predict(fatores_df.drop('diagnostico_hipertensao', axis=1))  # Realizando a clusterização
fatores_df['cluster_kmodes'] = clusters_kmodes  # Salvando os clusters no DataFrame original

# %% Clusterização com CLOPE

# Chamando a função CLOPE diretamente:
table_clope, clusters_clope = CLOPE(df=fatores_df, k=9, r=2.5, real_label="diagnostico_hipertensao")

# Extraindo os rótulos dos clusters da tabela de saída do CLOPE:
fatores_df['cluster_clope'] = table_clope['cluster_label']


# %% Clusterização Hierárquica
# Criando a matriz de ligação usando o método de Ward (minimiza a variância dentro dos clusters)
linkage_matrix = linkage(fatores_df.drop('diagnostico_hipertensao', axis=1), method='ward')

# Aplicando o corte na árvore hierárquica para definir 5 clusters
clusters_hierarquico = fcluster(linkage_matrix, t=5, criterion='maxclust')
fatores_df['cluster_hierarquico'] = clusters_hierarquico  # Salvando os clusters no DataFrame

# Visualizando o dendrograma para entender a hierarquia de clusters
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title("Dendrograma da Clusterização Hierárquica")
plt.xlabel("Observações")
plt.ylabel("Distância")
plt.show()

# %% Visualização dos Clusters
# Função para plotar clusters em 2D com base na projeção UMAP e destacar a classe de hipertensão
def plot_clusters_2d(data, labels, diagnostico, titulo):
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='Spectral', s=30, alpha=0.7, label='Clusters')
    
    # Sobreposição de pontos classificados pelo diagnóstico
    for classe in np.unique(diagnostico):
        indices = np.where(diagnostico == classe)
        plt.scatter(data[indices, 0], data[indices, 1], edgecolor='black', label=f'Diagnóstico {classe}', s=50, alpha=0.7)
    
    plt.colorbar(label='Clusters')
    plt.title(titulo)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.show()

# Visualizando os clusters para cada algoritmo
plot_clusters_2d(fatores_umap, clusters_kmodes, fatores_df['diagnostico_hipertensao'].to_numpy(), 'Clusters K-Modes com Diagnóstico')
plot_clusters_2d(fatores_umap, clusters_clope, fatores_df['diagnostico_hipertensao'].to_numpy(), 'Clusters CLOPE com Diagnóstico')
plot_clusters_2d(fatores_umap, clusters_hierarquico, fatores_df['diagnostico_hipertensao'].to_numpy(), 'Clusters Hierárquicos com Diagnóstico')

# %% Análise de Silhueta para K-Modes e Hierárquico
# Calculando o índice de silhueta para avaliar a qualidade dos clusters usando a métrica de distância 'hamming' (para dados categóricos)
silhouette_kmodes = silhouette_score(fatores_df.drop('diagnostico_hipertensao', axis=1), clusters_kmodes, metric='hamming')
silhouette_hierarquico = silhouette_score(fatores_df.drop('diagnostico_hipertensao', axis=1), clusters_hierarquico, metric='hamming')

# Exibindo os resultados do índice de silhueta para os dois modelos
print(f"Índice de Silhueta para K-Modes: {silhouette_kmodes:.3f}")
print(f"Índice de Silhueta para Clusterização Hierárquica: {silhouette_hierarquico:.3f}")
