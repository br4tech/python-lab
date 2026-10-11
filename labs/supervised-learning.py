import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =============================================================
# 1. SUPERVISED LEARNING (Ex: Classificação com Gabarito)
# =============================================================
# Cenário: Prever se um cliente compra (1) ou não (0)
# Features (X): [Idade, Renda Anual (em milhares)]
# Labels (y): [Comprou ou Não]
X_sup = np.array(
    [
        [22, 25],
        [25, 32],
        [47, 80],
        [52, 95],
        [23, 29],
        [45, 88],
        [20, 20],
        [55, 110],
    ]
)
y_sup = np.array([0, 0, 1, 1, 0, 1, 0, 1])  # DADOS ROTULADOS (Labeled Data)

# Treinamento: o algoritmo calcula os pesos ajustando o erro contra y
modelo_sup = LogisticRegression()
modelo_sup.fit(X_sup, y_sup)

# Predição em cliente novo
novo_cliente = np.array([[49, 85]])  # 49 anos, renda 85k
pred_classe = modelo_sup.predict(novo_cliente)
print(f"[Supervisionado] Previsão para novo cliente: Classe {pred_classe[0]}")


# =============================================================
# 2. UNSUPERVISED LEARNING (Ex: K-Means Clustering)
# =============================================================
# Cenário: Segmentação de Clientes. NÃO TEMOS labels (y), apenas features (X)
X_unsup = np.array(
    [
        [15, 39],
        [16, 81],
        [17, 6],
        [18, 77],
        [60, 40],
        [62, 42],
        [58, 45],
        [63, 38],
    ]
)

# O algoritmo agrupa por proximidade espacial (distância euclidiana)
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X_unsup)

# Atribuição dos grupos encontrados
grupos = kmeans.labels_
print(f"[Não Supervisionado] Grupos atribuídos automaticamente: {grupos}")
print(f"[Não Supervisionado] Centróides encontrados:\n{kmeans.cluster_centers_}")