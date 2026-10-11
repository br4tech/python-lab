import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 1. DADOS DE TREINAMENTO 
# Features: [X1, X2]
X = np.array([
    [1, 2],  # Ponto A
    [2, 3],  # Ponto B
    [3, 1],  # Ponto C
    [6, 5],  # Ponto D
    [7, 7],  # Ponto E
    [8, 6]   # Ponto F
])

# Classes correspondentes (0 ou 1)
y = np.array([0, 0, 0, 1, 1, 1])

# 2. DEFINIÇÃO DO MODELO COM K = 3
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X, y)

# 3. NOVO PONTO A SER CLASSIFICADO: P(5, 4)
new_point = np.array([[5, 4]])

# Predição de classe
prediction = knn.predict(new_point)
probabilities = knn.predict_proba(new_point)

# Inspeção dos vizinhos mais próximos encontrados
distances, indices = knn.kneighbors(new_point)

print(f"Classe Prevista para P(5, 4): {prediction[0]}")
print(f"Probabilidades [Classe 0, Classe 1]: {probabilities[0]}")
print(f"Distâncias dos 3 vizinhos: {np.round(distances[0], 2)}")
print(f"Índices dos vizinhos: {indices[0]} (Classes: {y[indices[0]]})")