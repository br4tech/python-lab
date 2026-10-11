import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

# 1. DADOS DE EXEMPLO 
# Duas classes claramente separadas no plano 2D
X = np.array([
    [2, 3], [3, 3], [2, 1],  # Amostras da Classe 0
    [6, 7], [7, 6], [6, 5]   # Amostras da Classe 1
])
y = np.array([0, 0, 0, 1, 1, 1])

# 2. TREINAMENTO COM KERNEL RBF (Conforme código da imagem)
# C controla a penalidade de classificação incorreta; gamma define o alcance de influência
model_rbf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
model_rbf.fit(X, y)

# 3. PREVISÃO E AVALIAÇÃO
y_pred_rbf = model_rbf.predict(X)
print("=== RESULTADOS KERNEL RBF ===")
print(f"Acurácia: {accuracy_score(y, y_pred_rbf):.2f}")
print("Classification Report:\n", classification_report(y, y_pred_rbf))

# 4. INSPEÇÃO DOS VETORES DE SUPORTE
print(f"Índices dos Support Vectors: {model_rbf.support_}")
print(f"Coordenadas dos Support Vectors:\n{model_rbf.support_vectors_}")
print(f"Quantidade de Support Vectors por classe: {model_rbf.n_support_}")

# 5. EXEMPLO COMPARATIVO: KERNEL LINEAR (Para extrair a equação w*x + b)
model_linear = svm.SVC(kernel='linear', C=1.0)
model_linear.fit(X, y)

w = model_linear.coef_[0]
b = model_linear.intercept_[0]
print("\n=== HIPERPLANO (KERNEL LINEAR) ===")
print(f"Pesos w: {np.round(w, 4)} | Intercepto b: {b:.4f}")
print(f"Margem geométrica (2 / ||w||): {2 / np.linalg.norm(w):.4f}")