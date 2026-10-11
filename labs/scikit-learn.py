import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------------------------------------------
# 1. LEARN FROM DATA: Dados históricos (Ex: Anos de experiência vs Salário em R$ mil)
# -------------------------------------------------------------
# X = Features (variáveis explicativas / entradas)
# y = Target (variável alvo / o que queremos prever)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([3.5, 4.2, 5.0, 6.8, 7.5, 8.9, 9.5, 11.0, 12.1, 13.5])

# Dividimos em treino (experiência passada) e teste (o teste surpresa)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------
# 2. FIND PATTERNS & 3. BUILD MODELS: O algoritmo ajusta seus parâmetros
# -------------------------------------------------------------
model = LinearRegression()

# O método .fit() é onde a mágica do aprendizado acontece:
# Ele descobre os pesos (pesos e viés: y = w*x + b) que minimizam o erro.
model.fit(X_train, y_train)

# -------------------------------------------------------------
# 4. MAKE PREDICTIONS: Aplicando a novos cenários
# -------------------------------------------------------------
novos_candidatos = np.array([[3.5], [12.0]])  # 3.5 anos e 12 anos de experiência
previsoes = model.predict(novos_candidatos)

for exp, sal in zip(novos_candidatos.flatten(), previsoes):
    print(f"Para {exp:.1f} anos de experiência -> Salário estimado: R$ {sal:.2f} mil")

# Avaliando nos dados de teste:
y_pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"\nErro Médio da Estimativa (RMSE): R$ {rmse:.2f} mil")