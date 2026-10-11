import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. DADOS DE EXEMPLO 
data = {
    "Area": [1000, 1200, 1500, 1800, 2000],
    "Price": [50, 60, 75, 90, 100],  # Preço em Lakhs
}
df = pd.DataFrame(data)

# X precisa ser uma matriz bidimensional (features); y é o vetor unidimensional (target)
X = df[["Area"]]
y = df["Price"]

# 2. DIVISÃO TREINO / TESTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. CRIAÇÃO E AJUSTE DO MODELO (.fit minimiza o MSE)
model = LinearRegression()
model.fit(X_train, y_train)

# 4. PREVISÕES
y_pred = model.predict(X_test)

# 5. EXTRAÇÃO DOS PARÂMETROS E MÉTRICAS
slope = model.coef_[0]
intercept = model.intercept_
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Equação da Reta : Preço = ({slope:.4f} * Area) + ({intercept:.4f})")
print(f"Inclinação (m)  : {slope:.4f}")
print(f"Intercepto (c)  : {intercept:.4f}")
print(f"Previsões Teste : {y_pred}")
print(f"MSE             : {mse:.4f}")
print(f"R² Score        : {r2:.4f}")