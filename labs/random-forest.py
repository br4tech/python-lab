import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# =============================================================
# 1. CLASSIFICAÇÃO: Random Forest Classifier 
# =============================================================
data_clf = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy'],
    'Humidity': ['High', 'High', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'No']
}
df_clf = pd.DataFrame(data_clf)

# Pré-processamento com One-Hot Encoding
X_clf = pd.get_dummies(df_clf[['Outlook', 'Humidity', 'Wind']], drop_first=True)
y_clf = df_clf['Play'].map({'No': 0, 'Yes': 1})

# Treinamento com 3 árvores (n_estimators=3 conforme o diagrama visual da imagem)
rf_clf = RandomForestClassifier(n_estimators=3, random_state=42)
rf_clf.fit(X_clf, y_clf)

# Simulação da previsão de uma nova entrada
exemplo_teste = X_clf.iloc[[2]] # Instância de teste
pred_clf = rf_clf.predict(exemplo_teste)
votos_individuais = [tree.predict(exemplo_teste.values)[0] for tree in rf_clf.estimators_]

print("=== RANDOM FOREST CLASSIFIER ===")
print(f"Votos de cada uma das 3 árvores : {votos_individuais}")
print(f"Predição Final (Votação Maioria): {'Play = Yes' if pred_clf[0] == 1 else 'Play = No'}")


# =============================================================
# 2. REGRESSÃO: Random Forest Regressor (House Price - Bloco 6)
# =============================================================
data_reg = {
    'Area': [1000, 1500, 2000, 2500, 3000],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Age': [5, 3, 1, 2, 1],
    'Price': [50, 80, 120, 150, 200]  # Preço em Lakhs
}
df_reg = pd.DataFrame(data_reg)

X_reg = df_reg[['Area', 'Bedrooms', 'Age']]
y_reg = df_reg['Price']

# Treinando com 5 estimadores (n_estimators=5 conforme o exemplo do bloco 6)
rf_reg = RandomForestRegressor(n_estimators=5, random_state=42)
rf_reg.fit(X_reg, y_reg)

novo_imovel = np.array([[2100, 3, 2]])
predicoes_arvores = [tree.predict(novo_imovel)[0] for tree in rf_reg.estimators_]
pred_final_reg = rf_reg.predict(novo_imovel)

print("\n=== RANDOM FOREST REGRESSOR ===")
for i, pred in enumerate(predicoes_arvores, 1):
    print(f"Tree {i} prevê: {pred:.1f} lakhs")
print(f"Previsão Final (Média): {pred_final_reg[0]:.1f} lakhs")