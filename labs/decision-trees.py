import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# 1. DATASET DE EXEMPLO 
dados = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}
df = pd.DataFrame(dados)

# 2. PRÉ-PROCESSAMENTO: ONE-HOT ENCODING DAS FEATURES
X = pd.get_dummies(df[['Outlook', 'Temperature', 'Humidity', 'Wind']], drop_first=False)
y = df['Play'].map({'No': 0, 'Yes': 1})

# 3. CRIAÇÃO E TREINAMENTO DO MODELO
# max_depth controla a profundidade para evitar overfitting
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X, y)

# 4. EXIBIÇÃO DA ESTRUTURA DA ÁRVORE (Equivalente ao bloco 7)
regras = export_text(clf, feature_names=list(X.columns))
print("=== REGRAS DE DECISÃO APRENDIDAS PELA ÁRVORE ===")
print(regras)

# 5. PREVISÃO EM UM NOVO CENÁRIO (ex: Dia Ensolarado com Humidade Normal)
novo_cenario = pd.DataFrame([{
    'Outlook_Overcast': 0, 'Outlook_Rainy': 0, 'Outlook_Sunny': 1,
    'Temperature_Cool': 0, 'Temperature_Hot': 0, 'Temperature_Mild': 1,
    'Humidity_High': 0, 'Humidity_Normal': 1,
    'Wind_Strong': 0, 'Wind_Weak': 1
}])

pred = clf.predict(novo_cenario)
print(f"Decisão para o novo dia: {'Joga Tênis (Yes)' if pred[0] == 1 else 'Não Joga (No)'}")