import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. DADOS DE EXEMPLO
# Features: Links e Palavras Suspeitas | Target: 1 (Spam) ou 0 (Não Spam)
data = {
    'Links': [3, 0, 2, 0],
    'Suspicious_Words': [5, 1, 4, 0],
    'Spam': [1, 0, 1, 0]
}
df = pd.DataFrame(data)

X = df[['Links', 'Suspicious_Words']] # Matriz 2D de atributos
y = df['Spam']                        # Vetor alvo

# 2. DIVISÃO TREINO / TESTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. CRIAÇÃO E AJUSTE DO MODELO
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. PREVISÕES (Classes e Probabilidades)
y_pred = model.predict(X_test)
# predict_proba retorna as probabilidades [P(y=0), P(y=1)]
y_prob = model.predict_proba(X_test)[:, 1] 

# 5. RESULTADOS E AVALIAÇÃO
print("=== PREVISÕES DO CONJUNTO DE TESTE ===")
for prob, pred in zip(y_prob, y_pred):
    print(f"Probabilidade Prevista: {prob:.4f} -> Classe: {pred}")

print("\n=== MÉTRICAS DE AVALIAÇÃO ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))