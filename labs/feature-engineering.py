import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# 1. ENCODING CATEGORICAL DATA
# -------------------------------------------------------------
df_cores = pd.DataFrame({'Color': ['Red', 'Blue', 'Green']})

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False)
ohe_array = ohe.fit_transform(df_cores[['Color']])
df_ohe = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(['Color']))

print("=== ONE-HOT ENCODING ===")
print(pd.concat([df_cores, df_ohe], axis=1))

# -------------------------------------------------------------
# 2. FEATURE SCALING
# -------------------------------------------------------------
df_valores = pd.DataFrame({'Value': [10.0, 20.0, 30.0]})

# Standardization (Z-score: mean=0, std=1)
std_scaler = StandardScaler()
df_valores['Z_score'] = std_scaler.fit_transform(df_valores[['Value']])

# Min-Max Scaling (Intervalo 0 a 1)
minmax_scaler = MinMaxScaler()
df_valores['MinMax_0_1'] = minmax_scaler.fit_transform(df_valores[['Value']])

print("\n=== FEATURE SCALING ===")
print(df_valores)

# -------------------------------------------------------------
# 3. FEATURE SELECTION
# -------------------------------------------------------------
# Criando dataset simulado com colunas da imagem
np.random.seed(42)
dados = {
    'Age': [25, 30, 45, 50, 23],
    'Salary': [3000, 4500, 8000, 9500, 3200],
    'City_Code': [1, 2, 1, 3, 2],
    'Experience': [1, 3, 10, 12, 1],
    'Target': [0, 0, 1, 1, 0]
}
df_selecao = pd.DataFrame(dados)

# Matriz de correlação com a variável target
correlacoes = df_selecao.corr()['Target'].abs().drop('Target')

# Filtro de corte: manter apenas features com correlação > 0.5
features_selecionadas = correlacoes[correlacoes > 0.5].index.tolist()

print("\n=== FEATURE SELECTION (CORRELAÇÃO) ===")
print("Correlações absolutas:")
print(correlacoes)
print(f"Features mantidas (> 0.5): {features_selecionadas}")

# -------------------------------------------------------------
# 4. TRAIN-TEST SPLIT COM ESTRATIFICAÇÃO 
# -------------------------------------------------------------
X = df_selecao[features_selecionadas]
y = df_selecao['Target']

# O parâmetro stratify garante proporção idêntica de classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTamanho Treino: {X_train.shape[0]} amostras | Tamanho Teste: {X_test.shape[0]} amostras")