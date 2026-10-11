import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. CRIANDO O DATASET IDÊNTICO AO DA TABELA DA IMAGEM
dados = {
    "Age": [25.0, np.nan, 35.0, 29.0],
    "Salary": [50000.0, 62000.0, np.nan, 50000.0],
    "City": ["Delhi", "Mumbai", "Delhi", "Delhi"],
    "Purchased": ["Yes", "No", "Yes", "No"],  # Substituindo o NaN do exemplo
}
df = pd.DataFrame(dados)
print("--- DADOS BRUTOS (RAW DATA) ---")
print(df)

# Separando X (features) e y (target)
X = df[["Age", "Salary", "City"]]
y = df["Purchased"].map({"No": 0, "Yes": 1})

# Divisão Treino e Teste antes de transformar (Evita Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 2. DEFINIÇÃO DAS TRANSFORMAÇÕES (Preprocessamento Robusto)
colunas_numericas = ["Age", "Salary"]
colunas_categoricas = ["City"]

# Pipeline para números: preenche com a média e padroniza a escala
pipeline_numerico = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

# Pipeline para categorias: One-Hot Encoding
pipeline_categorico = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
)

# Combinando tudo com ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", pipeline_numerico, colunas_numericas),
        ("cat", pipeline_categorico, colunas_categoricas),
    ]
)

# 3. APLICANDO O PRÉ-PROCESSAMENTO
# fit_transform no treino (aprende média e desvio padrão)
X_train_processado = preprocessor.fit_transform(X_train)

# transform no teste (usa parâmetros aprendidos no treino, sem recalcular)
X_test_processado = preprocessor.transform(X_test)

print("\n--- DADOS PROCESSADOS (PRONTOS PARA O MODELO) ---")
print(np.round(X_train_processado, 2))