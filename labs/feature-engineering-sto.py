import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. LOGS BRUTOS DE DISPAROS E ABERTURAS
logs_envios = pd.DataFrame([
    {"usuario_id": 1, "hora_envio": 9, "abriu": 1, "clicou": 1},
    {"usuario_id": 1, "hora_envio": 9, "abriu": 1, "clicou": 0},
    {"usuario_id": 1, "hora_envio": 15, "abriu": 0, "clicou": 0},
    {"usuario_id": 1, "hora_envio": 15, "abriu": 1, "clicou": 0},
])

# 2. AGREGAÇÃO POR HORÁRIO COM SUAVIZAÇÃO BAYESIANA
stats_horario = logs_envios.groupby(["usuario_id", "hora_envio"]).agg(
    total_enviados=("abriu", "count"),
    total_abertos=("abriu", "sum")
).reset_index()

# Suavização (alpha=1, beta=2)
stats_horario["taxa_abertura"] = (stats_horario["total_abertos"] + 1) / (stats_horario["total_enviados"] + 2)

# 3. HORÁRIOS CANDIDATOS PARA O DISPARO DE HOJE (Ex: 09h vs 15h)
candidatos = pd.DataFrame([
    {"usuario_id": 1, "hora_candidata": 9, "ativo_ultima_hora": 1.0},
    {"usuario_id": 1, "hora_candidata": 15, "ativo_ultima_hora": 0.0}
])

# Cruzamento dos atributos
df_features = pd.merge(
    candidatos,
    stats_horario,
    left_on=["usuario_id", "hora_candidata"],
    right_on=["usuario_id", "hora_envio"],
    how="left"
).fillna({"taxa_abertura": 0.5})  # Média neutra se não houver histórico

# 4. TRANSFORMAÇÃO DE CICLO TEMPORAL (Seno / Cosseno)
df_features["hora_seno"] = np.sin(2 * np.pi * df_features["hora_candidata"] / 24)
df_features["hora_cosseno"] = np.cos(2 * np.pi * df_features["hora_candidata"] / 24)

# 5. MATRIZ FINAL DE ENTRADA (X)
colunas_finais = ["taxa_abertura", "ativo_ultima_hora", "hora_seno", "hora_cosseno"]
X_bruto = df_features[colunas_finais].values

# Padronização para alimentar o modelo de ranking
scaler = StandardScaler()
X_processado = scaler.fit_transform(X_bruto)

print("Tabela final de features:")
print(df_features[["hora_candidata"] + colunas_finais])
print("\nMatriz numérica pronta para o modelo LTR:")
print(np.round(X_processado, 2))