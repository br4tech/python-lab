import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------
# 1. REGISTOS HISTÓRICOS DE DISPARO (Dados Brutos)
# -------------------------------------------------------------
logs = pd.DataFrame([
    {"user_id": 1, "send_hour": 9,  "opened": 1, "clicked": 1},  # Janela 2 (08h-12h)
    {"user_id": 1, "send_hour": 10, "opened": 1, "clicked": 0},  # Janela 2 (08h-12h)
    {"user_id": 1, "send_hour": 14, "opened": 0, "clicked": 0},  # Janela 3 (12h-16h)
    {"user_id": 1, "send_hour": 15, "opened": 1, "clicked": 0},  # Janela 3 (12h-16h)
    {"user_id": 1, "send_hour": 21, "opened": 0, "clicked": 0},  # Janela 5 (20h-24h)
])

# Mapear cada hora de envio para o bloco de 4 horas (0 a 5)
logs["window_id"] = logs["send_hour"] // 4

# -------------------------------------------------------------
# 2. AGREGAÇÃO POR UTILIZADOR E POR JANELA
# -------------------------------------------------------------
stats_janela = logs.groupby(["user_id", "window_id"]).agg(
    total_envios=("opened", "count"),
    total_abertos=("opened", "sum"),
    total_cliques=("clicked", "sum")
).reset_index()

# Suavização Bayesiana (alpha=1, beta=2)
stats_janela["taxa_abertura"] = (stats_janela["total_abertos"] + 1) / (stats_janela["total_envios"] + 2)
stats_janela["taxa_clique"] = (stats_janela["total_cliques"] + 1) / (stats_janela["total_envios"] + 2)

# -------------------------------------------------------------
# 3. CONSTRUÇÃO DA LISTA DE JANELAS CANDIDATAS (6 blocos por utilizador)
# -------------------------------------------------------------
janelas_rotulos = {
    0: "00h-04h",
    1: "04h-08h",
    2: "08h-12h",
    3: "12h-16h",
    4: "16h-20h",
    5: "20h-24h"
}

# Criar a grelha de 6 janelas para o utilizador 1
candidatos = pd.DataFrame([
    {"user_id": 1, "window_id": w, "descricao": janelas_rotulos[w]}
    for w in range(6)
])

# Simulação: indicar se o utilizador esteve activo na plataforma nessa janela hoje
# (Exemplo: esteve activo apenas no período da manhã, janela 2)
candidatos["activo_hoje"] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

# -------------------------------------------------------------
# 4. JUNÇÃO E CODIFICAÇÃO TRIGONOMÉTRICA
# -------------------------------------------------------------
df_features = pd.merge(
    candidatos,
    stats_janela,
    on=["user_id", "window_id"],
    how="left"
)

# Preencher janelas sem histórico com valores a priori neutros
df_features["taxa_abertura"] = df_features["taxa_abertura"].fillna(0.5)
df_features["taxa_clique"] = df_features["taxa_clique"].fillna(0.5)

# Transformação cíclica considerando as 6 janelas
df_features["sin_janela"] = np.sin(2 * np.pi * df_features["window_id"] / 6)
df_features["cos_janela"] = np.cos(2 * np.pi * df_features["window_id"] / 6)

# -------------------------------------------------------------
# 5. MATRIZ FINAL ESCALONADA (X)
# -------------------------------------------------------------
features_cols = ["taxa_abertura", "taxa_clique", "activo_hoje", "sin_janela", "cos_janela"]
X_raw = df_features[features_cols].values

scaler = StandardScaler()
X_escalonado = scaler.fit_transform(X_raw)

# -------------------------------------------------------------
# 6. RESULTADO
# -------------------------------------------------------------
print("Tabela de Candidatos e Atributos:")
print(df_features[["window_id", "descricao"] + features_cols])

print("\nMatriz final de entrada para ordenação (shape: 6 x 5):")
print(np.round(X_escalonado, 2))