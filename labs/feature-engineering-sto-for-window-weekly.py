import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------
# 1. LOGS HISTÓRICOS DE DISPARO (com dia_semana: 0=Seg a 6=Dom)
# -------------------------------------------------------------
logs = pd.DataFrame([
    {"user_id": 1, "dia_semana": 1, "hora_envio": 10, "abriu": 1},  # Terça, Janela 2 (08h-12h)
    {"user_id": 1, "dia_semana": 1, "hora_envio": 11, "abriu": 1},  # Terça, Janela 2 (08h-12h)
    {"user_id": 1, "dia_semana": 3, "hora_envio": 14, "abriu": 0},  # Quinta, Janela 3 (12h-16h)
    {"user_id": 1, "dia_semana": 5, "hora_envio": 20, "abriu": 1},  # Sábado, Janela 5 (20h-24h)
])

# Identificar janela de 4 horas
logs["janela_id"] = logs["hora_envio"] // 4

# -------------------------------------------------------------
# 2. AGREGAÇÕES HISTÓRICAS DO USUÁRIO
# -------------------------------------------------------------
# A. Histórico específico para a combinação exata (dia, janela)
hist_par = logs.groupby(["user_id", "dia_semana", "janela_id"]).agg(
    envios_par=("abriu", "count"),
    abertos_par=("abriu", "sum")
).reset_index()
hist_par["taxa_par"] = (hist_par["abertos_par"] + 1) / (hist_par["envios_par"] + 2)

# B. Histórico consolidado por dia (marginal)
hist_dia = logs.groupby(["user_id", "dia_semana"]).agg(
    envios_dia=("abriu", "count"),
    abertos_dia=("abriu", "sum")
).reset_index()
hist_dia["taxa_dia"] = (hist_dia["abertos_dia"] + 1) / (hist_dia["envios_dia"] + 2)

# C. Histórico consolidado por janela horária (marginal)
hist_janela = logs.groupby(["user_id", "janela_id"]).agg(
    envios_janela=("abriu", "count"),
    abertos_janela=("abriu", "sum")
).reset_index()
hist_janela["taxa_janela"] = (hist_janela["abertos_janela"] + 1) / (hist_janela["envios_janela"] + 2)

# -------------------------------------------------------------
# 3. GERAÇÃO DA GRADE COM OS 42 CANDIDATOS (7 dias x 6 janelas)
# -------------------------------------------------------------
dias = range(7)        # 0=Segunda ... 6=Domingo
janelas = range(6)     # 0=00h-04h ... 5=20h-24h

nomes_dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
rotulos_janelas = ["00h-04h", "04h-08h", "08h-12h", "12h-16h", "16h-20h", "20h-24h"]

candidatos_lista = list(itertools.product([1], dias, janelas))  # Lead user_id = 1
df_candidatos = pd.DataFrame(candidatos_lista, columns=["user_id", "dia_semana", "janela_id"])

# -------------------------------------------------------------
# 4. ENRIQUECIMENTO COM ATRIBUTOS (FEATURE MERGING)
# -------------------------------------------------------------
df_features = df_candidatos.merge(
    hist_par[["user_id", "dia_semana", "janela_id", "taxa_par"]], 
    on=["user_id", "dia_semana", "janela_id"], how="left"
).merge(
    hist_dia[["user_id", "dia_semana", "taxa_dia"]], 
    on=["user_id", "dia_semana"], how="left"
).merge(
    hist_janela[["user_id", "janela_id", "taxa_janela"]], 
    on=["user_id", "janela_id"], how="left"
)

# Imputação neutra (0.5) para combinações sem histórico prévio
df_features["taxa_par"] = df_features["taxa_par"].fillna(0.5)
df_features["taxa_dia"] = df_features["taxa_dia"].fillna(0.5)
df_features["taxa_janela"] = df_features["taxa_janela"].fillna(0.5)

# Transformações cíclicas e indicadoras
df_features["sin_janela"] = np.sin(2 * np.pi * df_features["janela_id"] / 6)
df_features["cos_janela"] = np.cos(2 * np.pi * df_features["janela_id"] / 6)
df_features["sin_dia"] = np.sin(2 * np.pi * df_features["dia_semana"] / 7)
df_features["cos_dia"] = np.cos(2 * np.pi * df_features["dia_semana"] / 7)
df_features["fim_de_semana"] = df_features["dia_semana"].apply(lambda d: 1.0 if d in [5, 6] else 0.0)

# Colunas numéricas finais para a matriz de treino/inferência
colunas_modelo = [
    "taxa_par", "taxa_dia", "taxa_janela",
    "sin_janela", "cos_janela", "sin_dia", "cos_dia", "fim_de_semana"
]

scaler = StandardScaler()
X_matriz = scaler.fit_transform(df_features[colunas_modelo].values)

# Adicionando rótulos amigáveis para inspeção
df_features["dia_nome"] = df_features["dia_semana"].map(lambda d: nomes_dias[d])
df_features["janela_nome"] = df_features["janela_id"].map(lambda w: rotulos_janelas[w])

print(f"Total de combinações geradas: {len(df_features)} linhas (7 dias x 6 janelas)")
print(f"Dimensão da matriz de entrada (X): {X_matriz.shape}")
print("\nExemplo das primeiras 5 opções candidatas:")
print(df_features[["dia_nome", "janela_nome"] + colunas_modelo[:3]].head())