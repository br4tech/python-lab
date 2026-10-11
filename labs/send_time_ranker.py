"""
Send Time Optimization (STO) - Learning-to-Rank (LTR) com XGBRanker
===================================================================

Demonstração de como otimizar o horário de disparo de mensagens por lead
usando Learning-to-Rank (LTR) agrupado por lead_id, comparado com heurísticas
de baseline (Tenant Global Frequency) e fallback para Cold-Start.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import xgboost as xgb


# -----------------------------------------------------------------------------
# 1. GERAÇÃO DE DADOS SINTÉTICOS DE LEADS E SLOTS
# -----------------------------------------------------------------------------

def generate_synthetic_sto_data(
    num_leads: int = 80,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], float]]:
    """
    Gera dados sintéticos de leads e slots candidatos para a semana (168 slots: 7 dias x 24h).

    Features de contexto:
      - `lead_total_opens`: total histórico de aberturas do lead.
      - `tenant_avg_opens_in_slot`: média global de aberturas de todos os clientes no slot.
      - `is_weekend`: flag binária se é sábado/domingo.
      - `hour_sin`, `hour_cos`: transformações cíclicas da hora do dia.

    Label:
      - `events_in_slot`: aberturas/cliques observados no slot para aquele lead.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Padrão global do Tenant (ex.: maior tráfego em dias úteis, picos às 10h e 15h)
    tenant_slot_affinity: Dict[Tuple[int, int], float] = {}
    for day in range(7):
        is_wknd = 1.0 if day in (5, 6) else 0.0
        day_factor = 0.4 if is_wknd else 1.0
        for hour in range(24):
            # Curva bimodal de horário comercial (picos por volta de 10h e 15h)
            peak_1 = np.exp(-((hour - 10) ** 2) / 8.0)
            peak_2 = np.exp(-((hour - 15) ** 2) / 8.0)
            base_curve = 0.15 + 0.85 * (0.6 * peak_1 + 0.4 * peak_2)
            tenant_slot_affinity[(day, hour)] = float(base_curve * day_factor)

    # Normalizar média do tenant
    max_t = max(tenant_slot_affinity.values())
    tenant_slot_affinity = {k: v / max_t for k, v in tenant_slot_affinity.items()}

    # 2. Perfis de comportamento de Leads
    # Cada perfil tem horários preferenciais diferentes
    profiles = [
        {"name": "early_bird_weekday", "preferred_days": [1, 2, 3], "preferred_hours": [8, 9]},
        {"name": "lunch_browser", "preferred_days": [1, 2, 3, 4], "preferred_hours": [12, 13]},
        {"name": "afternoon_worker", "preferred_days": [2, 3, 4], "preferred_hours": [15, 16]},
        {"name": "night_owl", "preferred_days": [0, 1, 2, 3, 4], "preferred_hours": [20, 21, 22]},
        {"name": "weekend_shopper", "preferred_days": [5, 6], "preferred_hours": [10, 11, 14, 15]},
    ]

    records: List[Dict[str, Any]] = []

    for lead_idx in range(num_leads):
        lead_id = f"lead_{lead_idx:04d}"
        profile = rng.choice(profiles)
        activity_level = rng.choice(["low", "medium", "high"], p=[0.25, 0.5, 0.25])

        if activity_level == "low":
            lead_total_opens = int(rng.integers(1, 10))
            base_noise = 0.05
        elif activity_level == "medium":
            lead_total_opens = int(rng.integers(10, 50))
            base_noise = 0.15
        else:
            lead_total_opens = int(rng.integers(50, 150))
            base_noise = 0.35

        # Criação de 168 slots candidatos (7 dias x 24 horas)
        for day in range(7):
            is_weekend = 1 if day in (5, 6) else 0

            for hour in range(24):
                hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
                hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
                tenant_avg = tenant_slot_affinity[(day, hour)]

                # Afinidade personalizada do lead
                is_pref_day = day in profile["preferred_days"]
                is_pref_hour = hour in profile["preferred_hours"]

                lead_affinity = 0.0
                if is_pref_day and is_pref_hour:
                    lead_affinity = 2.5 + rng.uniform(0.5, 2.0)
                elif is_pref_day:
                    lead_affinity = 0.4 + rng.uniform(0.0, 0.4)

                # Probabilidade e contagem de eventos de abertura/clique
                expected_events = (
                    0.6 * lead_affinity
                    + 0.4 * (tenant_avg * 1.5)
                    + rng.exponential(base_noise)
                )

                # Fator de escala por atividade total do lead
                events = int(rng.poisson(lam=max(0.01, expected_events * (lead_total_opens / 60.0))))

                records.append({
                    "lead_id": lead_id,
                    "day_of_week": day,
                    "hour": hour,
                    "is_weekend": is_weekend,
                    "hour_sin": hour_sin,
                    "hour_cos": hour_cos,
                    "lead_total_opens": lead_total_opens,
                    "tenant_avg_opens_in_slot": tenant_avg,
                    "events_in_slot": events,
                })

    df = pd.DataFrame(records)
    return df, tenant_slot_affinity


# -----------------------------------------------------------------------------
# 2. SEPARAÇÃO TREINO/TESTE POR LEAD (SEM DATA LEAKAGE)
# -----------------------------------------------------------------------------

def split_train_test_by_lead(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa estritamente os leads entre treino e teste.
    Garante que todos os slots de um mesmo lead permaneçam no mesmo split,
    evitando vazamento de dados (data leakage) entre grupos de ranking.
    """
    unique_leads = df["lead_id"].unique()
    rng = np.random.default_rng(random_seed)
    shuffled_leads = rng.permutation(unique_leads)

    train_size = int(len(unique_leads) * train_ratio)
    train_leads = set(shuffled_leads[:train_size])
    test_leads = set(shuffled_leads[train_size:])

    train_df = df[df["lead_id"].isin(train_leads)].copy()
    test_df = df[df["lead_id"].isin(test_leads)].copy()

    # Ordenar por lead_id para garantir consistência de agrupamento no XGBRanker
    train_df = train_df.sort_values(by=["lead_id", "day_of_week", "hour"]).reset_index(drop=True)
    test_df = test_df.sort_values(by=["lead_id", "day_of_week", "hour"]).reset_index(drop=True)

    return train_df, test_df


# -----------------------------------------------------------------------------
# 3. TREINAMENTO COM XGBRanker (LEARNING-TO-RANK)
# -----------------------------------------------------------------------------

FEATURE_COLS = [
    "day_of_week",
    "hour",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "lead_total_opens",
    "tenant_avg_opens_in_slot",
]


def train_xgb_ranker(
    train_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
) -> Tuple[xgb.XGBRanker, List[int]]:
    """
    Treina o XGBRanker otimizando para NDCG nos grupos de cada lead.
    """
    group_sizes = train_df.groupby("lead_id", sort=False).size().tolist()

    X_train = train_df[feature_cols]
    y_train = train_df["events_in_slot"]

    ranker = xgb.XGBRanker(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=5,
        objective="rank:ndcg",
        eval_metric="ndcg",
        ndcg_exp_gain=False,  # Ideal para contagens brutas (relevância linear)
        random_state=42,
    )

    ranker.fit(
        X_train,
        y_train,
        group=group_sizes,
        verbose=False,
    )

    return ranker, group_sizes


# -----------------------------------------------------------------------------
# 4. AVALIAÇÃO DE NDCG (MODELO vs. BASELINE HEURÍSTICO)
# -----------------------------------------------------------------------------

def evaluate_ranking(
    model: xgb.XGBRanker,
    test_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    k_list: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    """
    Avalia o XGBRanker contra o Baseline (heurística baseada na média global do Tenant).
    Calcula o NDCG médio no conjunto de teste para múltiplos top-k.
    """
    test_leads = test_df["lead_id"].unique()

    model_scores_by_k: Dict[int, List[float]] = {k: [] for k in k_list}
    baseline_scores_by_k: Dict[int, List[float]] = {k: [] for k in k_list}
    model_overall_ndcg: List[float] = []
    baseline_overall_ndcg: List[float] = []

    for lead in test_leads:
        lead_rows = test_df[test_df["lead_id"] == lead]
        y_true = lead_rows["events_in_slot"].to_numpy().reshape(1, -1)

        # Se não há nenhum evento no lead, NDCG não discrimina
        if y_true.sum() == 0:
            continue

        # Predição do Modelo (score de relevância predito)
        X_test = lead_rows[feature_cols]
        y_pred_model = model.predict(X_test).reshape(1, -1)

        # Predição do Baseline: média histórica do tenant no slot
        y_pred_baseline = lead_rows["tenant_avg_opens_in_slot"].to_numpy().reshape(1, -1)

        # NDCG geral (todos os 168 slots)
        model_overall_ndcg.append(float(ndcg_score(y_true, y_pred_model)))
        baseline_overall_ndcg.append(float(ndcg_score(y_true, y_pred_baseline)))

        # NDCG@k
        for k in k_list:
            model_scores_by_k[k].append(float(ndcg_score(y_true, y_pred_model, k=k)))
            baseline_scores_by_k[k].append(float(ndcg_score(y_true, y_pred_baseline, k=k)))

    results = {
        "model_mean_ndcg": float(np.mean(model_overall_ndcg)),
        "baseline_mean_ndcg": float(np.mean(baseline_overall_ndcg)),
        "by_k": {
            k: {
                "model_ndcg": float(np.mean(model_scores_by_k[k])),
                "baseline_ndcg": float(np.mean(baseline_scores_by_k[k])),
                "lift_pct": float(
                    ((np.mean(model_scores_by_k[k]) - np.mean(baseline_scores_by_k[k]))
                     / max(1e-6, np.mean(baseline_scores_by_k[k]))) * 100.0
                ),
            }
            for k in k_list
        },
    }

    return results


# -----------------------------------------------------------------------------
# 5. SIMULAÇÃO DE INFERÊNCIA EM TEMPO REAL COM SUPORTE A COLD-START
# -----------------------------------------------------------------------------

@dataclass
class RecommendationResult:
    lead_id: str
    recommended_day: int
    recommended_day_name: str
    recommended_hour: int
    relevance_score: float
    is_cold_start: bool
    strategy_used: str
    top_3_slots: List[Dict[str, Any]]


DAY_NAMES = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]


def predict_best_send_time(
    lead_profile: Dict[str, Any],
    model: xgb.XGBRanker,
    tenant_slot_affinity: Dict[Tuple[int, int], float],
    candidate_slots: Optional[List[Tuple[int, int]]] = None,
    feature_cols: List[str] = FEATURE_COLS,
) -> RecommendationResult:
    """
    Infere o melhor horário de envio para um lead específico.

    Trata Cold-Start:
      - Se `lead_total_opens == 0` ou sem histórico, recorre ao pico histórico do Tenant.
      - Caso contrário, ranqueia os slots usando o XGBRanker treinado.
    """
    lead_id = lead_profile.get("lead_id", "anonymous_lead")
    lead_total_opens = lead_profile.get("lead_total_opens", 0)

    # Se não foram fornecidos slots candidatos, avalia todos os 168 horários da semana
    if candidate_slots is None:
        candidate_slots = [(d, h) for d in range(7) for h in range(24)]

    # Tratamento de Cold-Start
    if lead_total_opens <= 0:
        # Fallback: seleciona o slot de maior afinidade geral no Tenant
        best_slot = max(candidate_slots, key=lambda slot: tenant_slot_affinity.get(slot, 0.0))
        sorted_slots = sorted(candidate_slots, key=lambda s: tenant_slot_affinity.get(s, 0.0), reverse=True)

        top_3 = [
            {
                "day_of_week": s[0],
                "day_name": DAY_NAMES[s[0]],
                "hour": s[1],
                "score": float(tenant_slot_affinity.get(s, 0.0)),
            }
            for s in sorted_slots[:3]
        ]

        return RecommendationResult(
            lead_id=lead_id,
            recommended_day=best_slot[0],
            recommended_day_name=DAY_NAMES[best_slot[0]],
            recommended_hour=best_slot[1],
            relevance_score=float(tenant_slot_affinity.get(best_slot, 0.0)),
            is_cold_start=True,
            strategy_used="Tenant Global Heuristic Fallback (Cold-Start)",
            top_3_slots=top_3,
        )

    # Preparação das features para cada slot candidato
    rows: List[Dict[str, Any]] = []
    for day, hour in candidate_slots:
        is_weekend = 1 if day in (5, 6) else 0
        hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
        hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
        tenant_avg = tenant_slot_affinity.get((day, hour), 0.0)

        rows.append({
            "day_of_week": day,
            "hour": hour,
            "is_weekend": is_weekend,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "lead_total_opens": lead_total_opens,
            "tenant_avg_opens_in_slot": tenant_avg,
        })

    cand_df = pd.DataFrame(rows)
    scores = model.predict(cand_df[feature_cols])
    cand_df["predicted_score"] = scores

    # Ordenação decrescente por score de relevância
    sorted_df = cand_df.sort_values(by="predicted_score", ascending=False).reset_index(drop=True)
    best_row = sorted_df.iloc[0]

    top_3 = [
        {
            "day_of_week": int(sorted_df.iloc[i]["day_of_week"]),
            "day_name": DAY_NAMES[int(sorted_df.iloc[i]["day_of_week"])],
            "hour": int(sorted_df.iloc[i]["hour"]),
            "score": float(sorted_df.iloc[i]["predicted_score"]),
        }
        for i in range(min(3, len(sorted_df)))
    ]

    return RecommendationResult(
        lead_id=lead_id,
        recommended_day=int(best_row["day_of_week"]),
        recommended_day_name=DAY_NAMES[int(best_row["day_of_week"])],
        recommended_hour=int(best_row["hour"]),
        relevance_score=float(best_row["predicted_score"]),
        is_cold_start=False,
        strategy_used="XGBRanker LTR (Learning-to-Rank)",
        top_3_slots=top_3,
    )


# -----------------------------------------------------------------------------
# 6. EXECUÇÃO PRINCIPAL E EXPLICAÇÃO DIDÁTICA
# -----------------------------------------------------------------------------

def print_architecture_explanation() -> None:
    banner = """
================================================================================
    SEND TIME OPTIMIZATION (STO) | LEARNING-TO-RANK vs. REGRESSÃO LINEAR
================================================================================
Por que usar Learning-to-Rank (LTR / XGBRanker) em vez de Regressão Clássica?

1. Objetivo de Negócio:
   Em STO, precisamos escolher O MELHOR slot (ou Top-K slots) para cada Lead,
   e NÃO prever o número exato de aberturas com precisão contínua.
   - Regressão Linear/MSE: penaliza erros em magnitudes absolutas (ex: prever 0.2
     vs 0.4 de abertura em horários irrelevantes consome perda do modelo).
   - LTR (LambdaMART/NDCG): otimiza a ordem relativa dos candidatos dentro de cada
     grupo (lead_id). Trocar o 1º com o 2º lugar gera alta penalidade, enquanto
     a ordem entre o 140º e o 150º slot é irrelevante.

2. Heterogeneidade e Desbalanceamento entre Leads:
   - Um lead ultra-engajado abre 50 e-mails; um lead frio abre 1 e-mail.
   - Na regressão global, o lead muito ativo domina a função de perda e distorce
     os pesos para todos os outros leads.
   - No XGBRanker com `group=group_sizes`, o cálculo de ranking e NDCG é feito
     ISOLADAMENTE dentro da lista de 168 slots de cada lead!

3. NDCG (Normalized Discounted Cumulative Gain):
   - Métrica padrão para ranking de busca e recomendação.
   - Aplica desconto logarítmico na posição: acertar o melhor horário na posição 1
     gera o ganho máximo da campanha.
================================================================================
"""
    print(banner)


def main() -> None:
    print_architecture_explanation()

    # 1. Geração de Dados
    print("[1/5] Gerando base sintética com leads e slots candidatos (7 dias x 24h = 168 slots/lead)...")
    df, tenant_affinity = generate_synthetic_sto_data(num_leads=80, random_seed=42)
    print(f"      Total de registros gerados: {len(df):,} linhas ({df['lead_id'].nunique()} leads).")

    # 2. Split por Lead
    print("[2/5] Dividindo base em 80% treino e 20% teste (Split estrito por lead_id)...")
    train_df, test_df = split_train_test_by_lead(df, train_ratio=0.8, random_seed=42)
    n_train_leads = train_df["lead_id"].nunique()
    n_test_leads = test_df["lead_id"].nunique()
    print(f"      Leads no Treino: {n_train_leads} ({len(train_df):,} slots)")
    print(f"      Leads no Teste:  {n_test_leads} ({len(test_df):,} slots)")

    # 3. Treinamento XGBRanker
    print("[3/5] Treinando XGBRanker com objetivo 'rank:ndcg' agrupado por lead_id...")
    ranker, group_sizes = train_xgb_ranker(train_df)
    print(f"      Modelo treinado com sucesso! Grupos configurados: {len(group_sizes)}.")

    # 4. Avaliação e Comparação com Baseline
    print("[4/5] Avaliando métrica NDCG no conjunto de teste contra Baseline de Tenant...")
    eval_results = evaluate_ranking(ranker, test_df)

    print("\n" + "-" * 70)
    print(f"{'Métrica':<20} | {'Baseline (Tenant)':<18} | {'XGBRanker (LTR)':<18} | {'Ganho (Lift)':<10}")
    print("-" * 70)
    print(
        f"{'NDCG Global':<20} | {eval_results['baseline_mean_ndcg']:<18.4f} | "
        f"{eval_results['model_mean_ndcg']:<18.4f} | "
        f"{((eval_results['model_mean_ndcg'] - eval_results['baseline_mean_ndcg']) / eval_results['baseline_mean_ndcg']) * 100:+.2f}%"
    )

    for k, metrics in eval_results["by_k"].items():
        print(
            f"{f'NDCG@{k}':<20} | {metrics['baseline_ndcg']:<18.4f} | "
            f"{metrics['model_ndcg']:<18.4f} | "
            f"{metrics['lift_pct']:+.2f}%"
        )
    print("-" * 70 + "\n")

    # 5. Simulação de Inferência em Tempo Real
    print("[5/5] Simulando inferência em tempo real para perfis de leads...")

    # Cenário A: Lead Ativo (Hábito noturno)
    lead_ativo = {
        "lead_id": "lead_user_night_owl",
        "lead_total_opens": 45,
    }
    rec_ativo = predict_best_send_time(lead_ativo, ranker, tenant_affinity)
    print(f"\n>> Cenário A: Lead Engajado ({lead_ativo['lead_id']}, {lead_ativo['lead_total_opens']} aberturas)")
    print(f"   Estratégia: {rec_ativo.strategy_used}")
    print(f"   Horário Recomendado: {rec_ativo.recommended_day_name} às {rec_ativo.recommended_hour:02d}:00")
    print(f"   Score Predito: {rec_ativo.relevance_score:.4f}")
    print("   Top 3 Slots:")
    for rank, slot in enumerate(rec_ativo.top_3_slots, 1):
        print(f"     #{rank} - {slot['day_name']} às {slot['hour']:02d}:00 (Score: {slot['score']:.4f})")

    # Cenário B: Lead Cold-Start (Sem histórico)
    lead_cold = {
        "lead_id": "lead_user_newcomer",
        "lead_total_opens": 0,
    }
    rec_cold = predict_best_send_time(lead_cold, ranker, tenant_affinity)
    print(f"\n>> Cenário B: Lead Cold-Start ({lead_cold['lead_id']}, {lead_cold['lead_total_opens']} aberturas)")
    print(f"   Estratégia: {rec_cold.strategy_used}")
    print(f"   Horário Recomendado (Fallback): {rec_cold.recommended_day_name} às {rec_cold.recommended_hour:02d}:00")
    print(f"   Score Baseline: {rec_cold.relevance_score:.4f}")
    print("   Top 3 Slots:")
    for rank, slot in enumerate(rec_cold.top_3_slots, 1):
        print(f"     #{rank} - {slot['day_name']} às {slot['hour']:02d}:00 (Score: {slot['score']:.4f})")

    print("\nExecução concluída com sucesso!")


if __name__ == "__main__":
    main()
