"""
Etapa 3 do Pipeline de MLOps: Avaliação do Modelo (Model Evaluation)
-------------------------------------------------------------------
Responsabilidade:
- Carregar o artefato do modelo salvo (models/salary_model.joblib)
- Carregar os dados de teste (data/processed/test.csv)
- Calcular métricas de performance (RMSE, R2)
- Salvar um relatório de métricas em json (models/metrics.json)
"""

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

TEST_DATA_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/salary_model.joblib")
METRICS_PATH = Path("models/metrics.json")


def run_evaluation():
    print("⏳ [3/4] Iniciando Avaliação do Modelo...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Artefato do modelo não encontrado em {MODEL_PATH}. "
            "Execute 'uv run python src/train.py' primeiro!"
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dados de teste não encontrados em {TEST_DATA_PATH}. "
            "Execute 'uv run python src/preprocess.py' primeiro!"
        )

    # Carrega modelo e dados de teste
    model = joblib.load(MODEL_PATH)
    data = np.genfromtxt(TEST_DATA_PATH, delimiter=",", skip_header=1)
    X_test = data[:, 0:1]
    y_test = data[:, 1]

    # Realiza as predições no teste
    y_pred = model.predict(X_test)

    # Calcula métricas
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "test_samples": len(y_test),
    }

    # Salva relatório de métricas
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"   -> Erro Médio (RMSE): R$ {rmse:.2f} mil")
    print(f"   -> Coeficiente de Determinação (R²): {r2:.4f}")
    print(f"   -> Relatório de métricas salvo em: {METRICS_PATH}")
    print("✅ Avaliação concluída com sucesso!\n")


if __name__ == "__main__":
    run_evaluation()
