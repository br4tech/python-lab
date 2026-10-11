"""
Etapa 2 do Pipeline de MLOps: Treinamento e Registro de Modelo
--------------------------------------------------------------
Responsabilidade:
- Ler os dados de treino preparados (data/processed/train.csv)
- Treinar o algoritmo de Machine Learning
- Serializar (salvar) o artefato do modelo treinado em (models/)
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

TRAIN_DATA_PATH = Path("data/processed/train.csv")
MODELS_DIR = Path("models")
MODEL_ARTIFACT_PATH = MODELS_DIR / "salary_model.joblib"


def run_training():
    print("⏳ [2/4] Iniciando Treinamento do Modelo...")

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de treino não encontrado em {TRAIN_DATA_PATH}. "
            "Execute 'uv run python src/preprocess.py' primeiro!"
        )

    # Carrega dados de treino
    data = np.genfromtxt(TRAIN_DATA_PATH, delimiter=",", skip_header=1)
    X_train = data[:, 0:1]
    y_train = data[:, 1]

    # Instancia e ajusta o modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    peso = model.coef_[0]
    vies = model.intercept_
    print(f"   -> Padrão aprendido: Salário = ({peso:.2f} * Anos_Exp) + {vies:.2f}")

    # Salva o artefato do modelo no diretório models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_ARTIFACT_PATH)

    print(f"   -> Artefato do modelo salvo em: {MODEL_ARTIFACT_PATH}")
    print("✅ Treinamento e registro concluídos com sucesso!\n")


if __name__ == "__main__":
    run_training()
