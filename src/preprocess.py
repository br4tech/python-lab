"""
Etapa 1 do Pipeline de MLOps: Preprocessamento de Dados
------------------------------------------------------
Responsabilidade:
- Ler os dados brutos (data/raw/)
- Validar e tratar valores
- Dividir em dados de Treino (80%) e Teste (20%)
- Salvar os dados processados em (data/processed/)
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/raw/salaries.csv")
PROCESSED_DIR = Path("data/processed")


def load_raw_data(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    # Leitura simples com numpy (sem cabeçalho)
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, 0:1]  # Features (anos de experiência)
    y = data[:, 1]    # Target (salário)
    return X, y


def run_preprocessing():
    print("⏳ [1/4] Iniciando Pré-processamento dos Dados...")
    X, y = load_raw_data(RAW_DATA_PATH)
    print(f"   -> Dados brutos carregados: {len(X)} registros")

    # Divisão em Treino (estudo) e Teste (prova surpresa)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Junta as colunas para salvar
    train_data = np.column_stack((X_train, y_train))
    test_data = np.column_stack((X_test, y_test))

    header = "anos_experiencia,salario"
    np.savetxt(PROCESSED_DIR / "train.csv", train_data, delimiter=",", header=header, comments="")
    np.savetxt(PROCESSED_DIR / "test.csv", test_data, delimiter=",", header=header, comments="")

    print(f"   -> Treino salvo em: {PROCESSED_DIR / 'train.csv'} ({len(train_data)} linhas)")
    print(f"   -> Teste salvo em: {PROCESSED_DIR / 'test.csv'} ({len(test_data)} linhas)")
    print("✅ Pré-processamento concluído com sucesso!\n")


if __name__ == "__main__":
    run_preprocessing()
