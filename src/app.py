"""
Etapa 4 do Pipeline de MLOps: Deployment & Serving (API com Flask)
-----------------------------------------------------------------
Responsabilidade:
- Carregar o modelo persistido em produção (models/salary_model.joblib)
- Expor rotas HTTP REST para inferência em tempo real
- Disponibilizar endpoints de Health Check (essencial para Kubernetes e Cloud)
"""

import sys
from pathlib import Path
import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_PATH = Path("models/salary_model.joblib")

app = Flask(__name__)
model = None


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {MODEL_PATH}. "
                "Execute os scripts de treino antes de iniciar o servidor!"
            )
        model = joblib.load(MODEL_PATH)
    return model


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "servico": "MLOps Salary Prediction API",
        "status": "online",
        "endpoints": {
            "GET /health": "Verifica se o serviço e o modelo estão operacionais",
            "POST /predict": "Faz estimativa salarial. Body exemplo: {'anos_experiencia': 5.0}"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Healthcheck probe (Liveness/Readiness em MLOps)."""
    try:
        _ = get_model()
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint de inferência em tempo real."""
    data = request.get_json(silent=True)
    if not data or "anos_experiencia" not in data:
        return jsonify({
            "erro": "Requisição inválida. Envie um JSON com o campo 'anos_experiencia'."
        }), 400

    try:
        anos_exp = float(data["anos_experiencia"])
        loaded_model = get_model()
        
        # O modelo espera uma matriz 2D: [[valor]]
        entrada = np.array([[anos_exp]])
        predicao = float(loaded_model.predict(entrada)[0])

        return jsonify({
            "anos_experiencia": anos_exp,
            "salario_estimado": round(predicao, 2),
            "unidade": "R$ mil",
            "status": "sucesso"
        }), 200
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    # Suporte a teste rápido via CLI se chamado com argumento --test
    if "--test" in sys.argv:
        print("🧪 Executando teste rápido de inferência com o modelo...")
        m = get_model()
        test_val = 5.0
        res = m.predict(np.array([[test_val]]))[0]
        print(f"   -> Anos de Experiência: {test_val} anos")
        print(f"   -> Salário Estimado: R$ {res:.2f} mil")
        print("✅ Teste de inferência concluído com sucesso!")
    else:
        print("🚀 [4/4] Iniciando Servidor Flask de Inferência na porta 5000...")
        print("   -> Teste o endpoint: curl http://localhost:5000/health")
        app.run(host="0.0.0.0", port=5000, debug=False)
