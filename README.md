# 🧪 Python Lab: Engenharia de Machine Learning & MLOps

Bem-vindo ao laboratório de **MLOps** e **Inteligência Artificial**! 🚀

Este projeto adota a arquitetura de produção padrão da indústria para Machine Learning, separando a esteira em etapas modulares e desacopladas: **dados**, **experimentação**, **pré-processamento**, **treinamento**, **avaliação**, **registro de artefatos** e **deployment em API**.

---

## 🏛️ Arquitetura do Projeto

Baseada no modelo canônico de engenharia de MLOps:

```text
python-lab/
├── data/                         # 📦 Armazenamento de dados
│   ├── raw/                      # Dados brutos imutáveis (salaries.csv)
│   └── processed/                # Dados limpos e particionados (train.csv, test.csv)
│
├── notebooks/                    # 🔬 EDA (Exploratory Data Analysis) e rascunhos de modelos
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/                          # ⚙️ Esteira de Produção de MLOps
│   ├── preprocess.py             # Limpeza, validação e partição (Treino/Teste)
│   ├── train.py                  # Treino do modelo e serialização do artefato
│   ├── evaluate.py               # Avaliação de métricas e auditoria (RMSE, R2)
│   └── app.py                    # Deployment e Serving via API REST (Flask)
│
├── models/                       # 💾 Registro de artefatos gerados
│   ├── salary_model.joblib       # Modelo serializado pronto para inferência
│   └── metrics.json              # Relatório auditável de performance
│
├── labs/                         # 🧪 Estudos conceituais avançados de NLP, Ranking & LLMs
│   ├── scikit-learn.py           # O clássico monolítico (referência de estudo)
│   ├── send_time_ranker.py       # Learning-to-Rank (LTR com XGBRanker) para Send Time Optimization (STO)
│   ├── embeddings_lead_scoring.py# Vetorização e cálculo de similaridade de cosseno
│   ├── multi_campaign_routing.py # Roteamento em lote via álgebra matricial
│   ├── zero_shot_classifier.py   # Inferência NLI sem retreino (Zero-Shot)
│   └── test_llm.py               # Integração com LLM local via Ollama
│
├── pyproject.toml & uv.lock      # 🔒 Gerenciador de dependências e ambiente ultra-rápido (uv)
└── README.md                     # 📖 Documentação do laboratório
```

---

## 🧭 Como Executar a Esteira de MLOps (Passo a Passo)

Utilizamos o **`uv`** para garantir execução isolada e reprodutível.

### 1. Pré-processamento dos Dados
Lê `data/raw/salaries.csv`, divide em treino/teste e grava em `data/processed/`:
```bash
uv run python src/preprocess.py
```

### 2. Treinamento e Registro do Modelo
Lê `data/processed/train.csv`, ajusta os parâmetros e salva o artefato binário em `models/salary_model.joblib`:
```bash
uv run python src/train.py
```

### 3. Avaliação de Métricas e Auditoria
Carrega o modelo registrado, testa contra `data/processed/test.csv` e exporta o relatório `models/metrics.json`:
```bash
uv run python src/evaluate.py
```

### 4. Servir o Modelo em Produção (API REST com Flask)
Inicia o servidor de inferência em tempo real:
```bash
# Iniciar o servidor HTTP na porta 5000:
uv run python src/app.py

# Ou testar uma predição rápida direta pelo terminal:
uv run python src/app.py --test
```

#### Testando a API via cURL:
```bash
# Health Check:
curl http://localhost:5000/health

# Inferência em tempo real:
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"anos_experiencia": 6.5}'
```

---

## 🔬 Laboratórios Conceituais (`labs/`)

Os scripts de exploração e estudo conceitual estão organizados em `labs/`:

* **`uv run python labs/send_time_ranker.py`**: Otimização Inteligente de Envio (**Send Time Optimization - STO**) usando **Learning-to-Rank (LTR)** com `xgboost.XGBRanker`. Avaliação em 168 slots candidatos (7 dias x 24h) agrupados por lead, métricas NDCG e NDCG@k comparadas a baseline heurístico de tenant, e inferência em tempo real com fallback para Cold-Start.
* **`uv run python labs/embeddings_lead_scoring.py`**: Representação de textos em 384 dimensões e cálculo de afinidade semântica para qualificação de leads.
* **`uv run python labs/multi_campaign_routing.py`**: Multiplicação de matrizes (`np.matmul`) para triagem de centenas de leads contra múltiplas campanhas simultâneas.
* **`uv run python labs/zero_shot_classifier.py`**: Classificação lógica usando o modelo multilíngue `mDeBERTa-v3` para lidar com negações e contextos mistos sem dados de treino prévios.
* **`uv run python labs/test_llm.py`**: Conexão com modelos locais (Ollama + Docker) utilizando o SDK padrão de mercado da OpenAI.
