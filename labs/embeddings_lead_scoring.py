import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Carrega um modelo leve especializado em gerar embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Cenário: Frases reais capturadas em formulários de leads
leads = [
    "Quero integrar a API do produto ao meu backend imediatamente",  # Lead 1: Foco técnico/urgência
    "Gostaria de ver a documentação técnica para desenvolvedores",   # Lead 2: Técnico
    "Qual o desconto para contratação anual do plano básico?",        # Lead 3: Preço/Compras
    "Gostaria de agendar uma reunião comercial com o time de vendas", # Lead 4: Comercial
]

# 3. Gerando os tensores (matriz de vetores numéricos)
# Cada frase vira um vetor de 384 dimensões (números float32)
embeddings = model.encode(leads)

print(f"Formato da matriz de embeddings: {embeddings.shape}")
print(f"Primeiros 5 valores do Lead 1: {embeddings[0][:5]}\n")


# 4. Função matemática de similaridade de cosseno (produto escalar normalizado)
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# 5. Cenário de Marketing: Criação de campanha segmentada
# O time cria uma nova campanha de inbound: "Campanha Dev Tools & APIs"
campanha_alvo = "Documentação de API para engenharia de software"
campanha_vetor = model.encode(campanha_alvo)

print(f"--- Roteamento de Leads para a Campanha: '{campanha_alvo}' ---")

# Comparamos o vetor da campanha com o vetor de cada lead que chegou
for lead_texto, lead_vetor in zip(leads, embeddings):
    score = cosine_similarity(campanha_vetor, lead_vetor)
    
    # Regra de automação de marketing baseada no score vetorial:
    status = "Qualificado para a campanha (MQL)" if score > 0.50 else "Descartar / Outra campanha"
    print(f"Score: {score:.4f} | {status} | Lead: '{lead_texto}'")