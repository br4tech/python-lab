import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Base de Campanhas ativas (seus múltiplos alvos)
campanhas = [
    {
        "id": "CAMP_DEV",
        "nome": "Developer Tools & APIs",
        "descricao": "Documentação técnica, SDKs, webhooks e integração de API para desenvolvedores",
    },
    {
        "id": "CAMP_FINANCE",
        "nome": "Condições Comerciais e Descontos",
        "descricao": "Preços, planos anuais, faturamento, propostas comerciais e negociação de descontos",
    },
    {
        "id": "CAMP_SALES_DEMO",
        "nome": "Demonstração com Executivo de Vendas",
        "descricao": "Agendamento de reunião executiva, demonstração da plataforma e contato com consultores",
    },
]

# 2. Leads que acabaram de entrar no funil
leads = [
    "Preciso de suporte para configurar a autenticação da API no meu backend",
    "Gostaria de agendar uma reunião comercial para conhecer o produto",
    "Qual o valor do plano para 50 usuários e como funciona o desconto anual?",
    "Queria apenas ler tutoriais sobre como criar automações em código",
]

# 3. Gerando matrizes de embeddings normalizadas
# (Quando os vetores são normalizados, o produto escalar é exatamente a similaridade de cosseno)
campanhas_textos = [c["descricao"] for c in campanhas]

# shape: (4, 384) -> 4 leads, cada um com 384 números
leads_embeddings = model.encode(leads, normalize_embeddings=True)

# shape: (3, 384) -> 3 campanhas, cada uma com 384 números
campanhas_embeddings = model.encode(campanhas_textos, normalize_embeddings=True)

# 4. Cálculo matricial direto: Matriz de Leads x Transposta da Matriz de Campanhas
# (4, 384) @ (384, 3) = Matriz de Scores (4, 3)
score_matrix = np.matmul(leads_embeddings, campanhas_embeddings.T)

# 5. Roteamento automático
print("=== ROTEAMENTO INTELIGENTE DE LEADS ===")
for i, lead in enumerate(leads):
    scores_do_lead = score_matrix[i]
    
    # Encontra o índice da campanha com maior pontuação para este lead
    melhor_indice = np.argmax(scores_do_lead)
    melhor_score = scores_do_lead[melhor_indice]
    campanha_vencedora = campanhas[melhor_indice]

    print(f"\nLead: \"{lead}\"")
    print(f"-> Encaminhado para: [{campanha_vencedora['nome']}] (Score: {melhor_score:.4f})")
    
    # Exibe a pontuação contra todas as outras campanhas
    for j, camp in enumerate(campanhas):
        marcador = "*" if j == melhor_indice else " "
        print(f"   [{marcador}] {camp['id']}: {scores_do_lead[j]:.4f}")