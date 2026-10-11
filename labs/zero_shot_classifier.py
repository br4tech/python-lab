from transformers import pipeline

# 1. Carrega uma pipeline de Zero-Shot Classification (Cross-Encoder / NLI)
# Usamos um modelo leve, rápido e multilingue (ótimo para português)
print("Carregando o classificador NLI...")
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

# 2. Gabaritos / Categorias de destino
candidate_labels = [
    "Dúvida técnica de infraestrutura e API",
    "Negociação financeira e desconto",
    "Contato direto com time de vendas comercial",
    "Cancelamento ou reclamação de suporte"
]

# 3. Frases de teste (incluindo negações e contextos mistos)
leads = [
    "Gostaria de ver o preço da ferramenta, mas de jeito nenhum quero falar com vendedor agora.",
    "Preciso documentar os endpoints da API para autenticar nossos microsserviços.",
    "A plataforma caiu e não consigo emitir faturas no sistema de produção!",
    "Estamos avaliando renovar o contrato anual se houver margem para 15% de desconto."
]

# 4. Avaliando cada lead
# hypothesis_template formata a relação lógica de NLI:
# Premissa: [Texto do Lead] -> Hipótese: "Este texto é sobre [Categoria]"
template = "Este texto é sobre {}."

print("\n=== CLASSIFICAÇÃO NLI (ZERO-SHOT) ===")
for lead in leads:
    result = classifier(
        lead,
        candidate_labels,
        hypothesis_template=template,
        multi_label=False # Retorna uma distribuição probabilística somando 1.0 (100%)
    )
    
    print(f"\nLead: \"{lead}\"")
    print("Probabilidades por categoria:")
    for label, score in zip(result["labels"], result["scores"]):
        marcador = "->" if score == max(result["scores"]) else "  "
        print(f" {marcador} {label}: {score * 100:.2f}%")