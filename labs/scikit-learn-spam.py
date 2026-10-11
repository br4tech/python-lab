from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. DADOS (Data)
emails = [
    "Ganhe dinheiro fácil trabalhando em casa agora",
    "Promoção imperdível, clique aqui e ganhe um prêmio",
    "Acesso grátis liberado para membros vip",
    "Reunião de alinhamento do projeto amanhã às 14h",
    "Segue em anexo o relatório financeiro do mês",
    "Você pode revisar o código do pull request?",
    "Desconto exclusivo e frete grátis só hoje",
    "Por favor, confirme sua presença na conferência",
]
labels = [
    "Spam",
    "Spam",
    "Spam",
    "Não Spam",
    "Não Spam",
    "Não Spam",
    "Spam",
    "Não Spam",
]

# 2. SEPARAÇÃO TREINO / TESTE
X_treino, X_teste, y_treino, y_teste = train_test_split(
    emails, labels, test_size=0.25, random_state=42
)

# 3. EXTRAÇÃO DE PADRÕES (Find Patterns)
# Converte texto cru em representação numérica (Bag of Words)
vetorizador = CountVectorizer()
X_treino_vetorizado = vetorizador.fit_transform(X_treino)

# 4. CONSTRUÇÃO DO MODELO (Build Models / Learn from Data)
modelo = MultinomialNB()
modelo.fit(X_treino_vetorizado, y_treino)

# 5. PREVISÕES EM DADOS NÃO VISTOS (Make Predictions)
novos_emails = [
    "Parabéns! Você ganhou um desconto exclusivo hoje",
    "Podemos marcar um alinhamento sobre o relatório?",
]

novos_emails_vetorizados = vetorizador.transform(novos_emails)
predicoes = modelo.predict(novos_emails_vetorizados)

for texto, pred in zip(novos_emails, predicoes):
    print(f"E-mail: '{texto}' -> Classificação: {pred}")