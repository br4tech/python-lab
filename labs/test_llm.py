from openai import OpenAI

# Ligação ao Ollama configurado via Docker
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "És um assistente de engenharia de software."},
        {"role": "user", "content": "Responde numa linha: qual a vantagem de utilizar Go em microsserviços?"}
    ]
)

print(response.choices[0].message.content)