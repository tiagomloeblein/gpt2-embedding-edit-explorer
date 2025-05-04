import json
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# === CONFIGURAÇÕES ===
ARQUIVO_EMBEDDINGS = "distilgpt2_embeddings.jsonl"  # caminho do seu .jsonl
TOKEN_ALVO = "he"  # token que você quer alterar
TOKEN_CAP = "He"   # forma capitalizada correspondente
GERAR_TEXTO = True

# === 1. Carregar modelo e tokenizer ===
print("🔄 Carregando modelo distilgpt2...")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()

# === 2. Criar matriz de embeddings vazia ===
embedding_matrix = torch.zeros(model.config.vocab_size, model.config.n_embd)

# === 3. Carregar embeddings do .jsonl ===
print("📥 Lendo embeddings do arquivo...")
with open(ARQUIVO_EMBEDDINGS, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        token = data["token"]
        idx = tokenizer.convert_tokens_to_ids(token)
        if idx is not None and idx < model.config.vocab_size:
            embedding_matrix[idx] = torch.tensor(data["embedding"])

# === 4. Calcular vetor delta_caps = cap - min ===
print("📊 Calculando delta_caps...")
id_min = tokenizer.convert_tokens_to_ids(TOKEN_ALVO)
id_cap = tokenizer.convert_tokens_to_ids(TOKEN_CAP)

if id_min is None or id_cap is None:
    raise ValueError("Não foi possível localizar os tokens no tokenizer.")

vec_min = embedding_matrix[id_min]
vec_cap = embedding_matrix[id_cap]
delta_caps = vec_cap - vec_min

# === 5. Aplicar delta_caps ao token desejado ===
print(f"🛠️ Aplicando delta_caps ao token '{TOKEN_ALVO}'...")
embedding_matrix[id_min] = vec_min + delta_caps

# === 6. Substituir os embeddings no modelo ===
model.get_input_embeddings().weight.data = embedding_matrix.clone().detach()
model.get_input_embeddings().weight.requires_grad = False

# === 7. Gerar texto com o modelo atualizado ===
if GERAR_TEXTO:
    prompt = "he is a"
    print(f"\n🚀 Prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n📄 Texto gerado:\n{generated_text}")

print("\n✅ Embedding modificado com sucesso.")
