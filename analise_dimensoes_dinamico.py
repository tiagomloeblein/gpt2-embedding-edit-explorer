import json
import pandas as pd
import numpy as np
from collections import Counter
import sys
from itertools import combinations
from numpy.linalg import norm

# === CONFIGURAÇÃO ===
ARQUIVO_EMBEDDINGS = "distilgpt2_embeddings.jsonl"
#PALAVRAS_BASE = ["pure", "the", "he", "in", "on", "a", "e", "i", "o", "u", "b", "at", "it", "is", "an", "or", "en", "es"]
#PALAVRAS_BASE = ["to", "and", "be", "am", "you", "we", "are", "with", "as", "if", "her", "him", "not", "no", "yes", "from", "have", "end", "all"]
#PALAVRAS_BASE = ["red", "our", "pt", "can", "go", "but", "ok", "will", "one", "who", "for", "police", "bit", "work", "god", "state", "back", "ball"]
PALAVRAS_BASE = ["cent", "face", "president", "app", "true", "false", "front","below","me", "id", "pet"]


ARQUIVO_LOG = "log_analise4.txt"

# === GERA VARIAÇÕES ===
def gerar_variacoes(palavra):
    return [
        palavra.lower(),
        palavra.capitalize(),
        palavra.upper(),
        "\u0120" + palavra.lower(),
        "\u0120" + palavra.capitalize(),
        "\u0120" + palavra.upper()
    ]

TOKENS_INTERESSE = []
for p in PALAVRAS_BASE:
    TOKENS_INTERESSE.extend(gerar_variacoes(p))

# === LOG BUFFER
log = []

def logprint(msg):
    print(msg)
    log.append(msg)

# === LEITURA DOS EMBEDDINGS ===
logprint(f"🔍 Procurando {len(TOKENS_INTERESSE)} tokens derivados de: {PALAVRAS_BASE}")
embeddings = {}
with open(ARQUIVO_EMBEDDINGS, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        token = data["token"]
        if token in TOKENS_INTERESSE:
            embeddings[token] = data["embedding"]
        if len(embeddings) == len(TOKENS_INTERESSE):
            break

encontrados = sorted(embeddings.keys())
logprint(f"✅ Tokens encontrados ({len(encontrados)}): {[repr(t) for t in encontrados]}")
if not encontrados:
    logprint("⚠️ Nenhum token encontrado.")
    sys.exit(1)

df = pd.DataFrame({token: embeddings[token] for token in encontrados})
df.index.name = "Dimensão"

# === DELTAS ENTRE VARIAÇÕES
logprint("\n🧠 Deltas entre formas (maiusc/minusc ou espaço):")

for base in PALAVRAS_BASE:
    colunas = {
        "min": base.lower(),
        "cap": base.capitalize(),
        "up": base.upper(),
        "Ġmin": "\u0120" + base.lower(),
        "Ġcap": "\u0120" + base.capitalize(),
        "Ġup": "\u0120" + base.upper(),
    }
    for a, b in [("min", "cap"), ("min", "up"), ("min", "Ġmin"), ("cap", "Ġcap"), ("up", "Ġup")]:
        if colunas.get(a) in df.columns and colunas.get(b) in df.columns:
            delta = df[colunas[b]] - df[colunas[a]]
            top = delta.abs().nlargest(5)
            logprint(f"\n🔹 {colunas[b]} - {colunas[a]} (Top 5 Δ):")
            for dim, val in top.items():
                logprint(f"  Dim {dim:3d} → Δ = {val:.5f}")

# === SIMILARIDADE ENTRE TOKENS
logprint("\n📐 Similaridade cosseno entre todos os tokens:")
def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (norm(a) * norm(b))

for t1, t2 in combinations(df.columns, 2):
    sim = cosine_sim(df[t1], df[t2])
    logprint(f"Cosine({repr(t1)} vs {repr(t2)}) = {sim:.6f}")

# === SIMILARIDADE ENTRE DELTAS DE CAPITALIZAÇÃO
logprint("\n🧪 Similaridade entre vetores delta_caps (min → cap):")
delta_caps = {}
for base in PALAVRAS_BASE:
    t_min = base.lower()
    t_cap = base.capitalize()
    if t_min in df.columns and t_cap in df.columns:
        delta = df[t_cap] - df[t_min]
        delta_caps[base] = delta

for b1, b2 in combinations(delta_caps.keys(), 2):
    sim = cosine_sim(delta_caps[b1], delta_caps[b2])
    logprint(f"Cosine(delta_caps[{b1}] vs delta_caps[{b2}]) = {sim:.6f}")

# === PADRÕES DE SINAL
logprint("\n🔍 Padrões de sinal (+/-):")
sign_matrix = df.applymap(lambda x: "+" if x >= 0 else "-")
padroes = sign_matrix.apply(lambda row: "".join(row.values), axis=1)
padrao_freq = Counter(padroes)

logprint("\n📊 Padrões mais comuns:")
for padrao, freq in padrao_freq.most_common(10):
    logprint(f"{padrao} → {freq} dimensões")

consistentes = [dim for dim, padrao in padroes.items()
                if all(c == "+" for c in padrao) or all(c == "-" for c in padrao)]
logprint(f"\n✅ Dimensões com sinal consistente total: {len(consistentes)}")
if consistentes:
    logprint(f"Exemplo: {consistentes[:10]}")

# === INVERSÃO ENTRE MINÚSCULA / MAIÚSCULA
logprint("\n🔎 Inversão minúscula ↔ maiúscula em todas as palavras:")
invertidas = []
for dim in df.index:
    sinais = []
    for base in PALAVRAS_BASE:
        low = base.lower()
        cap = base.capitalize()
        if low in df.columns and cap in df.columns:
            s_low = "+" if df.at[dim, low] >= 0 else "-"
            s_cap = "+" if df.at[dim, cap] >= 0 else "-"
            sinais.append((s_low, s_cap))
    if sinais and all(a != b for a, b in sinais):
        invertidas.append(dim)

logprint(f"📈 Dimensões com inversão entre minúscula/maiúscula em todas as palavras: {len(invertidas)}")
if invertidas:
    logprint(f"Exemplo: {invertidas[:10]}")

logprint("\n✅ Análise concluída.")

# === SALVAR LOG
with open(ARQUIVO_LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(log))
logprint(f"\n📝 Log salvo em: {ARQUIVO_LOG}")
