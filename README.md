
# GPT-2 Embedding Manipulation: Capitalization Vector

This project demonstrates a practical and interpretable modification of token embeddings in the DistilGPT-2 model to control **capitalization behavior** via direct vector editing.

## 🧠 What This Project Shows

We identified specific dimensions in the GPT-2 embedding space that correlate with **capitalization**, and applied this knowledge to:

- Extract the embedding vectors from the DistilGPT-2 model
- Calculate the difference vector between a lowercase and capitalized form of a token (called `delta_caps`)
- Manually modify the embedding of a token (e.g., turning "he" into "He")
- Inject the modified embeddings back into the model
- Generate text to observe the effect of this manual change

---

## 🔬 Background

Language models like GPT-2 use fixed-length embedding vectors to represent tokens. These embeddings encode not just meaning but also **textual form**, such as capitalization or word boundaries.

By analyzing multiple tokens, we discovered certain embedding dimensions (e.g., 362, 138, 92...) that consistently react to capitalization.

---

## 📁 Files Included

- `alterar_embedding_distilgpt2.py`: Script to load the model, apply `delta_caps`, and generate text
- `distilgpt2_embeddings.jsonl`: Token embeddings extracted from the model
- `analise_dimensoes_dinamico.py`: Script to identify capitalization dimensions
- Example logs and delta analyses for many tokens

---

## ▶️ How to Run

### Requirements
```
pip install torch transformers numpy
```

### Modify a token's embedding
```bash
python alterar_embedding_distilgpt2.py
```
This will:
- Apply the `delta_caps` vector to the embedding of "he"
- Replace it in the model
- Generate text starting with "he is a"

### Output Example
```
he is a very good person. He is a very good person. He is...
```

The model begins using the capitalized form naturally, even though the input was lowercase.

---

## 📈 Insights

- We found that **capitalization is not a flag but a learned semantic shift**
- The same transformation vector (`delta_caps`) works across many tokens
- This method opens doors to style manipulation, text control, and interpretability

---

## 📌 Potential Next Steps

- Apply similar techniques for tone, sentiment, or syntax
- Build a toolkit to generate custom embeddings
- Use PCA/t-SNE to visualize embedding behavior across forms
- Extend to other models (GPT-J, LLaMA, etc.)

---

## 🙋‍♂️ Author
This project was developed through practical experimentation and discovery using open weights from HuggingFace's `distilgpt2`.

Feel free to contribute, fork, or build upon it!
