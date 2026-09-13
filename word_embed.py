from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = SentenceTransformer("all-MiniLM-L6-v2")

words = [
    "king", "queen", "man", "woman",
    "apple", "orange", "car", "bus"
]

embeddings = model.encode(words)

print("Embedding shape:", embeddings.shape)

pairs = [
    ("king", "queen"),
    ("man", "woman"),
    ("apple", "orange"),
    ("car", "bus"),
    ("king", "apple")
]

for w1, w2 in pairs:
    i = words.index(w1)
    j = words.index(w2)

    score = cosine_similarity(
        [embeddings[i]],
        [embeddings[j]]
    )[0][0]

    print(f"{w1} <-> {w2}: {score:.4f}")

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(9, 6))
plt.scatter(reduced[:, 0], reduced[:, 1])

for i, word in enumerate(words):
    plt.annotate(
        word,
        (reduced[i, 0], reduced[i, 1])
    )

plt.title("Word Embedding Visualization using PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.show()