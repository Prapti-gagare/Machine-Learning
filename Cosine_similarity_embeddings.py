from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2') 

resume_vec = model.encode([resume_text])
jd_vec = model.encode([jd_text])

score = cosine_similarity(resume_vec, jd_vec)[0][0]
print(f"Semantic Match Score: {round(score * 100, 2)}%")