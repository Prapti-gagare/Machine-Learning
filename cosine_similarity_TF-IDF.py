from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resume_text = "Experienced Python developer skilled in Django, REST APIs, and SQL"
jd_text = "Looking for Python developer with Django and PostgreSQL experience"

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform([resume_text, jd_text])

score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
print(f"ATS Match Score: {round(score * 100, 2)}%")