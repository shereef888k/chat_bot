from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

index = faiss.read_index("qtest_index.faiss")

with open("qa_store.pkl", "rb") as f:
    store = pickle.load(f)

QUESTIONS = store["QUESTIONS"]
Q_TO_KEY = store["Q_TO_KEY"]
ANSWERS = store["ANSWERS"]

def get_answer(query: str) -> str:
    q_vec = model.encode([query])
    D, I = index.search(np.array(q_vec), k=1)

    # Safety threshold (avoid wrong replies)
    if D[0][0] > 1.2:
        return "Please contact our office for more details. 📞 9961 544 424"

    best_i = int(I[0][0])
    key = Q_TO_KEY[best_i]
    return ANSWERS[key]