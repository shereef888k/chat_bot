from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Multilingual model (English + Malayalam + Manglish)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ✅ Questions for retrieval (many ways users can ask)
QUESTIONS = [
    "manual course fee",
    "manual testing fee",
    "fee of manual course",
    "manual fee ethra",
    "manual course fee ethra",
    "manual testing course fee ethra",

    "selenium course fee",
    "automation course fee",
    "java selenium fee",
    "selenium fee ethra",
    "automation fee ethra",

    "placement",
    "placement undo",
    "placement assistance undo",

    "certificate",
    "certificate kittumo",
    "course certificate undo",

    "online class",
    "online undo",
    "offline undo",
    "online and offline"
]

# ✅ Fixed answers (predefined)
ANSWERS = {
    "manual_fee": "✅ Manual Testing course fee is ₹10,000 and duration is 3 months.",
    "selenium_fee": "✅ Java & Selenium Automation course fee is ₹18,000 and duration is 3 months.",
    "placement": "✅ Placement assistance is provided.",
    "certificate": "✅ Course completion certificate is provided.",
    "mode": "✅ Online & Offline classes available. Night & Sunday batches also available."
}

# ✅ Map each question to a fixed answer key
Q_TO_KEY = [
    "manual_fee","manual_fee","manual_fee","manual_fee","manual_fee","manual_fee",
    "selenium_fee","selenium_fee","selenium_fee","selenium_fee","selenium_fee",
    "placement","placement","placement",
    "certificate","certificate","certificate",
    "mode","mode","mode","mode"
]

# Build embeddings for questions (only)
embeddings = model.encode(QUESTIONS)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "qtest_index.faiss")

with open("qa_store.pkl", "wb") as f:
    pickle.dump({"QUESTIONS": QUESTIONS, "Q_TO_KEY": Q_TO_KEY, "ANSWERS": ANSWERS}, f)

print("✅ Vector Q/A store created successfully!")