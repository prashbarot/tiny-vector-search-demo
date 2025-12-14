# Tiny Vector Search Demo
# Author: Prashant Barot
# Goal: Explore AI vector similarity on small text data

from sentence_transformers import SentenceTransformer, util

# My small dataset
sentences = [
    "Oracle Cloud helps enterprises migrate workloads.",
    "Databricks provides cloud data analytics.",
    "AI vector search improves information retrieval."
]

# Initialize pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

# My query
query = "How can AI help search data efficiently?"

query_embedding = model.encode(query, convert_to_tensor=True)

# Find most similar sentence
cos_scores = util.cos_sim(query_embedding, embeddings)
best_match_idx = cos_scores.argmax()

print("Query:", query)
print("Best match:", sentences[best_match_idx])
