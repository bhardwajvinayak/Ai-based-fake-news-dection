from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample knowledge base
documents = [
    "PIB Fact Check: No scheme exists that gives ₹50,000 to every citizen.",
    "Government of India denied the viral message about free money.",
    "Fact-checking agencies classified the claim as false.",
    "Citizens should verify information before sharing on social media."
]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for documents
doc_embeddings = model.encode(documents)

# Create FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add vectors to FAISS
index.add(np.array(doc_embeddings))

# User query
query = input("Enter claim: ")

# Convert query to embedding
query_embedding = model.encode([query])

# Search top 2 similar documents
k = 2
distances, indices = index.search(np.array(query_embedding), k)

print("\nRetrieved Evidence:\n")

for i in indices[0]:
    print("-", documents[i])