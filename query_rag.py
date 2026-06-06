import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "averitec_index.bin"
)

with open(
    "documents.pkl",
    "rb"
) as f:

    documents = pickle.load(f)

print(
    "\nRAG Ready\n"
)

while True:

    query = input(
        "Enter Claim: "
    )

    if query.lower() == "exit":
        break

    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype(
        "float32"
    )

    distances, indices = index.search(
        query_embedding,
        3
    )

    print("\nTop Evidence\n")

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):

        print("=" * 70)

        print(
            f"Result {rank}"
        )

        print(
            documents[idx]
        )

        print()