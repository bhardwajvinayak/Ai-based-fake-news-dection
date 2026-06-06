import json
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

print("Loading Dataset...")

with open(
    "train.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)

documents = []

for item in data:

    claim = item.get(
        "claim",
        ""
    )

    justification = item.get(
        "justification",
        ""
    )

    label = item.get(
        "label",
        ""
    )

    speaker = item.get(
        "speaker",
        ""
    )

    document = f"""
CLAIM:
{claim}

JUSTIFICATION:
{justification}

LABEL:
{label}

SPEAKER:
{speaker}
"""

    documents.append(
        document
    )

print(
    f"Documents Loaded: {len(documents)}"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(
    "Generating Embeddings..."
)

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings
).astype(
    "float32"
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)

faiss.write_index(
    index,
    "averitec_index.bin"
)

with open(
    "documents.pkl",
    "wb"
) as f:

    pickle.dump(
        documents,
        f
    )

print(
    "FAISS Index Saved"
)