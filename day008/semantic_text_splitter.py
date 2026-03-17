import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

text = """
    Farmers were working hard in the fields, preparing the soil and planting seeds for next season.
    The sun was bright, and the air smelled of earth and fresh grass.
    IPL is the biggest cricket league. People all over the world watch the mathes and cheer for favourite team.

    Terrorism is a big danger to peace and safety. It cause harm to peoples and create fear in cities and villages.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                        chunk_overlap=0,
                        separators=["\n\n", "\n", "."])
results = splitter.split_text(text=text)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
document_embeddings = embeddings.embed_documents(results)

document_embeddings_matrix = np.array(document_embeddings)
similarity = cosine_similarity(document_embeddings_matrix, document_embeddings_matrix)

print(similarity)
print(np.std(similarity))

threshold = 0.10
chunks = []
current_chunk = [results[0]]

for i in range(1, len(results)):
    sim = cosine_similarity([document_embeddings[i-1]],[document_embeddings[i]])[0][0]

    if sim < threshold:
        chunks.append(" ".join(current_chunk))
        current_chunk = [results[i]]
    else:
        current_chunk.append(results[i])

chunks.append(" ".join(current_chunk))
print("Semantic Chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}\n")