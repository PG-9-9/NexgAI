from langchain_huggingface import HuggingFaceEmbeddings
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)
test_text = ["This is a test question."]
embeddings = embedder.embed_documents(test_text)
print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")