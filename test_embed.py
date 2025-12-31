from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Embedding loaded successfully")
