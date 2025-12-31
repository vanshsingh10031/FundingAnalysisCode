import os
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# -------------------------------------------------
# Configure LOCAL LLM (Ollama)
# -------------------------------------------------
Settings.llm = Ollama(
    model="gemma:2b",
    temperature=0,
    request_timeout=120,
    additional_kwargs={
        "num_predict": 300,
        "num_ctx": 2048
    }
)




Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# -------------------------------------------------
# Load documents recursively from data/ subfolders
# -------------------------------------------------
DATA_DIR = "data"

all_documents = []

for root, _, files in os.walk(DATA_DIR):
    if files:
        reader = SimpleDirectoryReader(
            input_dir=root,
            recursive=False
        )
        docs = reader.load_data()
        for doc in docs:
            # Add folder-level provenance (VERY IMPORTANT)
            doc.metadata["source_folder"] = root.replace("\\", "/")
            all_documents.append(doc)

if not all_documents:
    raise RuntimeError(
        "No documents found in data/ subfolders. "
        "Please add funding-related documents."
    )

documents = all_documents

# -------------------------------------------------
# Chunk documents
# -------------------------------------------------
parser = SentenceSplitter(
    chunk_size=300,
    chunk_overlap=30
)

nodes = parser.get_nodes_from_documents(documents)

# -------------------------------------------------
# Setup ChromaDB (local, in-memory for hackathon)
# -------------------------------------------------
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="funding_rag"
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# -------------------------------------------------
# Build Vector Index (ONCE)
# -------------------------------------------------
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context
)

query_engine = index.as_query_engine(
    similarity_top_k=4,          # reduce from 6
    response_mode="tree_summarize"
)


# -------------------------------------------------
# Core Funding Analysis Function (used by FastAPI)
# -------------------------------------------------
def analyze_funding(
    startup_description: str,
    stage: str,
    sector: str,
    geography: str,
    funding_goal: str
):
    query = f"""
You are an AI Investment Analyst specializing in startup funding.

STARTUP DETAILS:
Description:
{startup_description}

Stage:
{stage}

Sector:
{sector}

Geography:
{geography}

Funding Goal:
{funding_goal}

TASK:
Using ONLY the provided context documents, perform the following:

1. Identify investors that are a strong match for this startup
2. Rank investors by relevance
3. Explain WHY each investor is a good fit
   (stage alignment, sector focus, thesis)
4. Mention any potential mismatches or risks
5. If no suitable investors are found, say so explicitly

RULES:
- Use ONLY retrieved context
- Do NOT hallucinate investors or facts
- Be factual and analytical
- Avoid marketing language
- If information is insufficient, state it clearly

OUTPUT FORMAT:

FUNDING ANALYSIS

Top Matching Investors:
1. Investor Name
   Reason:

2. Investor Name
   Reason:

Potential Risks / Mismatches:

Summary Recommendation:
"""

    response = query_engine.query(query)

    return {
        "analysis": response.response,
        "sources": sorted(
            set(
                f"{n.metadata.get('source_folder')}/"
                f"{n.metadata.get('file_name')}"
                for n in response.source_nodes
            )
        )
    }
