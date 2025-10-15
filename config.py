import os

# Config to determine generation source
GEN_TYPE = "GROQ" # or "LOCAL"

# Absolute path to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# local models related directories
AGENTS_DIR = os.path.join(BASE_DIR, "agents")
LOCAL_MODELS_DIR = os.path.join(AGENTS_DIR, "local_models")

# Generation config
GEN_LLM = "llama3-8b-8192"
GEN_TYPES = {
    "GROQ": GEN_LLM,  # Groq expects model string
    "LOCAL": os.path.join(LOCAL_MODELS_DIR, GEN_LLM)  # llama.cpp expects path
}

# RAG Configs
RAG_EMBED_MODEL = "arctic-embed-m"
EMBED_MODEL_PATH = os.path.join(LOCAL_MODELS_DIR, RAG_EMBED_MODEL)
CHROMA_DB_DIR = os.path.join(AGENTS_DIR, "policy_vector_db")
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
NUM_OF_DOCS_RETRIEVED = 3
COLLECTION_CATEGORIES = ["HR", "IT", "Finance"]


