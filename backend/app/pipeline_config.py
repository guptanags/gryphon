import os
from qdrant_client import models

# Paths
# Define a persistent cache directory for Repos (Mount this as PVC in OpenShift)
WORKSPACE_ROOT = os.environ.get("WORKSPACE_ROOT", "/tmp/ai_workspace_cache")

# Qdrant
QDRANT_HOST = os.environ.get("QDRANT_HOST", "https://6916eb5b-7766-4a48-bdca-409766ee522d.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_PORT = 6333
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
CODE_COLLECTION_NAME = "code"
DOCS_COLLECTION_NAME = "docs"
FILE_SUMMARY_COLLECTION_NAME = "file_summaries"

# Vertex AI
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID","vrittera")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
GENERATIVE_MODEL_NAME = "gemini-2.5-flash-lite"
VECTOR_DIMENSION = 768
VECTOR_METRIC = models.Distance.COSINE

# Processing
BATCH_SIZE = 50 # Reduced to avoid Token Limit errors
MAX_EMBED_CHARS = 15000  # Safe character length for a single embedding input (approx. tokens * 4)

TEXT_EXTENSIONS = ('.html', '.css', '.md', '.txt', '.xml', '.json')
