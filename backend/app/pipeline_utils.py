import os
import shutil
import hashlib
import uuid
import logging
from git import Repo
from qdrant_client import QdrantClient, models
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pipeline_config import (
    QDRANT_API_KEY, QDRANT_HOST, QDRANT_PORT, CODE_COLLECTION_NAME, 
    DOCS_COLLECTION_NAME, FILE_SUMMARY_COLLECTION_NAME, VECTOR_DIMENSION, 
    VECTOR_METRIC, VERTEX_PROJECT_ID, VERTEX_LOCATION, EMBEDDING_MODEL_NAME, 
    GENERATIVE_MODEL_NAME, WORKSPACE_ROOT
)

log = logging.getLogger(__name__)

def setup_qdrant():
    """Connects to Qdrant and ensures collections and indices exist."""
    if not QDRANT_API_KEY:
        log.error("QDRANT_API_KEY environment variable not set.")
        return None
    
    client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    log.info(f"Connected to Qdrant at {QDRANT_HOST}")

    for collection_name in [CODE_COLLECTION_NAME, DOCS_COLLECTION_NAME, FILE_SUMMARY_COLLECTION_NAME]:
        if not client.collection_exists(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=VECTOR_METRIC)
            )
            log.info(f"Collection '{collection_name}' created.")
        
        # Create Payload Indices for filtering
        try:
            client.create_payload_index(collection_name=collection_name, field_name="repo_url", field_schema=models.PayloadSchemaType.KEYWORD)
            client.create_payload_index(collection_name=collection_name, field_name="file_path", field_schema=models.PayloadSchemaType.TEXT)
            client.create_payload_index(collection_name=collection_name, field_name="content_hash", field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass # Ignore if indices already exist

    return client

def setup_vertex_ai():
    """Initializes Vertex AI SDK."""
    if not VERTEX_PROJECT_ID or not VERTEX_LOCATION:
        log.error("VERTEX_PROJECT_ID and VERTEX_LOCATION environment variables must be set.")
        return None, None
    
    vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
    
    try:
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        generative_model = GenerativeModel(GENERATIVE_MODEL_NAME)
        log.info("Loaded Vertex AI models.")
        return embedding_model, generative_model
    except Exception as e:
        log.error(f"Error loading Vertex AI models: {e}")
        return None, None

def compute_hash(content: str) -> str:
    """Computes SHA256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def clone_repo(git_url: str, branch: str = None) -> str:
    """Smart Clone: Pulls if exists in cache, Clones if new."""
    repo_name = git_url.split("/")[-1].replace(".git", "")
    target_dir = os.path.join(WORKSPACE_ROOT, repo_name)
    
    os.makedirs(target_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(target_dir, ".git")):
        log.info(f"Repo {repo_name} found in cache. Pulling latest changes...")
        try:
            repo = Repo(target_dir)
            origin = repo.remotes.origin
            origin.pull()
            if branch:
                repo.git.checkout(branch)
            return target_dir
        except Exception as e:
            log.warning(f"Git pull failed: {e}. Re-cloning...")
            shutil.rmtree(target_dir)
            
    log.info(f"Cloning {git_url} to {target_dir}...")
    try:
        if branch:
            Repo.clone_from(git_url, target_dir, branch=branch)
        else:
            Repo.clone_from(git_url, target_dir)
        return target_dir
    except Exception as e:
        log.error(f"Failed to clone repo: {e}")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        return None

def chunk_text(text: str, source_url: str, doc_type: str, chunk_size=1000, chunk_overlap=150) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    split_texts = text_splitter.split_text(text)
    return [{
        "content": t,
        "metadata": {"doc_type": doc_type, "source_url": source_url, "chunk_index": i},
        "content_hash": compute_hash(t)
    } for i, t in enumerate(split_texts)]

def generate_deterministic_id(repo_url: str, file_path: str, chunk_index: int = 0) -> str:
    """
    Generates a consistent UUID based on the file location.
    Ensures that re-ingesting 'pom.xml' overwrites the old record instead of duplicating it.
    """
    unique_string = f"{repo_url}::{file_path}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
