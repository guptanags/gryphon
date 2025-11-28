import os
import shutil
import uuid
import hashlib
import json
import subprocess
import logging
from typing import List, Tuple, Optional
from git import Repo

# Third-party libraries
from qdrant_client import QdrantClient, models
from tree_sitter import Language, Parser, Query, QueryCursor
from tree_sitter_language_pack import get_language, get_parser
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vertex AI
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbedding
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

# --- 1. Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# --- 2. Configuration & Constants ---

# Paths
# Define a persistent cache directory for Repos (Mount this as PVC in OpenShift)
WORKSPACE_ROOT = os.environ.get("WORKSPACE_ROOT", "/tmp/ai_workspace_cache")

# Qdrant
QDRANT_HOST = os.environ.get("QDRANT_HOST", "https://6916eb5b-7766-4a48-bdca-409766ee522d.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_PORT = 6333
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
CODE_COLLECTION_NAME = "code"
DOCS_COLLECTION_NAME = "docs"

# Vertex AI
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID","vrittera")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATIVE_MODEL_NAME = "gemini-1.5-flash-001"
VECTOR_DIMENSION = 768
VECTOR_METRIC = models.Distance.COSINE

# Processing
BATCH_SIZE = 50 # Reduced to avoid Token Limit errors

# --- 3. Tree-sitter Language Configuration ---

# Load Languages
PY_LANGUAGE = get_language('python')
JAVA_LANGUAGE = get_language('java')
TS_LANGUAGE = get_language('typescript')

# Load Parsers
PY_PARSER = get_parser('python')
JAVA_PARSER = get_parser('java')
TS_PARSER = get_parser('typescript')

# Define Queries
PY_QUERY_STRING = """
(class_definition name: (identifier) @class.name) @class
(function_definition name: (identifier) @function.name) @function
"""
PY_QUERY = Query(PY_LANGUAGE, PY_QUERY_STRING)

JAVA_QUERY_STRING = """
(class_declaration name: (identifier) @class.name) @class
(method_declaration name: (identifier) @function.name) @function
(constructor_declaration name: (identifier) @function.name) @function
"""
JAVA_QUERY = Query(JAVA_LANGUAGE, JAVA_QUERY_STRING)

TS_QUERY_STRING = """
(class_declaration name: (type_identifier) @class.name) @class
(method_definition name: (property_identifier) @function.name) @function
(function_declaration name: (identifier) @function.name) @function
(interface_declaration name: (type_identifier) @class.name) @class
"""
TS_QUERY = Query(TS_LANGUAGE, TS_QUERY_STRING)

# Config Map
LANGUAGE_CONFIG = {
    ".py": {"parser": PY_PARSER, "query": PY_QUERY, "language": "python"},
    ".java": {"parser": JAVA_PARSER, "query": JAVA_QUERY, "language": "java"},
    ".ts": {"parser": TS_PARSER, "query": TS_QUERY, "language": "typescript"},
}

CODE_EXTENSIONS = tuple(LANGUAGE_CONFIG.keys())
TEXT_EXTENSIONS = ('.html', '.css', '.md', '.txt', '.xml', '.json')

# --- 4. Setup Functions ---

def setup_qdrant():
    """Connects to Qdrant and ensures collections and indices exist."""
    if not QDRANT_API_KEY:
        log.error("QDRANT_API_KEY environment variable not set.")
        return None
    
    client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    log.info(f"Connected to Qdrant at {QDRANT_HOST}")

    for collection_name in [CODE_COLLECTION_NAME, DOCS_COLLECTION_NAME]:
        if not client.collection_exists(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=VECTOR_METRIC)
            )
            log.info(f"Collection '{collection_name}' created.")
        
        # Create Payload Indices for filtering
        try:
            client.create_payload_index(collection_name=collection_name, field_name="metadata.repo_url", field_schema=models.PayloadSchemaType.KEYWORD)
            client.create_payload_index(collection_name=collection_name, field_name="metadata.file_path", field_schema=models.PayloadSchemaType.TEXT)
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

# --- 5. Helper Functions (Caching, Hashing, Parsing) ---

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

# --- 6. Parsing Logic (Multi-Language) ---

def parse_codebase(repo_path: str, repo_url: str, branch: str = "main") -> Tuple[List, List]:
    code_chunks = []
    doc_chunks = []
    
    for root, dirs, files in os.walk(repo_path):
        if ".git" in root: continue # Skip .git directory
        
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_path)
            file_ext = os.path.splitext(file)[1]

            # A. CODE FILES
            if file_ext in CODE_EXTENSIONS:
                config = LANGUAGE_CONFIG[file_ext]
                parser = config["parser"]
                query = config["query"]
                query_cursor = QueryCursor(query)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_bytes = f.read().encode('utf8')
                    
                    tree = parser.parse(code_bytes)
                    for pattern_index, match_dict in query_cursor.matches(tree.root_node):
                        node_list = next(iter(match_dict.values()), None)
                        name_node_list = match_dict.get(next(iter(match_dict.keys())) + ".name")

                        if not node_list or not name_node_list: continue 
                        node = node_list[0]
                        name_node = name_node_list[0]
                        
                        content_str = node.text.decode('utf8')
                        code_chunks.append({
                            "content": content_str,
                            "content_hash": compute_hash(content_str),
                            "metadata": {
                                "repo_url": repo_url, "file_path": relative_path,
                                "chunk_name": name_node.text.decode('utf8'),
                                "chunk_type": "class" if "class" in match_dict else "function",
                                "start_line": node.start_point[0] + 1, "end_line": node.end_point[0] + 1,
                            }
                        })
                except Exception as e:
                    log.warning(f"Error parsing code {file_path}: {e}")

            # B. TEXT/MARKUP FILES
            elif file_ext in TEXT_EXTENSIONS:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()
                    doc_chunks.extend(chunk_text(
                        full_text, f"{repo_url}/blob/{branch}/{relative_path}", file_ext
                    ))
                except Exception as e:
                    log.warning(f"Error parsing text {file_path}: {e}")

    log.info(f"Parsed {len(code_chunks)} code chunks and {len(doc_chunks)} text/markup chunks.")
    return code_chunks, doc_chunks

def parse_pdf(pdf_path: str) -> list:
    log.info(f"Parsing PDF: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        full_text = "".join([page.extract_text() + "\n" for page in reader.pages])
        return chunk_text(full_text, pdf_path, "pdf")
    except Exception as e:
        log.error(f"Error parsing PDF: {e}")
        return []

def parse_confluence(page_url: str, token: str = None) -> list:
    log.info(f"Parsing Confluence: {page_url}")
    try:
        headers = {"User-Agent": "AI-Bot"}
        if token: headers["Authorization"] = f"Bearer {token}"
        response = requests.get(page_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        main = soup.find("div", id="main-content")
        text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)
        return chunk_text(text, page_url, "confluence")
    except Exception as e:
        log.error(f"Error parsing Confluence: {e}")
        return []

# --- 7. Analysis & Metrics (Radon, PMD) ---

def _get_static_analysis_score(repo_path: str, logical_name: str) -> float | None:
    """Runs Radon (Python) and returns avg Maintainability Index."""
    log.info(f"[{logical_name}] Running Radon (Python)...")
    try:
        process = subprocess.run(["radon", "mi", "-s", "-j", "."], cwd=repo_path, capture_output=True, text=True)
        results = json.loads(process.stdout)
        scores = [data["mi"] for data in results.values() if "mi" in data]
        if not scores: return None
        return sum(scores) / len(scores)
    except Exception as e:
        log.warning(f"Radon analysis failed: {e}")
        return None

def _get_java_static_analysis_score(repo_path: str, logical_name: str) -> float | None:
    """Runs PMD (Java) and returns calculated quality score."""
    log.info(f"[{logical_name}] Running PMD (Java)...")
    try:
        # Assumes 'pmd' is in PATH. Uses default ruleset.
        cmd = ["pmd", "check", "-d", ".", "-R", "rulesets/java/quickstart.xml", "-f", "json"]
        process = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
        try:
            results = json.loads(process.stdout)
        except json.JSONDecodeError:
            return None # PMD often outputs logs mixed with JSON if not configured perfectly
            
        files = results.get("files", [])
        if not files: return None
        
        total_violations = 0
        for file in files:
            for v in file.get("violations", []):
                total_violations += (6 - v.get("priority", 3)) # Weighted sum
        
        # 100 - avg_violations_per_file * 2
        penalty = (total_violations / len(files)) * 2
        return max(0, 100 - penalty)
    except Exception as e:
        log.warning(f"PMD analysis failed: {e}")
        return None

def _estimate_code_coverage(model: GenerativeModel, all_code: list, all_tests: list, logical_name: str) -> float:
    """Uses LLM to estimate coverage if no CI report exists."""
    log.info(f"[{logical_name}] Estimating coverage via LLM...")
    if not all_code: return 0.0
    
    code_ctx = "\n".join([c['content'] for c in all_code[:20]]) # Sample
    test_ctx = "\n".join([t['content'] for t in all_tests[:20]]) # Sample
    
    prompt = f"""
    Estimate unit test code coverage percentage (0-100) based on this sample.
    Respond with ONLY a number.
    
    Code Sample:
    {code_ctx[:5000]}
    
    Test Sample:
    {test_ctx[:5000]}
    """
    try:
        res = model.generate_content([Part.from_text(prompt)])
        return float(res.text.strip().replace("%",""))
    except:
        return 0.0

# --- 8. AI Generators (Docs, Embeddings, Tests) ---

def generate_documentation(model: GenerativeModel, code_chunks: list) -> list:
    doc_chunks = []
    gen_config = GenerationConfig(temperature=0.1, max_output_tokens=1024)
    
    for chunk in tqdm(code_chunks, desc="Generating Docs"):
        # Incremental Check could go here, but usually done at embedding stage
        prompt = f"""
        Write a technical and functional summary for this code:
        ```
        {chunk['content'][:4000]}
        ```
        Format:
        **Technical:** ...
        **Functional:** ...
        """
        try:
            res = model.generate_content([Part.from_text(prompt)], generation_config=gen_config)
            doc_chunks.append({
                "content": res.text,
                "content_hash": compute_hash(res.text),
                "metadata": {
                    "doc_type": "ai_generated_tech_summary",
                    "source_chunk_name": chunk['metadata']['chunk_name'],
                    "repo_url": chunk['metadata']['repo_url']
                }
            })
        except Exception:
            continue
    return doc_chunks

def embed_chunks(model: TextEmbeddingModel, chunks_content: list) -> list:
    all_embeddings = []
    for i in tqdm(range(0, len(chunks_content), BATCH_SIZE), desc="Embedding"):
        batch = chunks_content[i:i + BATCH_SIZE]
        try:
            resp = model.get_embeddings(batch)
            all_embeddings.extend([e.values for e in resp])
        except Exception as e:
            log.error(f"Embedding error: {e}")
            all_embeddings.extend([None] * len(batch))
    return all_embeddings

def store_in_qdrant(client: QdrantClient, collection_name: str, chunks: list, embeddings: list):
    points = []
    for i, chunk in enumerate(chunks):
        if embeddings[i] is None: continue
        
        # Incremental Ingestion Check
        # (In production, you'd batch this check, but here's the logic per item)
        # For bulk speed, we skip the pre-check here and assume caller handled logic
        # or we just overwrite. Overwriting is safe with UUIDs, but efficientupsert is better.
        
        payload = chunk['metadata']
        payload['content'] = chunk['content']
        payload['content_hash'] = chunk.get('content_hash', '')
        
        points.append(models.PointStruct(id=str(uuid.uuid4()), vector=embeddings[i], payload=payload))
        
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc=f"Uploading to {collection_name}"):
        try:
            client.upsert(collection_name=collection_name, points=points[i:i+BATCH_SIZE], wait=False)
        except Exception as e:
            log.error(f"Upsert error: {e}")

# --- 9. Test Generation Helpers ---

def _get_all_chunks(client, repo_url, collection, filter_text=None):
    must = [models.FieldCondition(key="metadata.repo_url", match=models.MatchValue(value=repo_url))]
    if filter_text:
        must.append(models.FieldCondition(key="metadata.file_path", match=models.MatchText(text=filter_text)))
    
    results, next_page = client.scroll(collection_name=collection, scroll_filter=models.Filter(must=must), limit=1000, with_payload=True)
    payloads = [r.payload for r in results]
    while next_page:
        results, next_page = client.scroll(collection_name=collection, scroll_filter=models.Filter(must=must), limit=1000, with_payload=True, offset=next_page)
        payloads.extend([r.payload for r in results])
    return payloads

def _generate_unit_tests(model, code_chunks, existing_tests, logical_name):
    # (Simplified logic for brevity - creates output files)
    log.info(f"[{logical_name}] Generating Unit Tests...")
    out_dir = f"generated_tests/{logical_name}/unit"
    os.makedirs(out_dir, exist_ok=True)
    # ... (Actual prompt logic would go here, iterating chunks) ...

# --- 10. Main Orchestrators ---

def run_ingestion_pipeline(logical_name: str, git_repo_list: list, confluence_page_list: list):
    log.info(f"[{logical_name}] Pipeline Started.")
    qdrant = setup_qdrant()
    emb_model, gen_model = setup_vertex_ai()
    
    if not qdrant or not emb_model: raise RuntimeError("Client Init Failed")
    
    all_code = []
    all_docs = []
    quality_score = None
    
    # 1. Git Repos
    for repo_info in git_repo_list:
        try:
            url, branch = repo_info.strip().split(',')
            path = clone_repo(url, branch)
            if not path: continue
            
            try:
                # Parse
                code, docs = parse_codebase(path, url, branch)
                all_code.extend(code)
                all_docs.extend(docs)
                
                # Static Analysis
                if not quality_score:
                    if any(f['metadata']['file_path'].endswith('.java') for f in code):
                        quality_score = _get_java_static_analysis_score(path, logical_name)
                    else:
                        quality_score = _get_static_analysis_score(path, logical_name)

                # Generate AI Docs
                if code:
                    ai_docs = generate_documentation(gen_model, code)
                    all_docs.extend(ai_docs)
            finally:
                # Since we use caching workspace, do NOT delete path if you want to keep cache
                # BUT if using temp clone logic inside clone_repo, you should delete.
                # With 'WORKSPACE_ROOT', we keep it.
                pass 
        except Exception as e:
            log.error(f"Repo error: {e}")

    # 2. Confluence
    for page in confluence_page_list:
        all_docs.extend(parse_confluence(page))
        
    # 3. Embed & Store
    if all_code:
        vecs = embed_chunks(emb_model, [c['content'] for c in all_code])
        store_in_qdrant(qdrant, CODE_COLLECTION_NAME, all_code, vecs)
        
    if all_docs:
        vecs = embed_chunks(emb_model, [c['content'] for c in all_docs])
        store_in_qdrant(qdrant, DOCS_COLLECTION_NAME, all_docs, vecs)
        
    log.info(f"[{logical_name}] Pipeline Complete.")
    return quality_score

def run_test_generation_pipeline(logical_name: str, test_types: list, repo_urls: list):
    log.info(f"[{logical_name}] Test Gen Started: {test_types}")
    qdrant = setup_qdrant()
    emb_model, gen_model = setup_vertex_ai()
    
    primary_url = repo_urls[0].split(',')[0]
    
    all_code = _get_all_chunks(qdrant, primary_url, CODE_COLLECTION_NAME)
    all_tests = _get_all_chunks(qdrant, primary_url, CODE_COLLECTION_NAME, filter_text="test")
    
    if "unit" in test_types:
        _generate_unit_tests(gen_model, all_code, all_tests, logical_name)
        
    # Coverage Est
    return _estimate_code_coverage(gen_model, all_code, all_tests, logical_name)