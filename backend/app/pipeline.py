import os
import tempfile
import shutil
import uuid
import subprocess
import json
from git import Repo
from qdrant_client import QdrantClient, models
from tree_sitter import Language, Parser, Query, QueryCursor
from tree_sitter_language_pack import get_language, get_parser
from tqdm import tqdm 
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbedding
from vertexai.generative_models import GenerativeModel
from google.cloud.aiplatform_v1.types.content import Part
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration (Copied from ingest.py) ---
# Qdrant
QDRANT_HOST = os.environ.get("QDRANT_HOST", "https://6916eb5b-7766-4a48-bdca-409766ee522d.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_PORT = 6333
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
CODE_COLLECTION_NAME = "code"
DOCS_COLLECTION_NAME = "docs"

# Vertex AI
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION")
EMBEDDING_MODEL_NAME = "text-embedding-004"
VECTOR_DIMENSION = 768 # Specific to text-embedding-004
VECTOR_METRIC = models.Distance.COSINE
GENERATIVE_MODEL_NAME = "gemini-2.5-flash-lite" 

# Processing
BATCH_SIZE = 50 # Batch size for both embedding and uploading

# 1. Load Languages
PY_LANGUAGE = get_language('python')
JAVA_LANGUAGE = get_language('java')
TS_LANGUAGE = get_language('typescript') # 'typescript' is the name for the 'ts' language

# 2. Load Parsers
PY_PARSER = get_parser('python')
JAVA_PARSER = get_parser('java')
TS_PARSER = get_parser('typescript')

# 3. Define Queries
PY_QUERY_STRING = """
(class_definition name: (identifier) @class.name) @class
(function_definition name: (identifier) @function.name) @function
"""
PY_QUERY = Query(PY_LANGUAGE, PY_QUERY_STRING)

# Java Query
JAVA_QUERY_STRING = """
(class_declaration name: (identifier) @class.name) @class
(method_declaration name: (identifier) @function.name) @function
(constructor_declaration name: (identifier) @function.name) @function
"""
JAVA_QUERY = Query(JAVA_LANGUAGE, JAVA_QUERY_STRING)

# TypeScript Query (for Angular)
TS_QUERY_STRING = """
(class_declaration name: (type_identifier) @class.name) @class
(method_definition name: (property_identifier) @function.name) @function
(function_declaration name: (identifier) @function.name) @function
(interface_declaration name: (type_identifier) @class.name) @class
(lexical_declaration (variable_declarator name: (identifier) value: (arrow_function))) @function
"""
TS_QUERY = Query(TS_LANGUAGE, TS_QUERY_STRING)

# 4. Main Config Map
# This map tells our parser which parser and query to use for each file type
LANGUAGE_CONFIG = {
    ".py": {"parser": PY_PARSER, "query": PY_QUERY, "language": "python"},
    ".java": {"parser": JAVA_PARSER, "query": JAVA_QUERY, "language": "java"},
    ".ts": {"parser": TS_PARSER, "query": TS_QUERY, "language": "typescript"},
}

# Files to parse as code
CODE_EXTENSIONS = tuple(LANGUAGE_CONFIG.keys())

# Files to parse as plain text (HTML, CSS, Markdown, etc.)
TEXT_EXTENSIONS = ('.html', '.css', '.md', '.txt', '.xml')


# --- pipeline.py ---

def pipeline_setup_qdrant():
    if not QDRANT_API_KEY:
        log.error("QDRANT_API_KEY environment variable not set.")
        return None
    
    client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    log.info(f"Connected to Qdrant at {QDRANT_HOST}")

    for collection_name in [CODE_COLLECTION_NAME, DOCS_COLLECTION_NAME]:
        # 1. Create Collection if it doesn't exist
        if not client.collection_exists(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=VECTOR_METRIC)
            )
            log.info(f"Collection '{collection_name}' created.")
        else:
            log.info(f"Collection '{collection_name}' already exists.")

        # 2. Create Payload Indices (Fix for your error)
        # We need these to filter by repo_url and file_path efficiently
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.repo_url",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.file_path",
                field_schema=models.PayloadSchemaType.TEXT
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.doc_type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            log.info(f"Payload indices verified for '{collection_name}'.")
        except Exception as e:
            # It's okay if they already exist, but log other errors
            log.warning(f"Note on indexing for {collection_name}: {e}")

    return client

def pipeline_setup_vertex_ai():
    if not VERTEX_PROJECT_ID or not VERTEX_LOCATION:
        log.error("VERTEX_PROJECT_ID and VERTEX_LOCATION environment variables must be set.")
        return None, None
    
    vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
    log.info(f"Vertex AI initialized for project '{VERTEX_PROJECT_ID}'")
    
    try:
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        generative_model = GenerativeModel(GENERATIVE_MODEL_NAME)
        log.info(f"Loaded Vertex AI models: {EMBEDDING_MODEL_NAME}, {GENERATIVE_MODEL_NAME}")
        return embedding_model, generative_model
    except Exception as e:
        log.error(f"Error loading Vertex AI models: {e}")
        return None, None

# --- Helper Functions (Copied from ingest.py) ---
# (chunk_text, clone_repo, parse_codebase, parse_pdf, 
#  parse_confluence, generate_documentation, embed_chunks, store_in_qdrant)

def chunk_text(text: str, source_url: str, doc_type: str, chunk_size=1000, chunk_overlap=150) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    split_texts = text_splitter.split_text(text)
    doc_chunks = []
    for i, text_chunk in enumerate(split_texts):
        doc_chunks.append({
            "content": text_chunk,
            "metadata": {"doc_type": doc_type, "source_url": source_url, "chunk_index": i}
        })
    return doc_chunks

def clone_repo(git_url: str, branch: str = None):
    """Clones a git repo into a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    log.info(f"Cloning {git_url} (branch: {branch or 'default'}) into {temp_dir}...")
    try:
        if branch:
            Repo.clone_from(git_url, temp_dir, branch=branch)
        else:
            Repo.clone_from(git_url, temp_dir)
        log.info("Clone successful.")
        return temp_dir
    except Exception as e:
        log.error(f"Error cloning repo {git_url}: {e}")
        shutil.rmtree(temp_dir)
        return None

def parse_codebase(repo_path: str, repo_url: str, branch: str = "main") -> (list, list):
    """Parses the entire codebase, routing files to correct parsers."""
    code_chunks = []
    doc_chunks = []
    
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_path)

            if file.endswith(CODE_EXTENSIONS):
                file_ext = os.path.splitext(file)[1]
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
                        code_chunks.append({
                            "content": node.text.decode('utf8'),
                            "metadata": {
                                "repo_url": repo_url, "file_path": relative_path,
                                "chunk_name": name_node.text.decode('utf8'),
                                "chunk_type": "class" if "class" in match_dict else "function",
                                "start_line": node.start_point[0] + 1, "end_line": node.end_point[0] + 1,
                            }
                        })
                except Exception as e:
                    log.warning(f"Error parsing code file {file_path}: {e}")

            elif file.endswith(TEXT_EXTENSIONS):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()
                    text_chunks = chunk_text(
                        full_text,
                        source_url=f"{repo_url}/blob/{branch}/{relative_path}",
                        doc_type=os.path.splitext(file)[1]
                    )
                    doc_chunks.extend(text_chunks)
                except Exception as e:
                    log.warning(f"Error parsing text file {file_path}: {e}")

    log.info(f"Parsed {len(code_chunks)} code chunks and {len(doc_chunks)} text/markup chunks.")
    return code_chunks, doc_chunks

def parse_confluence(page_url: str, confluence_api_token: str = None) -> list:
    log.info(f"Parsing Confluence page: {page_url}...")
    try:
        headers = {"User-Agent": "My-AI-Ingestion-Bot/1.0"}
        if confluence_api_token:
            headers["Authorization"] = f"Bearer {confluence_api_token}"
        response = requests.get(page_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find("div", id="main-content")
        full_text = main_content.get_text(separator="\n", strip=True) if main_content else soup.get_text(separator="\n", strip=True)
        return chunk_text(full_text, source_url=page_url, doc_type="confluence")
    except Exception as e:
        log.error(f"Error parsing Confluence page {page_url}: {e}")
        return []

def generate_documentation(model: GenerativeModel, code_chunks: list) -> list:
    """Generates documentation for a list of code chunks."""
    generation_config = GenerationConfig(temperature=0.1, max_output_tokens=1024)
    prompt_template_start = "..." # (Keep your full prompt template)
    prompt_template_end = "\n```\n**Documentation:**\n"
    doc_chunks = []
    
    for chunk in tqdm(code_chunks, desc="Generating documentation"):
        code_content = chunk['content'][:4000] # Truncate large chunks
        prompt = f"{prompt_template_start}{code_content}{prompt_template_end}"
        try:
            response = model.generate_content([Part.from_text(prompt)])
            doc_chunks.append({
                "content": response.text,
                "metadata": {
                    "doc_type": "ai_generated_tech_summary",
                    "source_repo_url": chunk['metadata']['repo_url'],
                    "source_file_path": chunk['metadata']['file_path'],
                    "source_chunk_name": chunk['metadata']['chunk_name'],
                }
            })
        except Exception as e:
            log.warning(f"Error generating doc for {chunk['metadata']['chunk_name']}: {e}")
            continue
    return doc_chunks

def embed_chunks(model: TextEmbeddingModel, chunks_content: list[str]) -> list[list[float]]:
    """Embeds a list of text chunks in batches."""
    all_embeddings = []
    for i in tqdm(range(0, len(chunks_content), BATCH_SIZE), desc="Embedding chunks"):
        batch = chunks_content[i:i + BATCH_SIZE]
        try:
            embeddings_response: list[TextEmbedding] = model.get_embeddings(batch)
            all_embeddings.extend([e.values for e in embeddings_response])
        except Exception as e:
            log.error(f"Error embedding batch {i}-{i+BATCH_SIZE}: {e}")
            all_embeddings.extend([None] * len(batch))
    return all_embeddings

def store_in_qdrant(client: QdrantClient, collection_name: str, chunks: list, embeddings: list):
    """Stores chunks and their embeddings in Qdrant in batches."""
    points = []
    for i, chunk in enumerate(chunks):
        if embeddings[i] is None: continue
        payload = chunk['metadata']
        payload['content'] = chunk['content']
        points.append(models.PointStruct(id=str(uuid.uuid4()), vector=embeddings[i], payload=payload))
    
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc=f"Uploading to {collection_name}"):
        batch_points = points[i:i + BATCH_SIZE]
        try:
            client.upsert(collection_name=collection_name, points=batch_points, wait=False)
        except Exception as e:
            log.error(f"Error uploading batch {i}-{i+BATCH_SIZE} to Qdrant: {e}")
    log.info(f"Successfully stored {len(points)} vectors in '{collection_name}'.")


# --- The New Orchestrator Function ---

def run_ingestion_pipeline(logical_name: str, git_repo_list: list, confluence_page_list: list):
    """
    The main pipeline function to be run in a background task.
    Initializes its own clients.
    """
    log.info(f"[{logical_name}] Pipeline started. Initializing clients...")
    
    # Each background task initializes its own clients
    qdrant_client = pipeline_setup_qdrant()
    embedding_model, generative_model = pipeline_setup_vertex_ai()
    
    if not qdrant_client or not embedding_model or not generative_model:
        log.error(f"[{logical_name}] FATAL: Failed to initialize clients. Aborting task.")
        raise RuntimeError("Failed to initialize Vertex/Qdrant clients")
    
    all_code_chunks = []
    all_doc_chunks = []
    avg_static_analysis_score = None
    avg_quality_score = None

    # --- 1. Process Git Repos ---
    for repo_info in git_repo_list:
        try:
            repo_url, branch = repo_info.strip().split(',')
        except ValueError:
            log.warning(f"[{logical_name}] Skipping invalid git_repo format: {repo_info}. Must be 'url,branch'")
            continue
        
        log.info(f"[{logical_name}] Processing {repo_url} @ {branch}...")
        repo_path = clone_repo(repo_url, branch)
        if not repo_path:
            continue
        
        try:
            code_chunks, text_chunks = parse_codebase(repo_path, repo_url, branch)
            all_doc_chunks.extend(text_chunks)
            
            if code_chunks:
                all_code_chunks.extend(code_chunks)
                ai_doc_chunks = generate_documentation(generative_model, code_chunks)
                all_doc_chunks.extend(ai_doc_chunks)
            
            avg_static_analysis_score = _get_static_analysis_score(repo_path, logical_name)
        finally:
            log.info(f"[{logical_name}] Cleaning up {repo_path}...")
            shutil.rmtree(repo_path)
    
    # --- 2. Process Confluence Pages ---
    for page_url in confluence_page_list:
        confluence_doc_chunks = parse_confluence(page_url, confluence_api_token=None)
        all_doc_chunks.extend(confluence_doc_chunks)
    # --- 3. Get Code Quality Score (After processing all repos) ---
    if all_code_chunks:
        avg_quality_score = _get_code_quality_score(generative_model, all_code_chunks, logical_name)

    # --- 4. Embed and Store CODE ---
    if all_code_chunks:
        log.info(f"[{logical_name}] Embedding {len(all_code_chunks)} code chunks...")
        code_content_to_embed = [chunk['content'] for chunk in all_code_chunks]
        code_embeddings = embed_chunks(embedding_model, code_content_to_embed)
        store_in_qdrant(qdrant_client, CODE_COLLECTION_NAME, all_code_chunks, code_embeddings)
    
    # --- 5. Embed and Store DOCS ---
    if all_doc_chunks:
        log.info(f"[{logical_name}] Embedding {len(all_doc_chunks)} doc chunks...")
        doc_content_to_embed = [chunk['content'] for chunk in all_doc_chunks]
        doc_embeddings = embed_chunks(embedding_model, doc_content_to_embed)
        store_in_qdrant(qdrant_client, DOCS_COLLECTION_NAME, all_doc_chunks, doc_embeddings)

    log.info(f"[{logical_name}] --- Pipeline Finished Successfully ---")
    # Return the score to the background task wrapper
    return avg_quality_score, avg_static_analysis_score


def _get_all_chunks_for_repo(client: QdrantClient, repo_url: str, collection: str, path_filter: str = None) -> list:
    """Scrolls through Qdrant to get all chunks for a specific repo."""
    log.info(f"Fetching all chunks for {repo_url} from {collection}...")
    all_payloads = []
    
    filter_must = [
        models.FieldCondition(
            key="metadata.repo_url",
            match=models.MatchValue(value=repo_url)
        )
    ]
    
    if path_filter:
        filter_must.append(
            models.FieldCondition(
                key="metadata.file_path",
                match=models.MatchText(text=path_filter) # Find file paths containing this string
            )
        )

    results, next_page = client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(must=filter_must),
        limit=250,
        with_payload=True
    )
    all_payloads.extend([r.payload for r in results])
    
    while next_page:
        results, next_page = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=filter_must),
            limit=250,
            with_payload=True,
            offset=next_page
        )
        all_payloads.extend([r.payload for r in results])
        
    log.info(f"Found {len(all_payloads)} chunks.")
    return all_payloads


def _generate_unit_tests(model: GenerativeModel, code_chunks: list, existing_tests: list, logical_name: str):
    """Generates missing unit tests."""
    log.info(f"[{logical_name}] Starting unit test generation...")
    
    # Combine all existing tests into one large text block for context
    existing_tests_context = "\n".join([t['content'] for t in existing_tests])
    if not existing_tests_context:
        existing_tests_context = "No existing tests were found."

    # Create directory for output
    output_dir = os.path.join("generated_tests", logical_name, "unit")
    os.makedirs(output_dir, exist_ok=True)

    prompt_template = """
You are an expert SDET (Software Developer in Test). Your task is to generate a new, high-quality unit test for the given function.
Critically, you must **only** generate a test if it is **not** already covered by the existing tests provided.

**EXISTING TESTS (for context):**
{existing_tests_context}
**FUNCTION TO TEST:**
File: `{file_path}`
{code_chunk}
**Task:**
Review the FUNCTION TO TEST and the EXISTING TESTS.
1.  If the function is already well-tested, respond with only the text: "SKIP: Already covered."
2.  If the function is not covered, write a new, high-quality unit test for it using the appropriate framework (e.g., pytest for Python, JUnit for Java).
3.  Do not repeat existing tests. Focus on missing edge cases or core logic.

**NEW UNIT TEST:**
"""

    for chunk in tqdm(code_chunks, desc=f"[{logical_name}] Generating Unit Tests"):
        prompt = prompt_template.format(
            existing_tests_context=existing_tests_context[:10000], # Limit context size
            file_path=chunk['metadata']['file_path'],
            code_chunk=chunk['content']
        )
        
        try:
            response = model.generate_content(
                [Part.from_text(prompt)]
            )
            
            generated_test = response.text
            
            if "SKIP:" not in generated_test:
                # Save the new test
                test_file_name = f"test_{chunk['metadata']['file_path'].replace('/', '_')}.py"
                with open(os.path.join(output_dir, test_file_name), "a") as f:
                    f.write(f"\n# Test for {chunk['metadata']['file_path']} - {chunk['metadata']['chunk_name']}\n")
                    f.write(generated_test)
                    f.write("\n")
                    
        except Exception as e:
            log.warning(f"Failed to generate test for {chunk['metadata']['chunk_name']}: {e}")

def _get_static_analysis_score(repo_path: str, logical_name: str) -> float | None:
    """
    Runs a static analysis tool (Radon for Python) on the repo
    and returns an average Maintainability Index.
    """
    log.info(f"[{logical_name}] Running static analysis (Radon)...")
    
    # We will analyze Python files for now.
    # In the future, a "dispatcher" would check language and run PMD, ESLint, etc.
    
    try:
        # Run radon: 'mi' for Maintainability Index, '-s' for silent, '-j' for JSON output
        # We run it on the entire repo path '.'
        process = subprocess.run(
            ["radon", "mi", "-s", "-j", "."],
            cwd=repo_path, # Run the command *inside* the cloned repo
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the JSON output
        results = json.loads(process.stdout)
        
        scores = []
        for file, data in results.items():
            if "mi" in data:
                scores.append(data["mi"])
                
        if not scores:
            log.info(f"[{logical_name}] No files found for static analysis.")
            return None
            
        # Average the Maintainability Index (0-100, higher is better)
        avg_score = sum(scores) / len(scores)
        log.info(f"[{logical_name}] Average Maintainability Index: {avg_score:.2f}")
        return avg_score
        
    except FileNotFoundError:
        log.error(f"[{logical_name}] 'radon' command not found. Make sure it's installed.")
        return None
    except subprocess.CalledProcessError as e:
        log.error(f"[{logical_name}] Static analysis failed: {e.stderr}")
        return None
    except Exception as e:
        log.error(f"[{logical_name}] Error parsing analysis results: {e}")
        return None
    

def _generate_acceptance_tests(model: GenerativeModel, doc_chunks: list, logical_name: str):
    """Generates high-level acceptance tests from documentation."""
    log.info(f"[{logical_name}] Starting acceptance test generation...")
    
    doc_context = "\n".join([d['content'] for d in doc_chunks if d['metadata']['doc_type'] == 'confluence'])
    if not doc_context:
        doc_context = "\n".join([d['content'] for d in doc_chunks]) # Fallback to all docs
        
    if not doc_context:
        log.warning(f"[{logical_name}] No documentation found to generate acceptance tests.")
        return

    output_dir = os.path.join("generated_tests", logical_name, "acceptance")
    os.makedirs(output_dir, exist_ok=True)
    
    prompt = f"""
You are an expert QA Automation Engineer. Based on the following documentation, identify 3-5 key user-facing features and write high-level acceptance tests for them in Gherkin (Given/When/Then) format.

**DOCUMENTATION CONTEXT:**
{doc_context[:20000]}

**ACCEPTANCE TESTS (Gherkin Format):**
"""
    try:
        response = model.generate_content(
            [Part.from_text(prompt)]
        )
        # Save the Gherkin feature file
        with open(os.path.join(output_dir, "features.feature"), "w") as f:
            f.write(response.text)
    except Exception as e:
        log.error(f"Failed to generate acceptance tests: {e}")


def _generate_load_tests(model: GenerativeModel, code_chunks: list, logical_name: str):
    """Generates a k6 load test script for identified API endpoints."""
    log.info(f"[{logical_name}] Starting load test generation...")

    # A simple heuristic to find API routes (works for Flask, Spring)
    api_routes = []
    for chunk in code_chunks:
        if "@app.route" in chunk['content'] or "@GetMapping" in chunk['content'] or "@PostMapping" in chunk['content']:
            api_routes.append(chunk['content'])
            
    if not api_routes:
        log.warning(f"[{logical_name}] No API routes found to generate load tests.")
        return
        
    output_dir = os.path.join("generated_tests", logical_name, "load")
    os.makedirs(output_dir, exist_ok=True)
    
    prompt = f"""
You are a Performance Engineer. Based on the following API route definitions, write a basic JMeter load test script to simulate a simple load (e.g., 10 virtual users for 30 seconds) against these endpoints.
Assume the base URL is 'https://api.example.com'.

**API ROUTES:**
{api_routes[:20000]}
**JMeter LOAD TEST SCRIPT:**
"""
    try:
        response = model.generate_content(
            [Part.from_text(prompt)]
        )
        # Save the k6 script
        with open(os.path.join(output_dir, "load-test.js"), "w") as f:
            f.write(response.text)
    except Exception as e:
        log.error(f"Failed to generate load test: {e}")


# --- New Main Orchestrator for Test Generation ---

def run_test_generation_pipeline(logical_name: str, test_types: list, repo_urls: list):
    """
    The main pipeline function for generating tests.
    """
    log.info(f"[{logical_name}] Test Generation Pipeline started. Types: {test_types}")
    
    qdrant_client = pipeline_setup_qdrant()
    embedding_model, generative_model = pipeline_setup_vertex_ai()
    
    if not qdrant_client or not generative_model:
        raise RuntimeError("Failed to initialize Vertex/Qdrant clients")
    
    # We only need the *first* repo URL for this logic, assuming one logical_name = one primary repo
    # A more complex setup might merge code from all repos, but let's start simple.
    if not repo_urls:
        log.error(f"[{logical_name}] No repo URLs provided.")
        return
        
    primary_repo_url = repo_urls[0].split(',')[0] # Get URL from 'url,branch'
    
    # 1. Get all code and existing tests from Qdrant
    all_code = _get_all_chunks_for_repo(qdrant_client, primary_repo_url, CODE_COLLECTION_NAME)
    all_tests = _get_all_chunks_for_repo(qdrant_client, primary_repo_url, CODE_COLLECTION_NAME, path_filter="test")
    all_docs = _get_all_chunks_for_repo(qdrant_client, primary_repo_url, DOCS_COLLECTION_NAME)
    
    # 2. Run selected generators
    if "unit" in test_types:
        _generate_unit_tests(generative_model, all_code, all_tests, logical_name)
        
    if "acceptance" in test_types or "automation" in test_types:
        _generate_acceptance_tests(generative_model, all_docs, logical_name)
        
    if "load" in test_types:
        _generate_load_tests(generative_model, all_code, logical_name)

    # 3. Estimate Code Coverage
    # We do this *after* generating tests, but for now we'll just use existing tests
    # A future step would be to add the *newly* generated tests to all_tests
    coverage_score = _estimate_code_coverage(generative_model, all_code, all_tests, logical_name)

    log.info(f"[{logical_name}] --- Test Generation Pipeline Finished ---")
    
    # 4. Return the score
    return coverage_score


def _get_code_quality_score(model: GenerativeModel, code_chunks: list, logical_name: str) -> float | None:
    """Uses Gemini to analyze a sample of code and return a quality score."""
    log.info(f"[{logical_name}] Starting code quality analysis...")
    if not code_chunks:
        return None
    
    # We'll sample up to 20 chunks to get a representative score
    sample_size = min(len(code_chunks), 20)
    # Get a random sample of chunks
    import random
    code_samples = random.sample(code_chunks, sample_size)
    
    scores = []
    
    prompt_template = """
You are a Staff Software Engineer. Analyze the maintainability, readability, and complexity of the following code snippet.
Respond with ONLY a numeric score from 1 (terrible) to 100 (perfect).

Code:
{code}
Score (1-100):
"""
    
    for chunk in tqdm(code_samples, desc=f"[{logical_name}] Analyzing Quality"):
        try:
            prompt = prompt_template.format(code=chunk['content'][:4000]) # Truncate large chunks
            response = model.generate_content(
                [Part.from_text(prompt)]
            )
            # Clean up the response to get only the number
            score_text = response.text.strip().replace("'", "").replace('"',"")
            score = float(score_text)
            scores.append(score)
        except Exception as e:
            log.warning(f"Failed to get quality score for chunk {chunk['metadata']['chunk_name']}: {e}")
            continue
            
    if not scores:
        log.info(f"[{logical_name}] No quality scores were generated.")
        return None
        
    avg_score = sum(scores) / len(scores)
    log.info(f"[{logical_name}] Average code quality score: {avg_score:.2f}")
    return avg_score


# --- Add this new function to pipeline.py ---

def _estimate_code_coverage(model: GenerativeModel, all_code: list, all_tests: list, logical_name: str) -> float | None:
    """Uses Gemini to estimate test coverage."""
    log.info(f"[{logical_name}] Estimating code coverage...")
    if not all_code:
        return None
        
    # Combine code and tests into large context blocks
    code_context = "\n".join([c['content'] for c in all_code])
    test_context = "\n".join([t['content'] for t in all_tests])
    
    if not test_context:
        log.info(f"[{logical_name}] No tests found, estimating coverage as 0.")
        return 0.0

    # Truncate context to fit model limits
    code_context = code_context[:20000]
    test_context = test_context[:20000]
    
    prompt = f"""
You are a QA automation expert. Based on the provided source code and the test suite, please estimate the unit test code coverage percentage.
- Analyze the functions in the source code.
- Analyze the tests provided in the test suite.
- Provide your best estimate of what percentage of the source code is executed by the test suite.
- Respond with ONLY a numeric value from 0 to 100.

**SOURCE CODE (Sample):**
{code_context}
**TEST SUITE (Sample):**
{test_context}
**Estimated Coverage (0-100):**
"""
    try:
        response = model.generate_content(
            [Part.from_text(prompt)]
        )
        # Clean up the response to get only the number
        score_text = response.text.strip().replace("'", "").replace('"',"")
        score = float(score_text)
        log.info(f"[{logical_name}] Estimated code coverage: {score:.2f}%")
        return score
    except Exception as e:
        log.warning(f"Failed to estimate code coverage: {e}")
        return None
    

