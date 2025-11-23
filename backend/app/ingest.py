import os
import tempfile
import shutil
import uuid
from git import Repo
from qdrant_client import QdrantClient, models
from tree_sitter import Language, Parser, Query, QueryCursor # Add QueryCursor
from tree_sitter_language_pack import get_language, get_parser
from tqdm import tqdm # For progress bars

# --- Vertex AI Imports ---
# --- Vertex AI Imports ---
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbedding
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter



# --- Configuration ---
# Qdrant
QDRANT_HOST = "https://6916eb5b-7766-4a48-bdca-409766ee522d.europe-west3-0.gcp.cloud.qdrant.io"
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

# --- Setup Functions ---

def setup_qdrant():
    """Connects to Qdrant and ensures collections exist."""
    if not QDRANT_API_KEY:
        print("ERROR: QDRANT_API_KEY environment variable not set.")
        return None
    
    client = QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    print(f"Connected to Qdrant at {QDRANT_HOST}")

    for collection_name in [CODE_COLLECTION_NAME, DOCS_COLLECTION_NAME]:
        if not client.collection_exists(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=VECTOR_METRIC)
            )
            print(f"Collection '{collection_name}' created.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    return client

# Setup vertex ai
def setup_vertex_ai():
    """Initializes Vertex AI SDK and loads models."""
    if not VERTEX_PROJECT_ID or not VERTEX_LOCATION:
        print("ERROR: VERTEX_PROJECT_ID and VERTEX_LOCATION environment variables must be set.")
        return None, None
    
    vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
    print(f"Vertex AI initialized for project '{VERTEX_PROJECT_ID}' in '{VERTEX_LOCATION}'")
    
    try:
        # Load Embedding Model
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        print(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
        
        # Load Generative Model (the old, working way)
        generative_model = GenerativeModel(GENERATIVE_MODEL_NAME)
        print(f"Loaded generative model: {GENERATIVE_MODEL_NAME}")
        
        return embedding_model, generative_model
    except Exception as e:
        print(f"Error loading Vertex AI models: {e}")
        return None, None

# --- Module 1: Fetch ---

def clone_repo(git_url):
    """Clones a git repo into a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    print(f"Cloning {git_url} into {temp_dir}...")
    try:
        Repo.clone_from(git_url, temp_dir)
        print("Clone successful.")
        return temp_dir
    except Exception as e:
        print(f"Error cloning repo: {e}")
        shutil.rmtree(temp_dir)
        return None

# --- Module 1: Parse ---

# --- Module 1: Parse ---

def parse_codebase(repo_path: str, repo_url: str) -> (list, list):
    """
    Parses the entire codebase, routing files to the correct parser
    (tree-sitter for code, text-splitter for text/markup).
    
    Returns:
        (list_of_code_chunks, list_of_doc_chunks)
    """
    code_chunks = []
    doc_chunks = []
    
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_path)

            if file.endswith(CODE_EXTENSIONS):
                # --- This is a CODE file ---
                file_ext = os.path.splitext(file)[1]
                config = LANGUAGE_CONFIG[file_ext]
                parser = config["parser"]
                query = config["query"]
                
                query_cursor = QueryCursor(query) # Use the correct cursor logic
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_bytes = f.read().encode('utf8')
                    
                    tree = parser.parse(code_bytes)
                    
                    for pattern_index, match_dict in query_cursor.matches(tree.root_node):
                        node_list = next(iter(match_dict.values()), None)
                        name_node_list = match_dict.get(next(iter(match_dict.keys())) + ".name")

                        if not node_list or not name_node_list:
                            continue

                        node = node_list[0]
                        name_node = name_node_list[0]

                        chunk_name = name_node.text.decode('utf8')
                        chunk_type = "class" if "class" in match_dict else "function"

                        code_chunks.append({
                            "content": node.text.decode('utf8'),
                            "metadata": {
                                "repo_url": repo_url,
                                "file_path": relative_path,
                                "chunk_name": chunk_name,
                                "chunk_type": chunk_type,
                                "start_line": node.start_point[0] + 1,
                                "end_line": node.end_point[0] + 1,
                            }
                        })
                except Exception as e:
                    print(f"Error parsing code file {file_path}: {e}")

            elif file.endswith(TEXT_EXTENSIONS):
                # --- This is a TEXT/MARKUP file (HTML, CSS, etc.) ---
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()
                    
                    # Use our existing text chunker
                    text_chunks = chunk_text(
                        full_text,
                        source_url=f"{repo_url}/blob/main/{relative_path}",
                        doc_type=os.path.splitext(file)[1] # .html, .css
                    )
                    doc_chunks.extend(text_chunks)
                except Exception as e:
                    print(f"Error parsing text file {file_path}: {e}")

    print(f"Parsed {len(code_chunks)} code chunks.")
    print(f"Parsed {len(doc_chunks)} text/markup chunks.")
    return code_chunks, doc_chunks

# --- (Rest of your functions: parse_pdf, parse_confluence, generate_documentation, etc.) ---

# --- Module 1: PDF Parsing ---

def parse_pdf(pdf_path: str) -> list:
    """Extracts text from a PDF and splits it into chunks."""
    print(f"Parsing PDF: {pdf_path}...")
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        # Use the new chunk_text function
        return chunk_text(full_text, source_url=pdf_path, doc_type="pdf")
    except Exception as e:
        print(f"Error parsing PDF {pdf_path}: {e}")
        return []

# --- Module 1: Confluence/Web Parsing ---

def parse_confluence(page_url: str, confluence_api_token: str = None) -> list:
    """Fetches a Confluence page, extracts text, and splits it into chunks."""
    print(f"Parsing Confluence page: {page_url}...")
    try:
        # Set up headers. If you have an API token, you can use it.
        headers = {
            "User-Agent": "My-AI-Ingestion-Bot/1.0",
        }
        if confluence_api_token:
            headers["Authorization"] = f"Bearer {confluence_api_token}"

        response = requests.get(page_url, headers=headers)
        response.raise_for_status() # Raise error for bad responses
        
        # Use BeautifulSoup to parse the HTML and get only the text
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find the main content (this selector works well for Confluence)
        main_content = soup.find("div", id="main-content")
        if main_content:
            full_text = main_content.get_text(separator="\n", strip=True)
        else:
            # Fallback for general web pages
            full_text = soup.get_text(separator="\n", strip=True)
        
        # Use the new chunk_text function
        return chunk_text(full_text, source_url=page_url, doc_type="confluence")
    except Exception as e:
        print(f"Error parsing Confluence page {page_url}: {e}")
        return []


# --- Module 1: Generate Docs ---

def generate_documentation(model: GenerativeModel, code_chunks: list) -> list:
    """Generates documentation for a list of code chunks."""
    
    generation_config = GenerationConfig(
        temperature=0.1,
        max_output_tokens=1024
    )
    
    prompt_template_start = """
You are an expert Staff Software Engineer. Your job is to write clear, concise documentation.
Generate a technical and functional summary for the following code snippet.

Provide the output in this format:

**Technical Summary:**
A brief, one-sentence technical description of what the code does (e.g., "A function that takes a user ID and returns the user object from the database.")

**Functional Summary:**
A description of the code's business logic or purpose (e.g., "This function is responsible for retrieving customer details for the user profile page.")

**Code Snippet:**
```python
"""
    
    prompt_template_end = "\n```\n**Documentation:**\n"

    doc_chunks = []
    
    for chunk in tqdm(code_chunks, desc="Generating documentation"):
        code_content = chunk['content']
        
        if len(code_content) > 4000:
            code_content = code_content[:4000] + "\n... (truncated)"
            
        prompt = f"{prompt_template_start}{code_content}{prompt_template_end}"
        
        try:
            # The original, working call
            response = model.generate_content(
                [Part.from_text(prompt)],
                generation_config=generation_config
            )
            
            generated_text = response.text
            
            doc_chunks.append({
                "content": generated_text,
                "metadata": {
                    "doc_type": "ai_generated_tech_summary",
                    "source_repo_url": chunk['metadata']['repo_url'],
                    "source_file_path": chunk['metadata']['file_path'],
                    "source_chunk_name": chunk['metadata']['chunk_name'],
                    "source_start_line": chunk['metadata']['start_line']
                }
            })
            
        except Exception as e:
            print(f"Error generating doc for {chunk['metadata']['chunk_name']}: {e}")
            continue
            
    print(f"Generated {len(doc_chunks)} documentation snippets.")
    return doc_chunks

# --- Module 1: Text Chunking ---

def chunk_text(text: str, source_url: str, doc_type: str, chunk_size=1000, chunk_overlap=150) -> list:
    """Splits a large text document into smaller chunks."""
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # Split the document
    split_texts = text_splitter.split_text(text)
    
    # Create the doc_chunk objects
    doc_chunks = []
    for i, text_chunk in enumerate(split_texts):
        doc_chunks.append({
            "content": text_chunk,
            "metadata": {
                "doc_type": doc_type,
                "source_url": source_url,
                "chunk_index": i
            }
        })
        
    return doc_chunks

# --- Module 1: Embed ---

def embed_chunks(model: TextEmbeddingModel, chunks_content: list[str]) -> list[list[float]]:
    """Embeds a list of text chunks in batches."""
    all_embeddings = []
    # Use tqdm for a progress bar
    for i in tqdm(range(0, len(chunks_content), BATCH_SIZE), desc="Embedding chunks"):
        batch = chunks_content[i:i + BATCH_SIZE]
        try:
            # Get embeddings from Vertex AI
            embeddings_response: list[TextEmbedding] = model.get_embeddings(batch)
            # Extract the vector from each response
            all_embeddings.extend([e.values for e in embeddings_response])
        except Exception as e:
            print(f"Error embedding batch {i}-{i+BATCH_SIZE}: {e}")
            # Add Nones for failed batches to keep lengths consistent
            all_embeddings.extend([None] * len(batch))
            
    return all_embeddings

# --- Module 1: Store ---

def store_in_qdrant(client: QdrantClient, collection_name: str, chunks: list, embeddings: list):
    """Stores chunks and their embeddings in Qdrant in batches."""
    points = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings[i]
        # Skip if embedding failed
        if embedding is None:
            print(f"Skipping chunk {i} due to embedding error.")
            continue
        
        # Add the raw content to the payload
        payload = chunk['metadata']
        payload['content'] = chunk['content']
        
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()), # Generate a unique ID
                vector=embedding,
                payload=payload
            )
        )
    
    # Upsert in batches
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Uploading to Qdrant"):
        batch_points = points[i:i + BATCH_SIZE]
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch_points,
                wait=False # Set to True if you want to wait for confirmation
            )
        except Exception as e:
            print(f"Error uploading batch {i}-{i+BATCH_SIZE} to Qdrant: {e}")

    print(f"Successfully stored {len(points)} vectors in '{collection_name}'.")


if __name__ == "__main__":
    qdrant_client = setup_qdrant()
    embedding_model, generative_model = setup_vertex_ai()
    
    if not qdrant_client or not embedding_model or not generative_model:
        print("Exiting due to setup error.")
        exit()

    # --- Define All Our Sources ---
    GIT_REPOS = [
        #"https://github.com/pallets/flask.git",
        # Add a Java or Angular repo here for testing, e.g.:
        "https://github.com/spring-projects/spring-petclinic.git" # Java
        # "https://github.com/angular/angular-realworld-example-app.git" # Angular (TS)
    ]
    PDF_FILES = [
        # Add local paths to any PDFs
    ]
    CONFLUENCE_PAGES = [
        # Add any Confluence or web URLs
    ]
    
    all_code_chunks = []
    all_doc_chunks = [] # This will hold AI docs, text docs, PDFs, etc.

    # --- 1. Process Git Repos ---
    for repo_url in GIT_REPOS:
        print(f"\n--- Processing Repo: {repo_url} ---")
        repo_path = clone_repo(repo_url)
        if not repo_path:
            continue
        
        try:
            # Parse code and text files
            code_chunks, text_chunks = parse_codebase(repo_path, repo_url)
            
            # Add text chunks to our main doc list
            all_doc_chunks.extend(text_chunks)
            
            if not code_chunks:
                print(f"No code chunks found in {repo_url}.")
                continue
            
            all_code_chunks.extend(code_chunks)
            
            # Generate AI docs from code
            print(f"\nGenerating documentation for {len(code_chunks)} code chunks...")
            ai_doc_chunks = generate_documentation(generative_model, code_chunks)
            all_doc_chunks.extend(ai_doc_chunks)
            
        finally:
            print(f"\nCleaning up {repo_path}...")
            shutil.rmtree(repo_path)
            
    # --- 2. Process PDF Files ---
    for pdf_path in PDF_FILES:
        pdf_doc_chunks = parse_pdf(pdf_path)
        all_doc_chunks.extend(pdf_doc_chunks)
        
    # --- 3. Process Confluence Pages ---
    for page_url in CONFLUENCE_PAGES:
        confluence_doc_chunks = parse_confluence(page_url, confluence_api_token=None)
        all_doc_chunks.extend(confluence_doc_chunks)

    # --- 4. Embed and Store CODE ---
    if all_code_chunks:
        print(f"\nTotal {len(all_code_chunks)} code chunks to process.")
        code_content_to_embed = [chunk['content'] for chunk in all_code_chunks]
        code_embeddings = embed_chunks(embedding_model, code_content_to_embed)
        
        print(f"\nStoring {len(code_embeddings)} code vectors in Qdrant...")
        store_in_qdrant(qdrant_client, CODE_COLLECTION_NAME, all_code_chunks, code_embeddings)
    else:
        print("\nNo code chunks to store.")
        
    # --- 5. Embed and Store DOCS ---
    if all_doc_chunks:
        print(f"\nTotal {len(all_doc_chunks)} doc chunks to process.")
        doc_content_to_embed = [chunk['content'] for chunk in all_doc_chunks]
        doc_embeddings = embed_chunks(embedding_model, doc_content_to_embed)
        
        print(f"\nStoring {len(doc_embeddings)} doc vectors in Qdrant...")
        store_in_qdrant(qdrant_client, DOCS_COLLECTION_NAME, all_doc_chunks, doc_embeddings)
    else:
        print("\nNo doc chunks to store.")

    print("\n--- Ingestion Pipeline Complete ---")
