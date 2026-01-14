import os
import shutil
import logging
import uuid
import vertexai
from qdrant_client import models

from pipeline_config import CODE_COLLECTION_NAME, DOCS_COLLECTION_NAME
from pipeline_utils import setup_qdrant, setup_vertex_ai, clone_repo, chunk_text
from code_processing import parse_codebase
from doc_processing import (
    parse_pdf, parse_confluence, generate_documentation, embed_chunks, store_in_qdrant
)

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
            # Note: parse_codebase signature might have changed in code_processing.py (returns 3 values)
            # code_processing.py: return code_chunks, doc_chunks, file_summaries
            code_chunks, text_chunks, _ = parse_codebase(repo_path, repo_url)
            
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
            # We don't delete if we use persistent cache (WORKSPACE_ROOT logic in utils)
            # But the detailed logic involves checking if it was a temp clone.
            # pipeline_utils.clone_repo uses WORKSPACE_ROOT and doesn't delete.
            # So we pass here.
            pass
            
    # --- 2. Process PDF Files ---
    for pdf_path in PDF_FILES:
        pdf_doc_chunks = parse_pdf(pdf_path)
        all_doc_chunks.extend(pdf_doc_chunks)
        
    # --- 3. Process Confluence Pages ---
    for page_url in CONFLUENCE_PAGES:
        confluence_doc_chunks = parse_confluence(page_url, token=None)
        all_doc_chunks.extend(confluence_doc_chunks)

    # --- 4. Embed and Store CODE ---
    if all_code_chunks:
        print(f"\nTotal {len(all_code_chunks)} code chunks to process.")
        # embed_chunks expects dicts now in doc_processing.py, but handles extraction internally??
        # doc_processing.py embed_chunks: "Embeds a list of chunk objects. Each chunk must be a dict with a 'content' key."
        # It handles splitting internally.
        
        # Returns: expanded_chunk_objs, embeddings
        code_chunks_expanded, code_embeddings = embed_chunks(embedding_model, all_code_chunks)
        
        print(f"\nStoring {len(code_embeddings)} code vectors in Qdrant...")
        store_in_qdrant(qdrant_client, CODE_COLLECTION_NAME, code_chunks_expanded, code_embeddings)
    else:
        print("\nNo code chunks to store.")
        
    # --- 5. Embed and Store DOCS ---
    if all_doc_chunks:
        print(f"\nTotal {len(all_doc_chunks)} doc chunks to process.")
        doc_chunks_expanded, doc_embeddings = embed_chunks(embedding_model, all_doc_chunks)
        
        print(f"\nStoring {len(doc_embeddings)} doc vectors in Qdrant...")
        store_in_qdrant(qdrant_client, DOCS_COLLECTION_NAME, doc_chunks_expanded, doc_embeddings)
    else:
        print("\nNo doc chunks to store.")

    print("\n--- Ingestion Pipeline Complete ---")
