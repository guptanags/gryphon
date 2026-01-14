import os
import uuid
import logging
from tqdm import tqdm
from qdrant_client import models

from pipeline_config import (
    CODE_COLLECTION_NAME, DOCS_COLLECTION_NAME, FILE_SUMMARY_COLLECTION_NAME
)
from pipeline_utils import (
    setup_qdrant, setup_vertex_ai, clone_repo, compute_hash, chunk_text
)
from code_processing import (
    parse_codebase, generate_file_summary, _get_static_analysis_score, 
    _get_java_static_analysis_score
)
from doc_processing import (
    parse_confluence, parse_pdf, generate_documentation, embed_chunks, store_in_qdrant
)
from test_processing import (
    _get_all_chunks, _generate_unit_tests, _estimate_code_coverage
)

log = logging.getLogger(__name__)

# --- Main Orchestrators ---

def run_ingestion_pipeline(logical_name: str, git_repo_list: list, confluence_page_list: list):
    log.info(f"[{logical_name}] Pipeline Started.")
    qdrant = setup_qdrant()
    emb_model, gen_model = setup_vertex_ai()
    
    if not qdrant or not emb_model: raise RuntimeError("Client Init Failed")
    
    all_code = []
    all_docs = []
    all_summaries = []
    quality_score = None
    
    # 1. Git Repos
    for repo_info in git_repo_list:
        try:
            url_part = repo_info.strip()
            if ',' in url_part:
                url, branch = url_part.split(',')
            else:
                url, branch = url_part, None
                
            path = clone_repo(url, branch)
            if not path: continue
            
            try:
                # Parse
                code, docs, files = parse_codebase(path, url, branch if branch else "main")
                
                # Generate AI Summaries for Files
                log.info(f"[{logical_name}] Generating File Level Summaries...")
                for f in tqdm(files, desc="Summarizing Files"):
                    summ = generate_file_summary(gen_model, f['file_path'], f['content'])
                    if summ:
                        summ['repo_url'] = f['repo_url']
                        all_summaries.append(summ)
                
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
                pass 
        except Exception as e:
            log.error(f"Repo error: {e}")

    # 2. Confluence
    for page in confluence_page_list:
        all_docs.extend(parse_confluence(page))
        
    # 3. Embed & Store
    if all_code:
        all_code, vecs = embed_chunks(emb_model, all_code)
        store_in_qdrant(qdrant, CODE_COLLECTION_NAME, all_code, vecs)
        
    if all_docs:
        all_docs, vecs = embed_chunks(emb_model, all_docs)
        store_in_qdrant(qdrant, DOCS_COLLECTION_NAME, all_docs, vecs)

    if all_summaries:
        all_summaries, vecs = embed_chunks(emb_model, all_summaries)
        # Convert to Qdrant Points
        points = []
        for i, summ in enumerate(all_summaries):
            if i >= len(vecs) or vecs[i] is None: continue
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vecs[i],
                payload=summ
            ))
        qdrant.upsert(collection_name=FILE_SUMMARY_COLLECTION_NAME, points=points)

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