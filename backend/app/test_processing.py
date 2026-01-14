import os
import logging
from vertexai.generative_models import GenerativeModel, Part
from qdrant_client import models

log = logging.getLogger(__name__)

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

def _generate_unit_tests(model, code_chunks, existing_tests, logical_name):
    # (Simplified logic for brevity - creates output files)
    log.info(f"[{logical_name}] Generating Unit Tests...")
    out_dir = f"generated_tests/{logical_name}/unit"
    os.makedirs(out_dir, exist_ok=True)
    # ... (Actual prompt logic would go here, iterating chunks) ...
