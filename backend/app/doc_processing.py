import logging
import requests
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part
from vertexai.language_models import TextEmbeddingModel

from pipeline_config import BATCH_SIZE, MAX_EMBED_CHARS
from pipeline_utils import compute_hash, chunk_text, generate_deterministic_id

log = logging.getLogger(__name__)

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

def generate_documentation(model: GenerativeModel, code_chunks: list) -> list:
    """
    Generates detailed Functional, Technical, and Architectural documentation 
    for code chunks using GenAI.
    """
    doc_chunks = []
    # Increase output tokens to allow for comprehensive documentation
    gen_config = GenerationConfig(temperature=0.2, max_output_tokens=2048)
    
    log.info(f"Generating detailed documentation for {len(code_chunks)} chunks...")

    for chunk in tqdm(code_chunks, desc="AI Documentation"):
        # Skip trivial chunks (e.g., empty constructors or tiny helpers) to save cost
        if len(chunk['content']) < 100:
            continue

        file_path = chunk['metadata'].get('file_path', 'unknown')
        chunk_name = chunk['metadata'].get('chunk_name', 'unknown')
        chunk_type = chunk['metadata'].get('chunk_type', 'component')
        
        prompt = f"""
        Act as a Senior Software Architect and Technical Writer.
        Analyze the provided code component and generate comprehensive documentation.
        
        **Component Context:**
        - File: {file_path}
        - Name: {chunk_name}
        - Type: {chunk_type}
        
        **Code:**
        ```
        {chunk['content'][:8000]} 
        ```
        
        **Documentation Requirements:**
        1. **Functional Specification:** Describe WHAT this component does from a business/domain perspective. Who uses it and why?
        2. **Technical Details:** Describe HOW it works. List key parameters, return values, side effects, and complex algorithms.
        3. **Architecture & Patterns:** Identify design patterns used (e.g., Singleton, Builder, MVC Controller). List critical dependencies.
        
        **Output Format (Markdown):**
        # Documentation: {chunk_name}
        
        ## 📘 Functional Overview
        (Text here...)
        
        ## ⚙️ Technical Specification
        - **Inputs:** ...
        - **Outputs:** ...
        - **Logic:** ...
        
        ## 🏗️ Architecture & Design
        - **Patterns:** ...
        - **Dependencies:** ...
        """
        
        try:
            res = model.generate_content([Part.from_text(prompt)], generation_config=gen_config)
            
            doc_chunks.append({
                "content": res.text,
                "content_hash": compute_hash(res.text),
                "metadata": {
                    "doc_type": "ai_generated_documentation",
                    "repo_url": chunk['metadata']['repo_url'],
                    "file_path": file_path,
                    "source_chunk_name": chunk_name,
                    # Link back to the exact code hash this doc was generated for
                    "related_code_hash": chunk['content_hash'] 
                }
            })
        except Exception as e:
            log.warning(f"Doc generation failed for {chunk_name}: {e}")
            continue
            
    return doc_chunks

def _split_long_text(content: str, max_chars: int = MAX_EMBED_CHARS) -> list:
    """Split a long text into smaller segments while trying to split at newline boundaries.
    Returns a list of strings each not exceeding max_chars.
    """
    if not content or len(content) <= max_chars:
        return [content]

    chunks = []
    start = 0
    length = len(content)
    while start < length:
        end = min(start + max_chars, length)
        # try to split at last newline to keep logical boundaries
        newline_idx = content.rfind('\n', start, end)
        if newline_idx > start:
            end = newline_idx
        chunks.append(content[start:end])
        start = end
    return chunks

def embed_chunks(model: TextEmbeddingModel, chunk_objs: list) -> tuple[list, list]:
    """Embeds a list of chunk objects. Each chunk must be a dict with a 'content' key.
    Returns a tuple (expanded_chunk_objs, embeddings) where expanded_chunk_objs contains
    any additional chunks created by splitting long content.
    """
    expanded_chunks = []
    contents = []

    # Expand any long contents into multiple chunks
    for chunk in chunk_objs:
        content = chunk.get('content', '')
        if not isinstance(content, str):
            content = str(content)

        if len(content) > MAX_EMBED_CHARS:
            log.info(f"Content too long ({len(content)} chars); splitting into smaller chunks for embedding")
            parts = _split_long_text(content, MAX_EMBED_CHARS)
            for idx, p in enumerate(parts):
                new_chunk = dict(chunk)
                new_chunk['content'] = p
                # annotate metadata to indicate split index for debugging
                metadata = dict(new_chunk.get('metadata', {}))
                metadata['split_index'] = idx
                new_chunk['metadata'] = metadata
                expanded_chunks.append(new_chunk)
                contents.append(p)
        else:
            expanded_chunks.append(chunk)
            contents.append(content)

    all_embeddings: list = []
    # Batch and embed; if a batch fails due to token limit, retry per-item with smaller splits
    for i in tqdm(range(0, len(contents), BATCH_SIZE), desc="Embedding"):
        batch = contents[i:i + BATCH_SIZE]
        try:
            resp = model.get_embeddings(batch)
            all_embeddings.extend([e.values for e in resp])
        except Exception as e:
            # Log full error and attempt to embed items individually and further split if needed
            log.error(f"Embedding error on batch: {e}")
            for j, item in enumerate(batch):
                try:
                    resp_item = model.get_embeddings([item])
                    all_embeddings.append(resp_item[0].values)
                except Exception as e2:
                    log.error(f"Embedding error for single item (len={len(item)}): {e2}")
                    # Attempt to split item into smaller segments
                    parts = _split_long_text(item, int(MAX_EMBED_CHARS / 2))
                    part_vecs = []
                    for part in parts:
                        try:
                            resp_part = model.get_embeddings([part])
                            part_vecs.append(resp_part[0].values)
                        except Exception as e3:
                            log.error(f"Failed to embed split part (len={len(part)}): {e3}")
                            part_vecs.append(None)
                    # If any part vectors are valid, we average them as a simple heuristic
                    valid_vecs = [v for v in part_vecs if v is not None]
                    if valid_vecs:
                        avg_vec = np.mean(np.stack(valid_vecs), axis=0).tolist()
                        all_embeddings.append(avg_vec)
                    else:
                        all_embeddings.append(None)

    return expanded_chunks, all_embeddings

def store_in_qdrant(client: QdrantClient, collection_name: str, chunks: list, embeddings: list):
    points = []
    for i, chunk in enumerate(chunks):
        if i >= len(embeddings) or embeddings[i] is None: continue
        
        # Incremental Ingestion Check
        payload = chunk['metadata']
        payload['content'] = chunk['content']
        payload['content_hash'] = chunk.get('content_hash', '')

        point_id = generate_deterministic_id(
            chunk['metadata']['repo_url'],
            chunk['metadata']['file_path'],
            chunk['metadata'].get('chunk_index', 0)
        )
        
        points.append(models.PointStruct(id=point_id, vector=embeddings[i], payload=payload))
        
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc=f"Uploading to {collection_name}"):
        try:
            client.upsert(collection_name=collection_name, points=points[i:i+BATCH_SIZE], wait=False)
        except Exception as e:
            log.error(f"Upsert error: {e}")
