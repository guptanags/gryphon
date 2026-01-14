import os
import json
import logging
import subprocess
import re
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
from tree_sitter import Language, Parser, Query, QueryCursor
from tree_sitter_language_pack import get_language, get_parser
from vertexai.generative_models import GenerativeModel, Part

from pipeline_config import TEXT_EXTENSIONS
from pipeline_utils import compute_hash, chunk_text

log = logging.getLogger(__name__)

# --- Tree-sitter Language Configuration ---

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

# --- Parsing Logic ---

def _extract_imports(tree, language_name: str, content_bytes: bytes) -> list:
    """Extracts imported module/class names using Tree-sitter."""
    imports = []
    
    # Define queries for imports
    queries = {
        "python": """
            (import_statement name: (dotted_name) @imp)
            (import_from_statement module_name: (dotted_name) @imp)
        """,
        "java": """
            (import_declaration (scoped_identifier) @imp)
        """,
        "typescript": """
            (import_statement source: (string) @imp)
        """
    }
    
    if language_name not in queries:
        return []

    try:
        query = Query(get_language(language_name), queries[language_name])
        query_cursor = QueryCursor(query)
        
        for _, match_dict in query_cursor.matches(tree.root_node):
            node = next(iter(match_dict.values())) # Get the captured node
            if node:
                # Extract text, strip quotes/spaces
                imp_text = node.text.decode('utf8').strip("'\" ;")
                imports.append(imp_text)
    except Exception:
        pass
        
    return list(set(imports))

def _extract_metadata(node, language: str, content_bytes: bytes) -> dict:
    """Extracts docstrings and signatures from a Tree-sitter node."""
    metadata = {"docstring": "", "signature": ""}
    
    try:
        # 1. Signature (First line of the node usually contains the signature)
        # This is a heuristic; for perfect signatures, we'd need granular queries.
        start_byte = node.start_byte
        end_byte = node.end_byte
        
        # Limit signature length to avoid capturing body
        # Find the first brace or colon
        node_text = node.text.decode('utf8')
        first_line = node_text.split('\n')[0]
        metadata["signature"] = first_line[:200] # Cap at 200 chars
        
        # 2. Docstring
        # We look for a string literal immediately inside the block
        # This requires a language-specific query or traversal
        # Simplified approach: Look for the first string child node
        for child in node.children:
            if child.type == "block": # Python function body
                for grandchild in child.children:
                    if grandchild.type == "expression_statement":
                        for greatgrandchild in grandchild.children:
                            if greatgrandchild.type == "string":
                                metadata["docstring"] = greatgrandchild.text.decode('utf8').strip('\"\'')
                                break
        
        # Java/TS Docstrings (usually comments BEFORE the node)
        # Tree-sitter doesn't always attach comments to the node. 
        # We would need to look at the previous sibling.
        # Skipping for now to keep it simple, focusing on Python.
        
    except Exception:
        pass
        
    return metadata

def _calculate_complexity(content: str) -> int:
    """Calculates Cyclomatic Complexity for Python code."""
    try:
        # We can use radon programmatically if installed, or a simple heuristic
        # Heuristic: count branching keywords
        keywords = ["if ", "elif ", "for ", "while ", "except ", "with ", "&&", "||"]
        score = 1
        for kw in keywords:
            score += content.count(kw)
        return score
    except Exception:
        return 1

def parse_manifest_dependencies(file_path: str, content: str) -> list:
    """
    Parses build files to extract declared dependencies.
    Returns a list of strings: ["group:artifact:version", "npm-package:version"]
    """
    dependencies = []
    filename = os.path.basename(file_path)

    try:
        # 1. Maven (pom.xml)
        if filename == "pom.xml":
            root = ET.fromstring(content)
            # Handle XML namespaces if present (simplified here)
            ns = {'mvn': 'http://maven.apache.org/POM/4.0.0'}
            # Try with and without namespace
            deps = root.findall(".//dependency") or root.findall(".//mvn:dependency", ns)
            
            for dep in deps:
                g = dep.find("groupId") or dep.find("mvn:groupId", ns)
                a = dep.find("artifactId") or dep.find("mvn:artifactId", ns)
                v = dep.find("version") or dep.find("mvn:version", ns)
                
                if g is not None and a is not None:
                    dep_str = f"maven://{g.text}:{a.text}"
                    if v is not None: dep_str += f":{v.text}"
                    dependencies.append(dep_str)

        # 2. Node.js (package.json)
        elif filename == "package.json":
            data = json.loads(content)
            for section in ["dependencies", "devDependencies"]:
                if section in data:
                    for pkg, ver in data[section].items():
                        dependencies.append(f"npm://{pkg}:{ver}")

        # 3. Python (requirements.txt)
        elif filename == "requirements.txt":
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    dependencies.append(f"pypi://{line}")
        
        # 4. Gradle (build.gradle)
        elif filename.endswith(".gradle"):
            # Simple Regex for implementation 'group:name:version'
            # Matches: implementation 'com.google.guava:guava:30.1'
            pattern = r"implementation\s+['\"]([^'\"]+)['\"]"
            matches = re.findall(pattern, content)
            for m in matches:
                dependencies.append(f"gradle://{m}")

    except Exception as e:
        log.warning(f"Failed to parse dependencies in {file_path}: {e}")

    return dependencies

# New helper: Generate File Summary
def generate_file_summary(model: GenerativeModel, file_path: str, content: str) -> dict:
    """Generates a high-level summary of a file's responsibility."""
    # Truncate content to fit context (e.g., first 500 lines are usually enough for a summary)
    truncated_content = content[:15000] 
    
    prompt = f"""
    Analyze this code file and provide a 1-sentence technical summary of its responsibility.
    File Path: {file_path}
    
    Code:
    {truncated_content}
    
    Response Format: "Handles user authentication and JWT token generation."
    """
    try:
        res = model.generate_content([Part.from_text(prompt)])
        return {
            "file_path": file_path,
            "summary": res.text.strip(),
            "content_hash": compute_hash(res.text) # Hash of the summary itself
        }
    except Exception as e:
        log.warning(f"Summary generation failed for {file_path}: {e}")
        return None

def parse_codebase(repo_path: str, repo_url: str, branch: str = "main") -> Tuple[List, List, List]:
    code_chunks = []
    doc_chunks = []
    file_summaries = []
    manifest_files = {"pom.xml", "package.json", "requirements.txt", "build.gradle"}
    
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
                        full_content = f.read() # Need full content for summary
                        code_bytes = full_content.encode('utf8')
                    
                    # A. Generate Summary for the File
                    # (In a real batched pipeline, you might queue this, but here we do it inline)
                    # We only summarize if it's a "real" code file, not a tiny config
                    if len(full_content) > 100:
                        file_obj = {
                            "file_path": relative_path,
                            "content": full_content,
                            "repo_url": repo_url
                        }
                        file_summaries.append(file_obj)

                    tree = parser.parse(code_bytes)
                    dependencies = _extract_imports(tree, config["language"], code_bytes)
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
                                "dependencies": dependencies,
                                **_extract_metadata(node, config["language"], code_bytes),
                                "complexity": _calculate_complexity(content_str)
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

            # C. MANIFEST FILES
            if file in manifest_files or file.endswith(".gradle"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    deps = parse_manifest_dependencies(relative_path, content)
                    
                    if deps:
                        # Create a "Dependency Graph" document
                        # This allows the AI to find "Which repo uses Log4j?"
                        summary_text = f"Dependency Manifest for {relative_path}:\n" + "\n".join(deps)
                        
                        doc_chunks.append({
                            "content": summary_text,
                            "content_hash": compute_hash(summary_text),
                            "metadata": {
                                "doc_type": "dependency_graph",
                                "repo_url": repo_url,
                                "file_path": relative_path,
                                "dependency_count": len(deps),
                                "dependencies": deps # Store raw list for metadata filtering
                            }
                        })
                except Exception as e:
                    log.warning(f"Error processing manifest {file}: {e}")

    log.info(f"Parsed {len(code_chunks)} code chunks and {len(doc_chunks)} text/markup chunks.")
    return code_chunks, doc_chunks, file_summaries

# --- Analysis & Metrics (Radon, PMD) ---

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
