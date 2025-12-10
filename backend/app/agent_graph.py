import os
import json
import requests
import subprocess
import ast
import operator
from typing import Annotated, TypedDict, List, Dict
from langgraph.graph import StateGraph, END

# Internal Modules
from agent_tools import AgentTools
from pipeline import setup_qdrant, CODE_COLLECTION_NAME
from qdrant_client import models
from prompt_manager import prompt_manager # <--- NEW IMPORT
from flashrank import Ranker, RerankRequest

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")

# --- Configuration ---
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
MODEL_ID = "gemini-1.5-flash-001" 

# --- 1. JSON Schemas ---
CODER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thought_process": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "search_block": {
                        "type": "string", 
                        "description": "The EXACT existing code block to be replaced. Copy-paste relevant lines."
                    },
                    "replace_block": {
                        "type": "string", 
                        "description": "The new code to insert in place of the search_block."
                    },
                    "action": {
                        "type": "string", 
                        "enum": ["replace", "create_file"]
                    }
                },
                "required": ["filepath", "action"]
            }
        }
    },
    "required": ["thought_process", "edits"]
}

# --- 2. Helper Functions (REST & Auth) ---

def get_access_token():
    token = os.environ.get("GCLOUD_ACCESS_TOKEN")
    if token: return token
    try:
        return subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode("utf-8").strip()
    except:
        return None

def generate_content_rest(prompt: str, schema: dict = None, mime_type: str = "text/plain"):
    token = get_access_token()
    if not token: raise ValueError("No GCloud Token")

    url = f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/{VERTEX_PROJECT_ID}/locations/{VERTEX_LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"

    gen_config = {"temperature": 0.1, "maxOutputTokens": 8192}
    if schema:
        gen_config["responseMimeType"] = "application/json"
        gen_config["responseSchema"] = schema
    elif mime_type == "application/json":
        gen_config["responseMimeType"] = "application/json"

    payload = {
        "contents": [{ "role": "user", "parts": [{"text": prompt}] }],
        "generationConfig": gen_config
    }
    
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        res_json = resp.json()
        text = res_json['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text) if (schema or mime_type == "application/json") else text
    except Exception as e:
        print(f"Vertex Error: {e}")
        raise

# --- 3. State Definition ---

class AgentState(TypedDict):
    logical_name: str
    repo_path: str
    requirement: str
    language: str
    project_type: str
    plan: str
    relevant_files: List[str]
    code_changes: Dict[str, str]
    test_code: Dict[str, str]
    raw_requirement: str      # The user's original input
    requirement: str          # The REFINED requirement (used by Planner/Coder)
    dependency_context: str

    # Robustness Flags
    syntax_status: str
    review_status: str
    verifier_status: str
    missing_symbols: List[str]
    iterations: int
    
    history: Annotated[List[str], operator.add]

# --- 4. The Agent Team ---

class AutonomousDevTeam:
    def __init__(self, logical_name: str, repo_path: str):
        self.logical_name = logical_name
        self.repo_path = repo_path
        self.tools = AgentTools(repo_path)
        self.qdrant = setup_qdrant()

    # --- REQUIREMENT ANALYST ---
    def analyst_agent(self, state: AgentState):
        print("--- ANALYST AGENT ---")
        
        # Detect Stack (Duplicate logic from Planner, or move to a shared setup node)
        files_list = self.tools.list_files()
        language = "python"
        project_type = "generic"
        if any(f.endswith("pom.xml") for f in files_list):
            language = "java"; project_type = "spring-boot"
        elif any(f.endswith("package.json") for f in files_list):
            language = "typescript"; project_type = "node"

        # USE PROMPT MANAGER
        prompt = prompt_manager.render(
            'analyst', 'user',
            requirement=state['raw_requirement'], # Use raw input
            language=language,
            project_type=project_type
        )
        
        try:
            # We treat this as a refined string, not JSON, to allow flexible formatting
            refined_text = generate_content_rest(prompt)
            print(f"   [Analyst] Refined Requirement:\n{refined_text[:100]}...")
            
            return {
                "requirement": refined_text, # Overwrite the 'requirement' field for downstream agents
                "language": language,        # Pass these along so Planner doesn't need to re-detect
                "project_type": project_type,
                "history": ["Requirement refined by Analyst."]
            }
        except Exception as e:
            print(f"Analyst failed: {e}")
            # Fallback: Just use the raw input if Analyst crashes
            return {"requirement": state['raw_requirement'], "history": ["Analyst failed, using raw input."]}

    # --- PLANNER ---
    def planner_agent(self, state: AgentState):
        print("--- PLANNER AGENT ---")
        files_list = self.tools.list_files()
        
        # Detect Stack
        language = "python"
        project_type = "generic"
        if any(f.endswith("pom.xml") for f in files_list):
            language = "java"; project_type = "spring-boot"
        elif any(f.endswith("package.json") for f in files_list):
            language = "typescript"; project_type = "node"
        
        # Generate Skeleton
        skeleton = self.tools.generate_repo_skeleton()
        if len(skeleton) > 50000: skeleton = skeleton[:50000] + "\n...(truncated)"
        
        # USE PROMPT MANAGER
        prompt = prompt_manager.render(
            'planner', 'user',
            requirement=state['requirement'],
            language=language,
            project_type=project_type,
            skeleton=skeleton
        )
        
        plan_text = generate_content_rest(prompt)
        
        return {
            "plan": plan_text, 
            "language": language, 
            "project_type": project_type,
            "iterations": 0,
            "history": [f"Plan generated ({language})."]
        }

    # --- RESEARCHER ---
    def researcher_agent(self, state: AgentState):
        print("--- RESEARCHER AGENT ---")
        found_files = []
        dep_context = "No dependency information found."

        # Handle "Missing Symbols" loop from Verifier
        if state.get("missing_symbols"):
            print(f"   [Researcher] Hunting missing symbols: {state['missing_symbols']}")
            # Logic to find definition... (simplified for brevity)
            # found_files.append(...) 
        
        # USE PROMPT MANAGER
        prompt = prompt_manager.render('researcher', 'extract_files', plan=state['plan'])
        
        try:
            extraction = generate_content_rest(prompt, mime_type="application/json")
            for f_path in extraction.get("files", []):
                if self.tools.read_file(f_path): found_files.append(f_path)
        except: pass

        if not found_files:
            print("   [Researcher] Performing Hierarchical Search...")
            
            req_vector = self.embed_model.get_embeddings([state['requirement']])[0].values
            
            # STEP A: File Level Search (Find top 5 relevant files)
            file_results = self.qdrant.search(
                collection_name=FILE_SUMMARY_COLLECTION_NAME,
                query_vector=req_vector,
                limit=5,
                with_payload=True
            )
            
            # 2. Format for Reranker
            passages = [
                {"id": str(i), "text": res.payload['content'], "meta": res.payload} 
                for i, res in enumerate(file_results)
            ]
    
    # 3. Rerank! (The Magic Step)
            rerank_request = RerankRequest(query=state['requirement'], passages=passages)
            reranked_results = ranker.rerank(rerank_request)
            top_files = []
            for r in reranked_results[:5]:
                print(f"   [Reranker] Score {r['score']}: {r['meta']['file_path']}")
                top_files.append(r['meta']['file_path'])
            print(f"   [Researcher] Top Candidate Files: {top_files}")
            
            # Add them to our "Found" list so Coder gets the whole file if needed
            found_files.extend(top_files)
            
            # STEP B: Chunk Level Scoped Search (Optional optimization)
            # If you want to find specific *lines* inside those files to save context window:
            chunk_results = self.qdrant.search(
                collection_name=CODE_COLLECTION_NAME,
                query_vector=req_vector,
                limit=10,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path", 
                            match=models.MatchAny(any=top_files) # <--- SCOPED FILTER
                        )
                    ]
                )
            )

        try:
            # Search for the special doc_type we created in pipeline.py
            dep_results = self.qdrant.scroll(
                collection_name="docs", # Or wherever you stored doc_chunks
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.doc_type", 
                            match=models.MatchValue(value="dependency_graph")
                        ),
                        models.FieldCondition(
                            key="metadata.repo_url", 
                            # We assume we are working on the repo defined in the state
                            # You might need to fetch this from the first found file's metadata if multi-repo
                            match=models.MatchValue(value=state.get('repo_url', '')) 
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )
            
            if dep_results[0]:
                # This contains the "Maven://..." or "npm://..." list we generated
                dep_context = dep_results[0][0].payload['content']
                print(f"   [Researcher] Loaded Dependency Context ({len(dep_context)} chars).")
                
        except Exception as e:
            print(f"   [Researcher] Warning: Could not load dependencies: {e}")
        reuse_keywords = ["Util", "Helper", "Mapper", "Config", "Common"]
        reuse_context = []
        
        # NEW: Explicit "Reuse Hunt"
        # If the plan mentions "formatting dates" or "user validation", 
        # we specifically look for existing Utils/Helpers.
        for keyword in reuse_keywords:
            # Quick filename search (much faster than vector search)
            # You can add this method to agent_tools or use simple glob matching
            matches = [f for f in self.tools.list_files() if keyword in f]
            reuse_context.extend(matches)

        return {
            "relevant_files": list(set(found_files)), 
            "dependency_context": f"Existing Utilities you might use: {reuse_context}",
            "history": [f"Researched files and dependencies."]
        }
    # --- CONTEXT VERIFIER ---
    def context_verifier_agent(self, state: AgentState):
        print("--- CONTEXT VERIFIER ---")
        
        # USE PROMPT MANAGER
        prompt = prompt_manager.render(
            'verifier', 'user',
            requirement=state['requirement'],
            plan=state['plan'],
            found_files=state['relevant_files']
        )
        
        try:
            data = generate_content_rest(prompt, mime_type="application/json")
            if data['status'] == "rejected":
                return {
                    "history": [f"Context missing: {data['missing']}"],
                    "verifier_status": "rejected",
                    "missing_symbols": data['missing']
                }
        except: pass
        
        return {"history": ["Context verified."], "verifier_status": "approved"}

    # --- CODER ---
    def coder_agent(self, state: AgentState):
        print("--- CODER AGENT ---")
        
        current_iter = state.get("iterations", 0)
        if current_iter > 5:
            return {"history": ["FATAL: Loop limit reached."], "syntax_status": "fatal_error"}
        
        context_str = ""
        for path in state['relevant_files']:
            # Use the new tool that adds line numbers
            numbered = self.tools.read_file_with_lines(path)
            context_str += f"\nFile: {path}\n```\n{numbered}\n```\n"
        
        previous_err = state.get('history', [])[-1] if 'failed' in str(state.get('history', [])) else 'None'
        
        
        # USE PROMPT MANAGER
        prompt = prompt_manager.render(
            'coder', 'user',
            requirement=state['requirement'],
            language=state['language'],
            project_type=state['project_type'],
            plan=state['plan'],
            dependency_context=state.get('dependency_context', 'Unknown'), # <--- Pass it here
            previous_errors=previous_err,
            context_code=context_str
        )
        
        try:
            data = generate_content_rest(prompt, schema=CODER_RESPONSE_SCHEMA)
            for edit in data.get("edits", []):
                fpath = edit['filepath']
                action = edit['action']
                
                if action == "create_file":
                    # For new files, replace_block is the whole content
                    self.tools.write_file(fpath, edit['replace_block'])
                    
                elif action == "replace":
                    # For modifications, use the patcher
                    result = self.tools.apply_patch(
                        fpath, 
                        edit.get('search_block', ''), 
                        edit.get('replace_block', '')
                    )
                    if "Error" in result:
                        # If fuzzy patch failed, log it for the "Reflexion" loop
                        return {"history": [f"Patch failed for {fpath}: {result} - Try matching the existing code more precisely."]}
            return {
                "code_changes": result, 
                "history": [f"Code generated (Iter {current_iter})."],
                "syntax_status": "pending", 
                "iterations": current_iter + 1
            }
        except Exception as e:
            return {"history": [f"Coder Error: {e}"], "syntax_status": "failed", "iterations": current_iter + 1}

    # --- SYNTAX CHECKER ---
    def syntax_checker_agent(self, state: AgentState):
        print("--- SYNTAX CHECKER ---")
        errors = []
        if state.get("syntax_status") == "fatal_error": return {}

        for filepath, content in state['code_changes'].items():
            if filepath.endswith(".py"):
                try: ast.parse(content)
                except SyntaxError as e: errors.append(f"{filepath}: {e}")
            elif filepath.endswith(".java"):
                if content.count("{") != content.count("}"): errors.append(f"{filepath}: Braces mismatch.")
        
        if errors:
            return {"history": [f"Syntax: {errors}"], "syntax_status": "failed"}
        return {"history": ["Syntax OK."], "syntax_status": "passed"}

    # --- TESTER ---
    def tester_agent(self, state: AgentState):
        print("--- TESTER AGENT ---")
        changes_context = "\n".join([f"File: {p}\n{c}" for p, c in state['code_changes'].items()])
        
        # USE PROMPT MANAGER
        prompt = prompt_manager.render(
            'tester', 'user',
            language=state['language'],
            changes_context=changes_context
        )
        
        try:
            data = generate_content_rest(prompt, mime_type="application/json")
            return {"test_code": {data['filepath']: data['content']}, "history": ["Tests generated."]}
        except:
            return {"test_code": {}, "history": ["Tests skipped."]}

    # --- GIT MANAGER ---
    def git_manager_agent(self, state: AgentState):
        print("--- GIT MANAGER ---")
        branch = f"feature/ai-{state['requirement'][:10].replace(' ','-')}"
        self.tools.create_branch(branch)
        
        for p, c in state['code_changes'].items(): self.tools.write_file(p, c)
        for p, c in state['test_code'].items(): self.tools.write_file(p, c)
        
        res = self.tools.commit_and_push(f"AI: {state['requirement']}")
        return {"history": [f"Git: {res}"]}

# --- 5. Build Graph ---

def construct_graph(team):
    wf = StateGraph(AgentState)
    
    wf.add_node("analyst", team.analyst_agent)
    wf.add_node("planner", team.planner_agent)
    wf.add_node("researcher", team.researcher_agent)
    wf.add_node("verifier", team.context_verifier_agent) # Added Verifier
    wf.add_node("coder", team.coder_agent)
    wf.add_node("syntax", team.syntax_checker_agent)
    wf.add_node("tester", team.tester_agent)
    wf.add_node("git", team.git_manager_agent)
    
    # Edges
    wf.set_entry_point("analyst")
    wf.add_edge("analyst", "planner")
    wf.add_edge("planner", "researcher")
    wf.add_edge("researcher", "verifier")
    
    def check_verifier(state):
        return "researcher" if state.get("verifier_status") == "rejected" else "coder"
        
    wf.add_conditional_edges("verifier", check_verifier, {"researcher": "researcher", "coder": "coder"})
    
    wf.add_edge("coder", "syntax")
    
    def check_syntax(state):
        if state.get("syntax_status") == "fatal_error": return END
        return "coder" if state.get("syntax_status") == "failed" else "tester"
        
    wf.add_conditional_edges("syntax", check_syntax, {"coder": "coder", "tester": "tester", END: END})
    
    wf.add_edge("tester", "git")
    wf.add_edge("git", END)
    
    return wf.compile()

def build_agent_graph(logical_name: str, repo_path: str):
    team = AutonomousDevTeam(logical_name, repo_path)
    return construct_graph(team)

def get_graph_mermaid():
    """
    Generates the Mermaid diagram for the agent graph.
    Uses a dummy team to avoid expensive initialization.
    """
    class DummyTeam:
        def planner_agent(self, state): pass
        def researcher_agent(self, state): pass
        def context_verifier_agent(self, state): pass
        def coder_agent(self, state): pass
        def syntax_checker_agent(self, state): pass
        def tester_agent(self, state): pass
        def git_manager_agent(self, state): pass

    try:
        graph = construct_graph(DummyTeam())
        mermaid_graph = graph.get_graph().draw_mermaid()
        
        # Add Custom Styling
        # Dark Blue for agents, Light Blue for End
        custom_styles = """
        classDef darkBlue fill:#2D3748,stroke:#1a202c,stroke-width:2px,color:#ffffff;
        classDef lightBlue fill:#BEE3F8,stroke:#3182CE,stroke-width:2px,color:#000000;
        class planner,researcher,verifier,coder,syntax,tester,git darkBlue;
        class __end__ lightBlue;
        """
        return mermaid_graph + custom_styles
    except Exception as e:
        return f"Error generating mermaid graph: {str(e)}"