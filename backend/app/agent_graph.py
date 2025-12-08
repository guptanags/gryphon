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

# --- Configuration ---
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
MODEL_ID = "gemini-1.5-flash-001" 

# --- 1. JSON Schemas ---
CODER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thought_process": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "content": {"type": "string"},
                    "action": {"type": "string", "enum": ["create", "overwrite"]}
                },
                "required": ["filepath", "content", "action"]
            }
        }
    },
    "required": ["thought_process", "files"]
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
            
            top_file_paths = [res.payload['file_path'] for res in file_results]
            print(f"   [Researcher] Top Candidate Files: {top_file_paths}")
            
            # Add them to our "Found" list so Coder gets the whole file if needed
            found_files.extend(top_file_paths)
            
            # STEP B: Chunk Level Scoped Search (Optional optimization)
            # If you want to find specific *lines* inside those files to save context window:
            chunk_results = self.qdrant.search(
                collection_name=CODE_COLLECTION_NAME,
                query_vector=req_vector,
                limit=10,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.file_path", 
                            match=models.MatchAny(any=top_file_paths) # <--- SCOPED FILTER
                        )
                    ]
                )
            )
        
        return {"relevant_files": list(set(found_files)), "history": [f"Researched: {found_files}"]}

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
            context_str += f"\nFile: {path}\n```\n{self.tools.read_file(path)}\n```\n"
            
        previous_err = state.get('history', [])[-1] if 'failed' in str(state.get('history', [])) else 'None'
        
        # USE PROMPT MANAGER
        prompt = prompt_manager.render(
            'coder', 'user',
            requirement=state['requirement'],
            language=state['language'],
            project_type=state['project_type'],
            plan=state['plan'],
            previous_errors=previous_err,
            context_code=context_str
        )
        
        try:
            data = generate_content_rest(prompt, schema=CODER_RESPONSE_SCHEMA)
            changes = {f['filepath']: f['content'] for f in data.get("files", [])}
            return {
                "code_changes": changes, 
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
    
    wf.add_node("planner", team.planner_agent)
    wf.add_node("researcher", team.researcher_agent)
    wf.add_node("verifier", team.context_verifier_agent) # Added Verifier
    wf.add_node("coder", team.coder_agent)
    wf.add_node("syntax", team.syntax_checker_agent)
    wf.add_node("tester", team.tester_agent)
    wf.add_node("git", team.git_manager_agent)
    
    # Edges
    wf.set_entry_point("planner")
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
        return graph.get_graph().draw_mermaid()
    except Exception as e:
        return f"Error generating mermaid graph: {str(e)}"