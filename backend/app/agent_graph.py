import os
import json
import requests
import subprocess
import ast
import operator
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

# Internal Modules
from agent_tools import AgentTools
from pipeline import setup_qdrant, setup_vertex_ai, CODE_COLLECTION_NAME
from qdrant_client import models

# --- Configuration ---
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID","vrittera")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
MODEL_ID = "gemini-2.5-flash-lite" 

# --- 1. JSON Schemas for Structured Output ---

# Schema for the Coder Agent (Strict JSON enforcement)
CODER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thought_process": {
            "type": "string",
            "description": "Explain the implementation logic and changes made."
        },
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Relative path to the file"},
                    "content": {"type": "string", "description": "The COMPLETE new content of the file. NO PLACEHOLDERS."},
                    "action": {
                        "type": "string", 
                        "enum": ["create", "overwrite"],
                        "description": "Whether to create a new file or overwrite an existing one."
                    }
                },
                "required": ["filepath", "content", "action"]
            }
        }
    },
    "required": ["thought_process", "files"]
}

# --- 2. Helper Functions (REST & Auth) ---

def get_access_token():
    """Gets Google Cloud Access Token via gcloud CLI or Env Var."""
    token = os.environ.get("GCLOUD_ACCESS_TOKEN")
    if token:
        return token
    try:
        # Fallback to gcloud command
        return subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode("utf-8").strip()
    except Exception as e:
        print(f"Error getting gcloud token: {e}")
        return None

def generate_content_rest(prompt: str, schema: dict = None, mime_type: str = "text/plain"):
    """
    Calls Vertex AI via REST API. Supports JSON Mode if schema is provided.
    """
    token = get_access_token()
    if not token:
        raise ValueError("No GCloud Access Token found.")

    url = f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/{VERTEX_PROJECT_ID}/locations/{VERTEX_LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"

    generation_config = {
        "temperature": 0.1,
        "maxOutputTokens": 8192
    }
    
    # Enable JSON Mode if schema is present
    if schema:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = schema
    elif mime_type == "application/json":
        generation_config["responseMimeType"] = "application/json"

    payload = {
        "contents": [{ "role": "user", "parts": [{"text": prompt}] }],
        "generationConfig": generation_config
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        raw_text = result['candidates'][0]['content']['parts'][0]['text']
        
        if schema or mime_type == "application/json":
            return json.loads(raw_text)
        return raw_text
    except Exception as e:
        print(f"Vertex API Error: {e}")
        if 'response' in locals():
            print(f"Response: {response.text}")
        raise

# --- 3. Define the State ---

class AgentState(TypedDict):
    logical_name: str
    repo_path: str
    requirement: str
    
    # Context
    language: str       # 'python', 'java', 'typescript'
    project_type: str   # 'flask', 'spring-boot', 'node'
    
    # Scratchpad
    plan: str
    relevant_files: List[str]
    code_changes: Dict[str, str] # K=path, V=content
    test_code: Dict[str, str]
    
    # Robustness & Circuit Breakers
    syntax_status: str  # 'passed', 'failed', 'fatal_error'
    review_status: str  # 'passed', 'failed'
    iterations: int     # Counter to prevent infinite loops
    
    # Logs
    history: Annotated[List[str], operator.add]

# --- 4. The Agent Team Class ---

class AutonomousDevTeam:
    def __init__(self, logical_name: str, repo_path: str):
        self.logical_name = logical_name
        self.repo_path = repo_path
        self.tools = AgentTools(repo_path)
        
        # Clients
        self.qdrant = setup_qdrant() 
        # We use the SDK for embeddings (easier) but REST for Generation
        self.embed_model, _ = setup_vertex_ai() 

    # --- HELPER TOOLS ---
    def find_definition_tool(self, symbol_name: str):
        """Finds the file defining a specific Class or Function using Qdrant."""
        try:
            results, _ = self.qdrant.scroll(
                collection_name=CODE_COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="metadata.chunk_name", match=models.MatchValue(value=symbol_name))]
                ),
                limit=1,
                with_payload=True
            )
            if results:
                return results[0].payload['file_path']
        except Exception:
            pass
        return None

    # --- NODE: PLANNER ---
    def planner_agent(self, state: AgentState):
        print("--- PLANNER AGENT ---")
        files_list = self.tools.list_files()
        
        # 1. Detect Stack
        language = "python"
        project_type = "generic"
        
        if any(f.endswith("pom.xml") for f in files_list):
            language = "java"
            project_type = "spring-boot"
        elif any(f.endswith("package.json") for f in files_list):
            language = "typescript"
            project_type = "node"
        
        # 2. Generate Skeleton (Advanced Context)
        # Assumes generate_repo_skeleton exists in AgentTools
        skeleton = self.tools.generate_repo_skeleton()
        if len(skeleton) > 60000: skeleton = skeleton[:60000] + "\n...(truncated)"
        
        prompt = f"""
        You are a Technical Architect.
        Requirement: "{state['requirement']}"
        
        Context:
        - Language: {language}
        - Framework: {project_type}
        
        Repo Structure (Skeleton):
        {skeleton}
        
        Create a concise implementation plan.
        1. Identify specific existing files to modify.
        2. Identify new files needed (including tests).
        3. Explain the logic briefly.
        """
        
        plan_text = generate_content_rest(prompt)
        
        return {
            "plan": plan_text, 
            "language": language, 
            "project_type": project_type,
            "iterations": 0, # Reset counter
            "history": [f"Plan generated ({language}/{project_type})."]
        }

    # --- NODE: RESEARCHER ---
    def researcher_agent(self, state: AgentState):
        print("--- RESEARCHER AGENT ---")
        found_files = []
        
        # 1. Semantic Extraction from Plan
        prompt = f"""
        Extract the specific file paths mentioned in this plan that need to be read or modified.
        Also extract any specific Class or Function names mentioned.
        Plan: "{state['plan']}"
        Return JSON: {{ "files": ["path/to/file1"], "symbols": ["ClassName"] }}
        """
        try:
            extraction = generate_content_rest(prompt, mime_type="application/json")
            
            # Verify Files
            for f_path in extraction.get("files", []):
                if self.tools.read_file(f_path) and "Error" not in self.tools.read_file(f_path):
                    found_files.append(f_path)
            
            # Verify Symbols (Go To Definition)
            for symbol in extraction.get("symbols", []):
                def_path = self.find_definition_tool(symbol)
                if def_path and def_path not in found_files:
                    found_files.append(def_path)
                    
        except Exception as e:
            print(f"Researcher extraction error: {e}")

        # 2. Test-Driven Discovery (Fallback)
        if not found_files:
            print("   [Researcher] Fallback: Test-Driven Discovery...")
            try:
                req_vector = self.embed_model.get_embeddings([state['requirement']])[0].values
                test_results = self.qdrant.search(
                    collection_name=CODE_COLLECTION_NAME,
                    query_vector=req_vector,
                    limit=3,
                    with_payload=True,
                    query_filter=models.Filter(must=[models.FieldCondition(key="metadata.file_path", match=models.MatchText(text="test"))])
                )
                for res in test_results:
                    found_files.append(res.payload['file_path'])
            except Exception:
                pass

        return {"relevant_files": list(set(found_files)), "history": [f"Researched files: {found_files}"]}

    # --- NODE: CODER (REST + JSON + Circuit Breaker) ---
    def coder_agent(self, state: AgentState):
        print("--- CODER AGENT ---")
        
        # Circuit Breaker Check
        current_iter = state.get("iterations", 0)
        if current_iter > 5:
            return {"history": ["FATAL: Max iterations reached. Agent stuck."], "syntax_status": "fatal_error"}
        
        # Prepare Context
        context = ""
        for path in state['relevant_files']:
            content = self.tools.read_file(path)
            context += f"\nFile: {path}\n```\n{content}\n```\n"
            
        prompt = f"""
        You are a Senior Developer. Implement this requirement: "{state['requirement']}"
        Language: {state['language']}
        Framework: {state['project_type']}
        
        Plan: {state['plan']}
        
        Previous Errors (if any): {state.get('history', [])[-1] if 'failed' in str(state.get('history', [])) else 'None'}
        
        Context Code:
        {context}
        
        **CRITICAL INSTRUCTIONS:**
        1. Return the COMPLETE source code for any modified file. **NO PLACEHOLDERS**.
        2. If Java: Ensure correct package declarations and imports.
        3. Output must match the JSON Schema provided.
        """
        
        try:
            data = generate_content_rest(prompt, schema=CODER_RESPONSE_SCHEMA)
            
            changes = {}
            for file_obj in data.get("files", []):
                changes[file_obj['filepath']] = file_obj['content']
                
            return {
                "code_changes": changes, 
                "history": [f"Code generated (Iter {current_iter})."],
                "syntax_status": "pending", 
                "review_status": "pending",
                "iterations": current_iter + 1
            }
        except Exception as e:
            return {"history": [f"Coder Error: {str(e)}"], "syntax_status": "failed", "iterations": current_iter + 1}

    # --- NODE: SYNTAX CHECKER ---
    def syntax_checker_agent(self, state: AgentState):
        print("--- SYNTAX CHECKER ---")
        errors = []
        
        if state.get("syntax_status") == "fatal_error":
            return {"history": ["Skipping syntax check due to fatal error."]}

        for filepath, content in state['code_changes'].items():
            # Python Syntax
            if filepath.endswith(".py"):
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    errors.append(f"{filepath}: {e}")
            
            # Java Braces Check (Heuristic)
            elif filepath.endswith(".java"):
                if content.count("{") != content.count("}"):
                    errors.append(f"{filepath}: Mismatched curly braces.")
        
        if errors:
            return {
                "history": [f"Syntax Errors: {'; '.join(errors)}"],
                "syntax_status": "failed"
            }
        return {"history": ["Syntax check passed."], "syntax_status": "passed"}

    # --- NODE: REVIEWER ---
    def reviewer_agent(self, state: AgentState):
        print("--- REVIEWER AGENT ---")
        errors = []
        
        for filepath, content in state['code_changes'].items():
            # Lazy LLM Check
            if "# ..." in content or "// ..." in content or "existing code" in content:
                errors.append(f"{filepath} contains lazy placeholders.")
            
            # Size reduction check
            original = self.tools.read_file(filepath)
            if original and "Error" not in original:
                if len(content) < len(original) * 0.5:
                    errors.append(f"{filepath} is dangerously smaller than original.")

        if errors:
            return {
                "history": [f"Review Failed: {'; '.join(errors)}"],
                "review_status": "failed"
            }
        return {"history": ["Code review passed."], "review_status": "passed"}

    # --- NODE: TESTER ---
    def tester_agent(self, state: AgentState):
        print("--- TESTER AGENT ---")
        
        changes_context = ""
        for path, content in state['code_changes'].items():
            changes_context += f"\nFile: {path}\n{content}\n"
            
        prompt = f"""
        You are an SDET. Write a unit test for this code.
        Language: {state['language']}
        
        Code:
        {changes_context}
        
        Instructions:
        - If Java: Use JUnit 5 & Mockito.
        - If Python: Use Pytest.
        - Return JSON: {{ "filepath": "tests/TestFile.java", "content": "..." }}
        """
        
        try:
            data = generate_content_rest(prompt, mime_type="application/json")
            return {
                "test_code": {data['filepath']: data['content']},
                "history": ["Tests generated."]
            }
        except Exception:
            return {"test_code": {}, "history": ["Test generation skipped due to error."]}

    # --- NODE: GIT MANAGER ---
    def git_manager_agent(self, state: AgentState):
        print("--- GIT MANAGER AGENT ---")
        
        # 1. Create Branch
        clean_req = "".join(c for c in state['requirement'] if c.isalnum() or c == ' ')[:15].replace(' ', '-')
        branch = f"feature/ai-{clean_req}"
        self.tools.create_branch(branch)
        
        # 2. Write Files
        for path, content in state['code_changes'].items():
            self.tools.write_file(path, content)
        for path, content in state['test_code'].items():
            self.tools.write_file(path, content)
            
        # 3. Commit & Push
        result = self.tools.commit_and_push(f"AI Implementation: {state['requirement']}")
        
        return {"history": [f"Git Ops: {result}"]}

# --- 5. Build the Graph ---

def build_agent_graph(logical_name: str, repo_path: str):
    team = AutonomousDevTeam(logical_name, repo_path)
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("planner", team.planner_agent)
    workflow.add_node("researcher", team.researcher_agent)
    workflow.add_node("coder", team.coder_agent)
    workflow.add_node("syntax_checker", team.syntax_checker_agent)
    workflow.add_node("reviewer", team.reviewer_agent)
    workflow.add_node("tester", team.tester_agent)
    workflow.add_node("git_manager", team.git_manager_agent)
    
    # Edges & Gates
    def check_syntax_gate(state):
        if state.get("syntax_status") == "fatal_error":
            return END
        if state.get("syntax_status") == "failed":
            return "coder"
        return "reviewer"

    def check_review_gate(state):
        if state.get("review_status") == "failed":
            return "coder"
        return "tester"

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "coder")
    
    workflow.add_edge("coder", "syntax_checker")
    
    workflow.add_conditional_edges(
        "syntax_checker",
        check_syntax_gate,
        {"coder": "coder", "reviewer": "reviewer", END: END}
    )
    
    workflow.add_conditional_edges(
        "reviewer",
        check_review_gate,
        {"coder": "coder", "tester": "tester"}
    )
    
    workflow.add_edge("tester", "git_manager")
    workflow.add_edge("git_manager", END)
    
    return workflow.compile()