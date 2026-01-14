import os
import json
import requests
import subprocess
import ast
import operator
import logging
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

# --- Third Party ---
from qdrant_client import models
try:
    from flashrank import Ranker, RerankRequest
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False

# --- Internal Modules ---
from agent_tools import AgentTools
from pipeline import setup_qdrant, CODE_COLLECTION_NAME, FILE_SUMMARY_COLLECTION_NAME, DOCS_COLLECTION_NAME
from prompt_manager import prompt_manager
from audit_manager import get_logger

# --- Configuration ---
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
MODEL_ID = "gemini-1.5-flash-001" 

log = logging.getLogger(__name__)

# --- 1. JSON Schemas (Strict Output Guardrails) ---

# Schema for Coder (Patching Strategy)
CODER_PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "thought_process": {
            "type": "string",
            "description": "Step-by-step reasoning for the changes."
        },
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "search_block": {
                        "type": "string", 
                        "description": "The EXACT existing code block to replace (for 'replace' action)."
                    },
                    "replace_block": {
                        "type": "string", 
                        "description": "The new code content."
                    },
                    "action": {
                        "type": "string", 
                        "enum": ["replace", "create_file"],
                        "description": "Use 'replace' to modify existing files, 'create_file' for new ones."
                    }
                },
                "required": ["filepath", "action"]
            }
        }
    },
    "required": ["thought_process", "edits"]
}

# --- 2. Helper Functions (LLM & Auth) ---

def get_access_token():
    """Retrieves Google Cloud Access Token."""
    token = os.environ.get("GCLOUD_ACCESS_TOKEN")
    if token: return token
    try:
        return subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode("utf-8").strip()
    except Exception as e:
        log.error(f"Auth Error: {e}")
        return None

def generate_content_rest(prompt: str, schema: dict = None, mime_type: str = "text/plain", run_id: str = "sys", agent_name: str = "System"):
    """
    Calls Vertex AI via REST with JSON enforcement and Audit Logging.
    """
    logger = get_logger(run_id)
    token = get_access_token()
    if not token: 
        logger.log(agent_name, "Error", "No GCloud Token")
        raise ValueError("No GCloud Access Token found.")

    url = f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/{VERTEX_PROJECT_ID}/locations/{VERTEX_LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"

    # Config
    gen_config = {"temperature": 0.1, "maxOutputTokens": 8192}
    if schema:
        gen_config["responseMimeType"] = "application/json"
        gen_config["responseSchema"] = schema
    elif mime_type == "application/json":
        gen_config["responseMimeType"] = "application/json"

    # Payload
    payload = {
        "contents": [{ "role": "user", "parts": [{"text": prompt}] }],
        "generationConfig": gen_config
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Audit Log: Request
    logger.log(agent_name, "LLM Request", "Sending prompt", {"prompt_preview": prompt[:200], "schema": str(schema is not None)})

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result_json = response.json()
        raw_text = result_json['candidates'][0]['content']['parts'][0]['text']
        
        # Audit Log: Response
        logger.log(agent_name, "LLM Response", "Received output", {"text_preview": raw_text[:200]})

        if schema or mime_type == "application/json":
            return json.loads(raw_text)
        return raw_text

    except Exception as e:
        logger.log(agent_name, "LLM Error", str(e))
        raise

# --- 3. Agent State Definition ---

class AgentState(TypedDict):
    # Meta
    run_id: str
    logical_name: str
    repo_path: str
    
    # Requirements
    raw_requirement: str
    requirement: str        # Refined by Analyst
    
    # Context
    language: str
    project_type: str
    dependency_context: str # From pom.xml/package.json
    
    # Artifacts
    plan: str
    multi_plans: str           # Raw text of 3 options
    critic_verdict: Dict       # Scores and selection
    selected_plan: str         # The final plan used by Coder
    relevant_files: List[str]
    code_changes: Dict[str, str] # Deprecated in favor of edits logic, but kept for state
    edits: List[Dict]            # The structured patch list
    test_code: Dict[str, str]
    style_example_code: str      # Stores the 'Gold Standard' code

    # Flags & Counters (Circuit Breakers)
    verifier_status: str    # approved/rejected
    missing_symbols: List[str]
    syntax_status: str      # passed/failed/fatal_error
    review_status: str      # passed/failed
    iterations: int
    
    # Output Log
    history: Annotated[List[str], operator.add]

# --- 4. The Autonomous Team Class ---

class AutonomousDevTeam:
    def __init__(self, logical_name: str, repo_path: str):
        self.logical_name = logical_name
        self.repo_path = repo_path
        self.tools = AgentTools(repo_path)
        self.qdrant = setup_qdrant()
        self.embed_model = setup_qdrant() # Placeholder, usually needs Vertex init
        # Initialize Reranker (runs locally on CPU)
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank") if HAS_RERANKER else None

    # --- NODE 1: REQUIREMENT ANALYST ---
    def analyst_agent(self, state: AgentState):
        run_id = state.get('run_id', 'unknown')
        logger = get_logger(run_id)
        print("--- ANALYST AGENT ---")
        
        # 1. Detect Stack
        files_list = self.tools.list_files()
        language, project_type = "python", "generic"
        if any(f.endswith("pom.xml") for f in files_list):
            language, project_type = "java", "spring-boot"
        elif any(f.endswith("package.json") for f in files_list):
            language, project_type = "typescript", "node"

        # 2. Refine Requirement
        prompt = prompt_manager.render(
            'analyst', 'user',
            requirement=state['raw_requirement'],
            language=language,
            project_type=project_type
        )
        
        refined_text = generate_content_rest(prompt, run_id=run_id, agent_name="Analyst")
        
        return {
            "requirement": refined_text,
            "language": language,
            "project_type": project_type,
            "history": ["Requirement refined."]
        }

    # --- NODE 2: PLANNER ---
    def planner_agent(self, state: AgentState):
        run_id = state.get('run_id')
        logger = get_logger(run_id)
        
        prompt = prompt_manager.render(
            'planner', 'user',
            requirement=state['requirement'],
            language=state['language'],
            project_type=state['project_type'],
            skeleton=self.tools.generate_repo_skeleton()
        )
        
        multi_plans = generate_content_rest(prompt, run_id=run_id, agent_name="Planner")
        
        logger.log("Planner", "Generation", "Generated 3 architectural options.")
        return {"multi_plans": multi_plans, "history": ["Generated 3 plans for debate."]}

    # 2. CRITIC: Scores and Selects the Winner
    def critic_agent(self, state: AgentState):
        run_id = state.get('run_id')
        logger = get_logger(run_id)
        
        prompt = prompt_manager.render(
            'critic', 'user',
            requirement=state['requirement'],
            plans_text=state['multi_plans']
        )
        
        # We enforce JSON for the Critic
        verdict = generate_content_rest(prompt, mime_type="application/json", run_id=run_id, agent_name="Critic")
        
        selected_option = verdict.get('selection', 'OptionB')
        
        # Logic to extract the specific text of the selected plan from the multi_plans string
        # Simplified: We pass the whole debate context to the Researcher
        logger.log("Critic", "Decision", f"Selected {selected_option}", {"verdict": verdict})
        
        return {
            "critic_verdict": verdict,
            "selected_plan": f"Selected Strategy: {selected_option}\nDetails: {state['multi_plans']}",
            "history": [f"Critic selected {selected_option} based on SOLID principles."]
        }

    # --- NODE 3: RESEARCHER (Hierarchical + Rerank) ---
    def researcher_agent(self, state: AgentState):
        run_id = state.get('run_id')
        logger = get_logger(run_id)
        
        print("--- RESEARCHER AGENT (Full Context) ---")
        
        # A. File Discovery (from Plan)
        prompt = prompt_manager.render('researcher', 'extract_files', plan=state['selected_plan'])
        extraction = generate_content_rest(prompt, mime_type="application/json", run_id=run_id, agent_name="Researcher")
        found_files = [f for f in extraction.get("files", []) if self.tools.read_file(f)]

        # B. Dependency Context (Reusable Components)
        dep_context = "No manifest found."
        try:
            dep_res = self.qdrant.scroll(
                collection_name=DOCS_COLLECTION_NAME,
                scroll_filter=models.Filter(must=[models.FieldCondition(key="metadata.doc_type", match=models.MatchValue(value="dependency_graph"))]),
                limit=1, with_payload=True
            )
            if dep_res[0]: dep_context = dep_res[0][0].payload['content']
        except: pass

        # C. Style Hunter (Dynamic Few-Shot)
        style_code = ""
        if found_files:
            example_path = self.tools.find_style_example(found_files[0])
            if example_path: style_code = self.tools.read_file(example_path)

        logger.log("Researcher", "Files Found", "Found files: {}".format(found_files))
        logger.log("Researcher", "Dependency Context", "Dependency context: {}".format(dep_context))
        logger.log("Researcher", "Style Example", "Style example: {}".format(style_code))

        return {
            "relevant_files": list(set(found_files)),
            "dependency_context": dep_context,
            "style_example_code": style_code,
            "history": ["Gathered code, deps, and style examples."]
        }
    
        # --- NODE 5: CODER (Patching Strategy) ---
    def coder_agent(self, state: AgentState):
        run_id = state.get('run_id')
        logger = get_logger(run_id)
        print("--- CODER AGENT ---")
        
        # Circuit Breaker
        current_iter = state.get("iterations", 0)
        if current_iter > 3:
            return {"syntax_status": "fatal_error", "history": ["Max iterations reached."]}

        # Prepare Context with Line Numbers
        context_str = ""
        for path in state['relevant_files']:
            numbered = self.tools.read_file_with_lines(path)
            context_str += f"\nFile: {path}\n```\n{numbered}\n```\n"

        prev_error = state.get('history', [])[-1] if 'failed' in str(state.get('history', [])) else 'None'

        prompt = prompt_manager.render(
            'coder', 'user',
            requirement=state['requirement'],
            language=state['language'],
            project_type=state['project_type'],
            plan=state['plan'],
            dependency_context=state.get('dependency_context', ''),
            context_code=context_str,
            previous_errors=prev_error
        )

        try:
            # Force Patch Schema
            data = generate_content_rest(prompt, schema=CODER_PATCH_SCHEMA, run_id=run_id, agent_name="Coder")
            
            # Apply Edits (Virtual Application or storing for Reviewer)
            # We store them in state['edits'] for the Reviewer/Syntax Checker to validate BEFORE writing
            return {
                "edits": data.get("edits", []),
                "syntax_status": "pending",
                "review_status": "pending",
                "iterations": current_iter + 1,
                "history": [f"Generated {len(data.get('edits', []))} edits."]
            }
        except Exception as e:
            return {"syntax_status": "failed", "history": [f"Coder Error: {e}"]}

    # --- NODE 6: SYNTAX CHECKER ---
    def syntax_checker_agent(self, state: AgentState):
        print("--- SYNTAX CHECKER ---")
        errors = []
        if state.get("syntax_status") == "fatal_error": return {}

        # We simulate the patch application to check syntax
        for edit in state.get("edits", []):
            if edit['action'] == 'create_file':
                content = edit['replace_block']
            else:
                # Simulation of patch (simplified)
                # In prod, apply to temp copy. Here we check the raw block syntax if possible
                content = edit['replace_block']

            # Python Syntax
            if edit['filepath'].endswith(".py"):
                try: ast.parse(content)
                except SyntaxError as e: errors.append(f"{edit['filepath']}: {e}")
            
            # Java Braces
            elif edit['filepath'].endswith(".java"):
                if content.count("{") != content.count("}"): 
                    errors.append(f"{edit['filepath']}: Braces mismatch.")

        if errors:
            return {"syntax_status": "failed", "history": [f"Syntax Errors: {errors}"]}
        return {"syntax_status": "passed", "history": ["Syntax Check Passed."]}

    # --- NODE 7: TESTER ---
    def tester_agent(self, state: AgentState):
        run_id = state.get('run_id')
        print("--- TESTER AGENT ---")
        
        changes_ctx = "\n".join([f"File: {e['filepath']}\n{e.get('replace_block','')}" for e in state.get('edits', [])])
        
        prompt = prompt_manager.render('tester', 'user', language=state['language'], changes_context=changes_ctx)
        
        try:
            data = generate_content_rest(prompt, mime_type="application/json", run_id=run_id, agent_name="Tester")
            return {"test_code": {data['filepath']: data['content']}, "history": ["Tests Generated."]}
        except:
            return {"test_code": {}, "history": ["Tests skipped."]}

    def executor_agent(self, state: AgentState):
        run_id = state.get('run_id')
        logger = get_logger(run_id)
        print("--- EXECUTOR AGENT (Runtime) ---")

        # 1. Apply the edits temporarily to the disk (to run tests)
        # In this phase, we write the code to the branch
        for edit in state.get('edits', []):
            if edit['action'] == 'create_file':
                self.tools.write_file(edit['filepath'], edit['replace_block'])
            elif edit['action'] == 'replace':
                self.tools.apply_patch(edit['filepath'], edit['search_block'], edit['replace_block'])
        
        for p, c in state.get('test_code', {}).items():
            self.tools.write_file(p, c)

        # 2. Run the tests
        logger.log("Executor", "Runtime", "Executing generated Unit Tests")
        test_results = self.tools.run_unit_tests(state['language'])

        if test_results['success']:
            logger.log("Executor", "Success", "All tests passed at runtime.")
            return {"syntax_status": "passed", "history": ["Runtime execution successful."]}
        else:
            # Capture the failure logs to feed back to the Coder
            logger.log("Executor", "Failure", "Tests failed at runtime", {"logs": test_results['output']})
            return {
                "syntax_status": "failed", 
                "history": [f"Runtime Failure: {test_results['output'][:500]}"] # Feed logs back
            }

    # --- NODE 8: GIT MANAGER ---
    def git_manager_agent(self, state: AgentState):
        run_id = state.get('run_id')
        logger = get_logger(run_id)
        print("--- GIT MANAGER ---")
        
        # 1. Branch
        branch = f"feature/ai-{str(state['run_id'])[:8]}"
        self.tools.create_branch(branch)
        
        # 2. Apply Edits (Real Disk Write)
        for edit in state.get('edits', []):
            if edit['action'] == 'create_file':
                self.tools.write_file(edit['filepath'], edit['replace_block'])
            elif edit['action'] == 'replace':
                res = self.tools.apply_patch(edit['filepath'], edit['search_block'], edit['replace_block'])
                if "Error" in res:
                    logger.log("GitManager", "PatchError", f"Failed to patch {edit['filepath']}", {"error": res})
        
        # 3. Write Tests
        for p, c in state.get('test_code', {}).items():
            self.tools.write_file(p, c)
            
        # 4. Commit
        res = self.tools.commit_and_push(f"AI: {state['requirement']}")
        return {"history": [f"Git: {res}"]}

# --- 5. Build Graph ---

def build_agent_graph(logical_name: str, repo_path: str):
    team = AutonomousDevTeam(logical_name, repo_path)
    wf = StateGraph(AgentState)
    
    wf.add_node("analyst", team.analyst_agent)
    wf.add_node("planner", team.planner_agent)
    wf.add_node("critic", team.critic_agent) 
    wf.add_node("researcher", team.researcher_agent)
    wf.add_node("coder", team.coder_agent)
    wf.add_node("syntax", team.syntax_checker_agent)
    wf.add_node("tester", team.tester_agent)
    wf.add_node("executor", team.executor_agent)
    wf.add_node("git", team.git_manager_agent)
    
    # Workflow
    wf.set_entry_point("analyst")
    wf.add_edge("analyst", "planner")
    wf.add_edge("planner", "critic")

    def check_consensus(state):
        scores = state['critic_verdict'].get('scores', {})
        max_score = max(scores.values()) if scores else 0
        
        if max_score < 5:
            # If all plans are bad, send back to Planner to rethink
            return "planner"
        return "researcher"

    wf.add_conditional_edges("critic", check_consensus, {
        "planner": "planner", 
        "researcher": "researcher"
    })
    
    wf.add_edge("researcher", "verifier")
    
    def check_context(state):
        return "researcher" if state.get("verifier_status") == "rejected" else "coder"
    wf.add_conditional_edges("verifier", check_context, {"researcher": "researcher", "coder": "coder"})
    
    wf.add_edge("coder", "syntax")
    
    def check_syntax(state):
        if state.get("syntax_status") == "fatal_error": return END
        return "coder" if state.get("syntax_status") == "failed" else "tester"
    wf.add_conditional_edges("syntax", check_syntax, {"coder": "coder", "tester": "tester", END: END})
    wf.add_edge("tester", "executor")

    def check_runtime_results(state):
        if state.get("syntax_status") == "failed":
            # LOOP BACK: Send error logs to Coder to fix
            return "coder"
        return "git"
    wf.add_conditional_edges("executor", check_runtime_results, {
        "coder": "coder", 
        "git": "git"
    })    
    
    wf.add_edge("git", END)
    
    return wf.compile()