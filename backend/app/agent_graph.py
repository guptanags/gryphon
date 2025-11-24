# agent_graph.py
import os
import operator
from typing import Annotated, TypedDict, List, Dict
from langgraph.graph import StateGraph, END

import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud.aiplatform_v1.types.content import Part

from agent_tools import AgentTools
from qdrant_client import QdrantClient, models
# Import your existing embedding setup
from pipeline import pipeline_setup_qdrant as setup_qdrant
from pipeline import pipeline_setup_vertex_ai as setup_vertex_ai, CODE_COLLECTION_NAME

# --- 1. Define the State ---
# This is the "memory" passed between agents
class AgentState(TypedDict):
    logical_name: str
    repo_path: str
    requirement: str
    
    # "Scratchpad" data
    relevant_files: List[str] # Found by Researcher
    plan: str                 # Created by Planner
    code_changes: Dict[str, str] # K=Filename, V=NewContent
    test_code: Dict[str, str]    # K=Filename, V=Content
    
    # Logs
    history: Annotated[List[str], operator.add]
    edit_mode: str # 'rewrite' or 'append'

# --- 2. Define the Nodes (The Agents) ---

class AutonomousDevTeam:
    def __init__(self, logical_name: str, repo_path: str):
        self.logical_name = logical_name
        self.repo_path = repo_path
        self.tools = AgentTools(repo_path)
        
        # Clients
        self.qdrant = setup_qdrant()
        self.embed_model, self.gen_model = setup_vertex_ai()

    def planner_agent(self, state: AgentState):
        """Analyzes requirement and creates a step-by-step plan."""
        print("--- PLANNER AGENT ---")
        prompt = f"""
        You are a Technical Lead.
        Requirement: "{state['requirement']}"
        
        Available Files in Repo:
        {self.tools.list_files()[:50]} # Limit to avoid token overflow
        
        Create a concise implementation plan.
        1. Identify which files likely need modification.
        2. Identify what new files (tests) are needed.
        """
        prompt += """
        3. Decide on the editing strategy:
           - 'APPEND': If adding a new standalone class or function at the end of a file.
           - 'REWRITE': If modifying existing logic inside functions.
           
        Return the plan and the Strategy (APPEND or REWRITE).
        """
        response = self.gen_model.generate_content(prompt)
        return {"plan": response.text, "history": ["Plan generated."]}

    def researcher_agent(self, state: AgentState):
        """
        Smart Researcher: First looks for files mentioned in the plan,
        then falls back to vector search if needed.
        """
        print("--- RESEARCHER AGENT ---")
        found_files = []
        
        # --- STRATEGY 1: EXTRACT FROM PLAN (Deterministic) ---
        # The planner sees the file list, so it often knows best.
        prompt = f"""
        You are a code researcher.
        
        The Architect provided this implementation plan:
        "{state['plan']}"
        
        Based on this plan, LIST the specific file paths that need to be modified or read.
        - Return ONLY a comma-separated list of file paths.
        - Do not include new files that don't exist yet.
        - If the plan is vague, return "SEARCH".
        """
        response = self.gen_model.generate_content(prompt)
        clean_response = response.text.strip().replace("'", "").replace('"', "")
        
        if "SEARCH" not in clean_response:
            potential_files = [f.strip() for f in clean_response.split(',')]
            # Verify these files actually exist to avoid hallucinations
            for f_path in potential_files:
                # Try reading the first few bytes to check existence
                content = self.tools.read_file(f_path)
                if content and "Error reading file" not in content:
                    found_files.append(f_path)
            
            if found_files:
                print(f"Researcher: Found target files in plan: {found_files}")

        # --- STRATEGY 2: VECTOR SEARCH (Fallback) ---
        # If the plan didn't yield valid files, use RAG to find semantically relevant code.
        if not found_files:
            print("Researcher: Plan was vague. Falling back to Vector Search...")
            req_vector = self.embed_model.get_embeddings([state['requirement']])[0].values
            
            results = self.qdrant.search(
                collection_name=CODE_COLLECTION_NAME,
                query_vector=req_vector,
                limit=3, # Keep it focused
                with_payload=True
            )
            
            for res in results:
                path = res.payload['file_path']
                found_files.append(path)
                
        # Deduplicate
        found_files = list(set(found_files))
        
        return {"relevant_files": found_files, "history": [f"Researched files: {found_files}"]}
    
    
    def coder_agent(self, state: AgentState):
        """Writes the code changes."""
        print("--- CODER AGENT ---")
        
        # Gather context from the files identified by Researcher
        context = ""
        file_sizes = {} # Track original sizes
        for path in state['relevant_files']:
            content = self.tools.read_file(path)
            file_sizes[path] = len(content)
            context += f"\nFile: {path}\n```\n{content}\n```\n"

        prompt = f"""
        You are a Senior Developer. Implement this requirement: "{state['requirement']}"
        
        Plan: {state['plan']}
        
        Context Code:
        {context}
        
        **CRITICAL INSTRUCTIONS:**
        1. You must return the **COMPLETE** source code for any file you modify.
        2. **DO NOT** use placeholders like `# ... existing code ...` or `// ... rest of file`.
        3. If you are adding a new endpoint or function, DO NOT delete existing functions.
        4. If you output a file that is significantly smaller than the original, you will fail the task.
        5. Ensure all imports are preserved.
        
        Format your response strictly as:
        
        ### FILE: path/to/file.py
        (Full, complete file content here)
        ### END FILE
        """
        
        response = self.gen_model.generate_content(prompt)
                
        # Parse the response to extract file changes
        changes = {}
        raw_text = response.text
        
        # Simple parser (in production, use structured JSON output)
        import re
        file_blocks = re.split(r'### FILE: ', raw_text)
        for block in file_blocks[1:]: # Skip empty first split
            try:
                path, content = block.split('\n', 1)
                content = content.split('### END FILE')[0]
                changes[path.strip()] = content.strip()
            except ValueError:
                continue

        for path, new_content in changes.items():
            old_size = file_sizes.get(path, 0)
            new_size = len(new_content)
            
            # If new file is < 50% of old file, reject it (heuristic)
            if old_size > 0 and new_size < (old_size * 0.5):
                return {
                    "history": [f"ERROR: Coder generated a suspiciously small file for {path}. Rejected to prevent data loss."],
                    # We don't update 'code_changes' so the next step fails or loops (if we built a loop)
                }
                
        return {"code_changes": changes, "history": ["Code generated."]}

    def tester_agent(self, state: AgentState):
        """Generates a test file for the changes."""
        print("--- TESTER AGENT ---")
        
        changes_context = ""
        for path, content in state['code_changes'].items():
            changes_context += f"\nFile: {path}\n{content}\n"
            
        prompt = f"""
        You are an SDET. Write a unit test for this new code.
        
        New Code:
        {changes_context}
        
        Return only the python test code.
        """
        response = self.gen_model.generate_content(prompt)
        
        # Heuristic to name the test file
        first_file = list(state['code_changes'].keys())[0] if state['code_changes'] else "script.py"
        test_filename = f"tests/test_feature_{os.path.basename(first_file)}"
        
        # Clean up code blocks
        test_code = response.text.replace("```python", "").replace("```", "")
        
        return {"test_code": {test_filename: test_code}, "history": ["Tests generated."]}

    def git_manager_agent(self, state: AgentState):
        """Applies changes and pushes."""
        print("--- GIT MANAGER AGENT ---")
        
        # 1. Create Branch
        clean_req_name = "".join(c for c in state['requirement'] if c.isalnum() or c == ' ')[:20].replace(' ', '-')
        branch_name = f"feature/ai-{clean_req_name}"
        self.tools.create_branch(branch_name)
        
        # 2. Write Code Changes
        for path, content in state['code_changes'].items():
            self.tools.write_file(path, content)
            
        # 3. Write Tests
        for path, content in state['test_code'].items():
            self.tools.write_file(path, content)
            
        # 4. Commit
        result = self.tools.commit_and_push(f"feat: AI Implementation of '{state['requirement']}'")
        
        return {"history": [f"Git operations completed: {result}"]}

    def reviewer_agent(self, state: AgentState):
        """Checks if the code looks safe (no lazy deletions)."""
        print("--- REVIEWER AGENT ---")
        
        errors = []
        for path, new_content in state['code_changes'].items():
            original_content = self.tools.read_file(path)
            
            # Check 1: Lazy Placeholders
            if "# ..." in new_content or "// ..." in new_content:
                errors.append(f"File {path} contains lazy placeholders.")
                
            # Check 2: Massive Deletion (Code shrank by > 30%)
            if len(original_content) > 0 and len(new_content) < len(original_content) * 0.7:
                errors.append(f"File {path} shrank significantly. Possible code deletion.")

        if errors:
            return {
                "history": [f"Review failed: {'; '.join(errors)}"], 
                "review_status": "failed"
            }
        else:
            return {"history": ["Code review passed."], "review_status": "passed"}
# --- 3. Build the Graph ---

def build_agent_graph(logical_name: str, repo_path: str):
    team = AutonomousDevTeam(logical_name, repo_path)
    
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("planner", team.planner_agent)
    workflow.add_node("researcher", team.researcher_agent)
    workflow.add_node("coder", team.coder_agent)
    workflow.add_node("tester", team.tester_agent)
    workflow.add_node("git_manager", team.git_manager_agent)
    workflow.add_node("reviewer", team.reviewer_agent)
    # Add Edges (Linear flow for now)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "coder")
    workflow.add_edge("coder", "tester")
    workflow.add_edge("tester", "git_manager")
    workflow.add_edge("git_manager", END)
    # Conditional Edge
    def should_retry(state):
        if state.get("review_status") == "failed":
            return "coder"
        return "tester"

    workflow.add_edge("coder", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        should_retry,
        {
            "coder": "coder",
            "tester": "tester"
        }
    )
    return workflow.compile()