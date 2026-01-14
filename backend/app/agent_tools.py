# agent_tools.py
import os
import shutil
from git import Repo
import logging
import subprocess

log = logging.getLogger(__name__)

class AgentTools:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.repo = Repo(repo_path)

    def read_file(self, file_path: str) -> str:
        """Reads a file from the repository."""
        full_path = os.path.join(self.repo_path, file_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, file_path: str, content: str):
        """Writes content to a file (overwrites or creates)."""
        full_path = os.path.join(self.repo_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        log.info(f"Wrote to file: {file_path}")

    def create_branch(self, branch_name: str):
        """Creates and checks out a new feature branch."""
        try:
            current = self.repo.active_branch
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            log.info(f"Checked out new branch: {branch_name}")
            return f"Success: On branch {branch_name}"
        except Exception as e:
            return f"Error creating branch: {str(e)}"

    def commit_and_push(self, message: str):
        """Stages all changes, commits, and pushes to origin."""
        try:
            # 1. Add all changes
            self.repo.git.add(A=True)
            
            # 2. Commit
            self.repo.index.commit(message)
            
            # 3. Push
            # We explicitly push the current branch to the 'origin' remote
            current_branch = self.repo.active_branch.name
            origin = self.repo.remote(name='origin')
            
            # Push and set upstream
            push_info = origin.push(refspec=f'{current_branch}:{current_branch}')
            
            # Check for errors in the push result
            if push_info[0].flags & push_info[0].ERROR:
                return f"Error pushing: {push_info[0].summary}"
                
            return f"Success: Committed and Pushed to {current_branch}"
            
        except Exception as e:
            return f"Error during git op: {str(e)}"
            
    def list_files(self):
        """Lists all files in the repo to help the agent orient itself."""
        files_list = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if ".git" not in root:
                    files_list.append(os.path.relpath(os.path.join(root, file), self.repo_path))
        return files_list
    
    def append_to_file(self, file_path: str, content: str):
        """Appends content to the end of a file."""
        full_path = os.path.join(self.repo_path, file_path)
        with open(full_path, "r", encoding="utf-8") as f:
            original = f.read()
        
        # Ensure we start on a new line
        if not original.endswith("\n"):
            content = "\n" + content
            
        with open(full_path, "a", encoding="utf-8") as f:
            f.write(content)
        log.info(f"Appended to file: {file_path}")

    def generate_repo_skeleton(self, max_depth: int = 3) -> str:
        """
        Generates a tree-like string representation of the repository structure.
        Respects common ignore patterns.
        """
        skeleton = []
        ignore_dirs = {'.git', '__pycache__', 'node_modules', 'venv', '.env', '.DS_Store'}
        
        for root, dirs, files in os.walk(self.repo_path):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            level = root.replace(self.repo_path, '').count(os.sep)
            if level > max_depth:
                continue
                
            indent = ' ' * 4 * (level)
            skeleton.append(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                if f not in ignore_dirs:
                    skeleton.append(f"{subindent}{f}")
                    
        return "\n".join(skeleton)

    def read_file_with_lines(self, relative_path: str) -> str:
        """Reads a file and adds line numbers for the LLM's context."""
        content = self.read_file(relative_path)
        if not content or "Error" in content: return content
        
        lines = content.split('\n')
        numbered_lines = [f"{i+1:04d} | {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)

    def apply_patch(self, relative_path: str, search_block: str, replace_block: str) -> str:
        """
        Locates the 'search_block' in the file and replaces it with 'replace_block'.
        Uses fuzzy matching to handle minor whitespace differences.
        """
        full_path = os.path.join(self.repo_path, relative_path)
        if not os.path.exists(full_path):
            return f"Error: File {relative_path} not found."

        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Exact Match Try
        if search_block in content:
            new_content = content.replace(search_block, replace_block, 1) # Replace only first occurrence
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return "Patch applied successfully (Exact match)."

        # 2. Relaxed Whitespace Match (Handling indentation differences)
        # Normalize: strip leading/trailing whitespace from every line
        search_lines = [l.strip() for l in search_block.split('\n') if l.strip()]
        file_lines = content.split('\n')
        
        # Simple sliding window search
        start_idx = -1
        match_count = 0
        
        for i, line in enumerate(file_lines):
            if match_count < len(search_lines):
                if search_lines[match_count] in line: # Relaxed check
                    if match_count == 0: start_idx = i
                    match_count += 1
                else:
                    match_count = 0 # Reset if sequence breaks
                    start_idx = -1
            
            if match_count == len(search_lines):
                # Found the block! Replace lines from start_idx to i
                # Note: This is a naive replacement, distinct from strict diffs, but works for LLMs
                # A safer production way is to use 'ed' scripts or 'git apply', but this is python-native.
                
                # Construct new content
                before = file_lines[:start_idx]
                after = file_lines[i+1:]
                
                # Determine indentation of the start line to respect existing style
                original_indent = file_lines[start_idx][:len(file_lines[start_idx]) - len(file_lines[start_idx].lstrip())]
                
                # Apply indentation to replacement block
                replace_lines = replace_block.split('\n')
                indented_replace = [original_indent + l for l in replace_lines]
                
                new_content_lines = before + indented_replace + after
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(new_content_lines))
                return "Patch applied successfully (Fuzzy match)."

        return "Error: Could not locate the 'search_block' in the file. Patch failed."

    def run_unit_tests(self, language: str) -> dict:
        """
        Executes the unit tests for the specific language and returns the results.
        """
        try:
            if language == "python":
                # Run pytest on the repo
                result = subprocess.run(
                    ["pytest", "--json-report", "--json-report-file=report.json", self.repo_path],
                    capture_output=True, text=True, timeout=30
                )
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout if result.returncode == 0 else result.stderr
                }
            
            elif language == "java":
                # Run Maven Test
                result = subprocess.run(
                    ["mvn", "test", "-DfailIfNoTests=false"],
                    cwd=self.repo_path, capture_output=True, text=True, timeout=60
                )
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout if result.returncode == 0 else result.stderr
                }
            
            return {"success": False, "output": "Unsupported language for execution."}
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "Tests timed out (Possible infinite loop)."}
        except Exception as e:
            return {"success": False, "output": str(e)}

    def find_style_example(self, target_file: str) -> str:
        """
        Finds a 'peer' file to act as a coding style guide.
        Priority: 1. Same directory, 2. Same suffix (e.g., Service.java)
        """
        directory = os.path.dirname(target_file)
        target_name = os.path.basename(target_file)
        suffix = target_name.split('.')[-1]
        
        # 1. Try to find another file in the same directory
        peers = [f for f in os.listdir(os.path.join(self.repo_path, directory)) 
                 if f.endswith(suffix) and f != target_name]
        
        if peers:
            return os.path.join(directory, peers[0])
            
        # 2. Fallback: Search the whole repo for a file with the same suffix
        # (This would be better served by a Qdrant search in production)
        return None