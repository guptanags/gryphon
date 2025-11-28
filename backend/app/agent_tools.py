# agent_tools.py
import os
import shutil
from git import Repo
import logging

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

    