import os
import yaml
from jinja2 import Template

class PromptManager:
    def __init__(self, prompt_file="prompts/agents.yaml"):
        # Ensure path is absolute or relative to execution root
        self.prompt_file = prompt_file
        self.templates = self._load_templates()

    def _load_templates(self):
        """Loads YAML and pre-compiles Jinja2 templates."""
        if not os.path.exists(self.prompt_file):
            # Fallback or error - strictly raising error here to ensure config exists
            raise FileNotFoundError(f"Prompt file not found at: {self.prompt_file}")
            
        with open(self.prompt_file, 'r') as f:
            raw_prompts = yaml.safe_load(f)
            
        # Convert string prompts into Jinja2 Template objects
        templates = {}
        for agent, prompts in raw_prompts.items():
            templates[agent] = {}
            for key, prompt_text in prompts.items():
                templates[agent][key] = Template(prompt_text)
        return templates

    def render(self, agent_name: str, prompt_type: str, **kwargs) -> str:
        """
        Renders a prompt with the given variables.
        Usage: pm.render('planner', 'user', requirement='...')
        """
        try:
            template = self.templates.get(agent_name, {}).get(prompt_type)
            if not template:
                return f"Error: Template for {agent_name}.{prompt_type} not found."
            return template.render(**kwargs)
        except Exception as e:
            return f"Error rendering prompt: {str(e)}"

# Singleton Instance
prompt_manager = PromptManager()