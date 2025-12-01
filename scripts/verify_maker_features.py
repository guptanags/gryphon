import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json

# Add backend and backend/app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/app')))

from app.agent_graph import AutonomousDevTeam, CODER_RESPONSE_SCHEMA

class TestMakerFeatures(unittest.TestCase):

    def setUp(self):
        # Mock dependencies for AutonomousDevTeam
        with patch('app.agent_graph.setup_qdrant') as mock_qdrant, \
             patch('app.agent_graph.setup_vertex_ai') as mock_vertex, \
             patch('app.agent_graph.AgentTools') as mock_tools_cls:
            
            mock_vertex.return_value = (MagicMock(), MagicMock())
            
            # Instantiate Agent
            self.agent = AutonomousDevTeam("test_agent", "/tmp")
            
            # Mock tools methods used in planner
            self.agent.tools.list_files.return_value = ["README.md", "app.py"]
            self.agent.tools.generate_repo_skeleton.return_value = "Repo Structure..."
            self.agent.tools.read_file.return_value = "Existing Code..."

    @patch('app.agent_graph.requests.post')
    @patch('app.agent_graph.get_access_token', return_value="fake_token")
    def test_best_of_n_planning(self, mock_get_token, mock_post):
        """
        Verifies that planner_agent generates 3 plans and selects the best one.
        """
        print("\n--- Testing Best-of-N Planning ---")
        
        # Mock State
        state = {
            "requirement": "Build a login page", # Changed from user_request to requirement
            "context": "Existing auth system",
            "plan": "",
            "code_files": [],
            "review_feedback": "",
            "iterations": 0
        }

        # Mock Responses
        # 1. Plan Generation (returns 3 plans)
        plan_1 = "Plan 1: Use Flask"
        plan_2 = "Plan 2: Use Django"
        plan_3 = "Plan 3: Use FastAPI"
        
        def create_plan_response(text):
            return MagicMock(json=MagicMock(return_value={
                "candidates": [{"content": {"parts": [{"text": text}]}}]
            }), raise_for_status=MagicMock())

        mock_post.side_effect = [
            create_plan_response(plan_1),
            create_plan_response(plan_2),
            create_plan_response(plan_3),
            create_plan_response("Plan 3") # Judge response
        ]

        # Run Agent
        result = self.agent.planner_agent(state)

        # Verify
        print(f"Selected Plan: {result['plan']}")
        self.assertEqual(result['plan'], "Plan 3")
        self.assertEqual(mock_post.call_count, 4)
        print("✅ Best-of-N Planning Verified")

    @patch('app.agent_graph.requests.post')
    @patch('app.agent_graph.get_access_token', return_value="fake_token")
    def test_red_flagging_retry(self, mock_get_token, mock_post):
        """
        Verifies that coder_agent retries when it generates lazy code.
        """
        print("\n--- Testing Enhanced Red-Flagging ---")

        # Mock State
        state = {
            "requirement": "Build a login page",
            "plan": "Use Flask",
            "context": "",
            "relevant_files": ["app.py"],
            "code_files": [],
            "review_feedback": "",
            "iterations": 0,
            "language": "python"
        }

        # Mock Responses
        
        # 1. Lazy Response (Red Flag)
        lazy_json = {
            "thought_process": "I will implement this later.",
            "files": [
                {
                    "filepath": "app.py",
                    "content": "# TODO: implement login", # Lazy content
                    "action": "create"
                }
            ]
        }
        lazy_response = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps(lazy_json)
                    }]
                }
            }]
        }

        # 2. Valid Response (Success)
        valid_json = {
            "thought_process": "Implementing login with Flask.",
            "files": [
                {
                    "filepath": "app.py",
                    "content": "from flask import Flask\napp = Flask(__name__)",
                    "action": "create"
                }
            ]
        }
        valid_response = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps(valid_json)
                    }]
                }
            }]
        }

        # Configure mock
        mock_post.side_effect = [
            MagicMock(json=MagicMock(return_value=lazy_response), raise_for_status=MagicMock()),
            MagicMock(json=MagicMock(return_value=valid_response), raise_for_status=MagicMock())
        ]

        # Run Agent
        result = self.agent.coder_agent(state)

        # Verify
        self.assertEqual(len(result['code_changes']), 1)
        self.assertIn("from flask import Flask", result['code_changes']['app.py'])
        self.assertEqual(mock_post.call_count, 2) # Should have retried once
        print("✅ Red-Flagging Retry Verified")

if __name__ == '__main__':
    unittest.main()
