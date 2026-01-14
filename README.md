## Current Approach: The "Surgical" Agent Squad
Gryphon has moved away from the "delete and rewrite" model. It now follows a high-precision, stateful, and context-aware lifecycle.

### Requirement Refinement (The Analyst): 
Vague user inputs are expanded into detailed technical specifications. It identifies implied tasks (like adding dependencies or updating configs) before the "builders" ever see the request.

### Hierarchical Retrieval (The Zoom-In RAG): 
To handle your large banking repositories, it first searches a specialized file_summaries collection to find the "Right File." Only then does it perform a scoped vector search within those files to find the "Right Lines."

### Dependency Awareness (The Architect): 
By parsing pom.xml and package.json, the Researcher provides a list of available internal libraries. This prevents the Coder from "hallucinating" external dependencies that don't exist in your secure environment.

### Line-Numbered Context & Patching (The Senior Dev): 
We feed the LLM line-numbered code. Instead of returning a whole file, the Coder returns a search_block (exactly matching existing lines) and a replace_block. This ensures the AI "reads" before it "writes."

### Multi-Gate Validation:

**Syntax Gate:** Local compiler checks (Python/Java).

**Reviewer Gate:** A security-focused agent checking for vulnerabilities and business logic alignment.

**Tester Gate:** Automated generation of Unit Tests (JUnit/Pytest) for the new code.
