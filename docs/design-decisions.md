Gryphon Backend Technical Architecture
## 1. Overview
The Gryphon backend is an autonomous coding agent system designed to plan, implement, review, and test software changes. It leverages Agentic AI, RAG (Retrieval-Augmented Generation), and Reliability Engineering patterns to deliver high-quality code.

## 2. Core Architecture: The Agent Graph
The system is built on LangGraph, defining a state machine where nodes represent specialized agents and edges represent the workflow.

### 2.1 Agent Roles
Planner (Pragmatic Architect): Analyzes requirements and the repository skeleton. Uses Best-of-N Planning (generates 3 plans, selects the best) to ensure a solid technical direction.
Researcher: Expands context by retrieving relevant files. Uses Dependency Graph Expansion (forward/reverse imports) and Test-Driven Discovery (finding tests relevant to the requirement).
Coder (Senior Engineer): Implements the plan. Features Enhanced Red-Flagging (internal retry loop) to catch "lazy coding" (placeholders) and invalid JSON before outputting.
Syntax Checker: Validates code syntax (AST for Python, heuristics for others).
Reviewer (Security Auditor): A two-stage gatekeeper:
Heuristics: Checks for forbidden patterns (e.g., TODO, significant size reduction).
LLM Review: A "Strict Security Auditor" persona checks for logic bugs and security vulnerabilities.
Tester (SDET): Generates unit tests (Pytest/JUnit) for the new code.
Git Manager: Handles branching, committing, and pushing changes.
### 2.2 Workflow Control
Gates: Conditional edges determine the path (e.g., Syntax Checker -> Failed -> Coder).
Circuit Breakers: Limits the number of iterations (e.g., max 5 coding attempts) to prevent infinite loops.
## 3. Data Ingestion & RAG Pipeline
The system maintains a semantic understanding of the codebase using Qdrant (Vector DB) and Vertex AI (Embeddings).

### 3.1 Ingestion Pipeline (
pipeline.py
)
Parsing: Uses Tree-sitter to parse code into chunks (classes/functions).
Metadata Extraction:
Signatures: Extracts function/class signatures for precise API context.
Docstrings: Captures intent and usage instructions.
Complexity: Calculates Cyclomatic Complexity to warn agents about fragile code.
File Summaries: Generates high-level summaries for entire files, stored in a dedicated file_summaries collection.
Storage:
code
: Code chunks with metadata.
docs: Documentation and text files.
file_summaries: High-level file descriptions.
### 3.2 Context Construction
Structured Context: Code is presented to the LLM using XML-like tags (<file path="...">...</file>) to clearly delimit file boundaries and metadata.
## 4. Reliability, Accuracy & Determinism
We employ a multi-layered approach to ensure the system produces high-quality, working code.

### 4.1 Reliability (System Stability)
Circuit Breakers: The iterations counter prevents infinite loops between Coder and Reviewer. If the agent gets stuck (e.g., >5 attempts), it fails gracefully.
Syntax Validation: The Syntax Checker parses the code (AST for Python) to catch syntax errors immediately, preventing invalid code from reaching the Reviewer.
Enhanced Red-Flagging (Fast Fail): The Coder Agent runs an internal retry loop. If it detects "lazy" placeholders (e.g., # ...) or invalid JSON, it retries immediately without propagating the error.
### 4.2 Accuracy (Correctness)
Best-of-N Planning: The Planner generates 3 distinct plans and uses an LLM "Judge" to select the best one. This reduces the risk of hallucinated files or poor architectural choices.
RAG & Metadata: By extracting signatures and docstrings, we provide the LLM with precise API contracts, reducing hallucinated method calls.
Two-Stage Review:
Heuristic: Checks for code size reduction and forbidden patterns.
LLM Review: A "Strict Security Auditor" checks for logic bugs and security flaws.
### 4.3 Determinism (Consistency)
Structured Output (JSON Mode): We enforce JSON schemas for all agent outputs, ensuring the response structure is predictable and machine-readable.
Structured Context: We use XML-like tags (<file>) to delimit context, preventing the LLM from confusing prompt instructions with code content.
Temperature Control:
Coding: Low temperature (0.1) for precise, deterministic code generation.
Planning: Higher temperature (0.7) for diverse idea generation during the voting phase.
Chain-of-Thought (CoT): Prompts explicitly ask for "Step-by-Step" reasoning, which grounds the LLM's logic before it generates code.
## 5. Technology Stack
Orchestration: LangGraph, Python
LLM: Google Vertex AI (Gemini 1.5 Pro/Flash)
Vector DB: Qdrant
Parsing: Tree-sitter
VCS: GitPython

## 6. Design Decisions

Here is the updated summary of the key **AI Architectural Paradigms** in Gryphon, now including the advanced decision-making, quality assurance, and processing patterns:

### 6.1 Core Agentic Framework
* **Stateful Graph Architecture (LangGraph):** Uses a **Cyclic State Graph** to allow agents to maintain memory, pass artifacts, and loop back for corrections (e.g., Coder $\to$ Syntax Error $\to$ Coder).
* **Role-Based Delegation:** Decomposed the "Software Engineer" into specialized nodes: **Planner**, **Researcher**, **Coder**, **Reviewer**, and **Git Manager**.

### 6.2 Decision Making & "Human-in-the-Loop"
* **Multi-Plan Generation:** The Planner is capable of generating **multiple valid implementation strategies** (e.g., "Plan A: Modify existing Service" vs. "Plan B: Create new Strategy Pattern").
* **Selection Step:** The workflow supports a "pause" state where a human (or a Senior Architect Agent) reviews and **selects the best plan** before any code is written, preventing wasted effort on bad approaches.

### 6.3 "Critic-Actor" Quality Loops
* **Reflexive Self-Correction:** The system doesn't trust the LLM's first draft.
    * **Syntax Guardrail:** A deterministic Python/Java syntax checker validates compilation/parsing.
    * **Lazy Code Detector:** A heuristic scanner ensures the LLM didn't return placeholders (e.g., `# ... rest of code`) instead of the full file.
* **Static Analysis Integration:** The pipeline runs industrial-grade tools (**Radon** for Python, **PMD** for Java) *during* the generation loop. If the "Code Quality Score" drops below a threshold, the agent is forced to refactor.

### 6.4 Asynchronous & Event-Driven Processing
* **Decoupled Execution (Celery + Redis):** We moved from fragile in-memory background tasks to a robust **Producer-Consumer** model.
    * The **API** (Producer) pushes a "Job" to Redis and returns immediately (Non-blocking).
    * The **Worker Pods** (Consumers) pick up jobs and execute the heavy LangGraph workflows independently.
* **Resiliency:** If a worker pod crashes, the job remains in the queue and is retried by another pod, ensuring zero data loss.

### 6.5 Advanced RAG (Retrieval-Augmented Generation)
* **Graph-Augmented Retrieval:** Implemented **Dependency Linking** (AST parsing) and **"Go To Definition"** tools so agents find code based on logical relationships (imports/calls), not just keywords.
* **Hierarchical Context:**
    * **Skeleton View:** The Planner sees a lightweight "Tree View" of the entire architecture.
    * **Deep View:** The Coder sees the full content of only the specific relevant files.
* **Test-Driven Discovery:** The Researcher looks for existing tests first to infer production code dependencies.

### 6.6 LLM Interaction & Determinism
* **REST-based JSON Enforcement:** Bypassed standard SDKs to use raw REST calls with `responseSchema`. This forces the LLM to output strictly structured JSON, solving parsing errors.
* **Prompt Engineering via Template Engine:** Externalized prompts using Jinja2/YAML templates to separate prompt logic from application code.


### 6.7 Context Verification & Refactoring


#### 6.7.1 Hierarchical Retrieval (The "Zoom-In" Strategy)
The Problem: Standard RAG treats a 10-line helper function the same as a main Controller class. In a large repo, a search for "User" returns 1,000 results. The Solution: Search in two stages. First find the Right File, then find the Right Lines.

Step A (File Level): Create a separate Qdrant collection called file_summaries. During ingestion, ask the LLM to summarize each file (e.g., "This file handles User Authentication logic").

Step B (Chunk Level): When a query comes in, first search file_summaries to find the top 5 relevant files.

Step C (Scoped Search): Then, perform the dense vector search ONLY within those 5 files.

#### 6.7.2 Cross-Encoder Reranking (The "Relevance Judge")
The Problem: Vector search (Cosine Similarity) is fast but "fuzzy." It often ranks a file that mentions "login" higher than the actual LoginService.java. The Solution: Add a Reranking Step. Use a highly accurate "Cross-Encoder" model (like BGE-Reranker or Cohere) to re-score the top 50 results from Qdrant and pick the top 5.

Impact: This typically boosts retrieval accuracy by 20-30% in large codebases.

#### 6.7.3 The "Context Verifier" Agent (Pre-Flight Check)
The Problem: The Coder agent sometimes starts coding even if it's missing a critical dependency (e.g., "I don't see the User class definition, so I'll just guess its fields"). The Solution: Insert a Verifier Node before the Coder.

Role: It reads the gathered context and asks: "Do I have the definition for every class/function I need to modify?"

Action: If No, it sends the flow back to the Researcher with a specific instruction: "Go find the definition of UserDTO class."