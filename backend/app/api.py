import os
import uvicorn
# We no longer need threading
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends # Add Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import threading
from agent_graph import build_agent_graph, AgentState
# --- NEW: SQLAlchemy Imports ---
from sqlalchemy.orm import Session
import models
from database import SessionLocal, engine, get_db

# --- Import from our pipeline file ---
from pipeline import (
    run_ingestion_pipeline, 
    run_test_generation_pipeline, 
    setup_qdrant, # Rename to avoid conflict
    setup_vertex_ai
)
from vertexai.generative_models import GenerativeModel, Part
from celery_worker import task_ingest_repo

models.Base.metadata.create_all(bind=engine)

def get_db_session_for_task():
    """Helper to create a new, independent DB session for background tasks."""
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

# --- API Setup ---
app = FastAPI(
    title="Grypon AI API",
    description="API for ingesting and querying codebases."
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- In-Memory DB for Deduplication and Status ---
# This is our simple, local "in-memory database"
# We use a lock to make it thread-safe for background tasks
DB_LOCK = threading.Lock()
DB: Dict[str, Any] = {
    "ingested_repos": {} # K: logical_name, V: status dict
}
PENDING_TASKS = set() # A set of logical_names currently being processed
TEST_TASKS_IN_PROGRESS = set() # a set of logical_names for TESTING
AGENT_LIVE_STATUS = {}

# --- Pydantic Models for API ---
class IngestRequest(BaseModel):
    logical_name: str = Field(..., description="A unique name for this data source, e.g., 'my-project-backend'")
    git_repos: List[str] = Field(default_factory=list, description="List of 'url,branch' strings. e.g., ['[https://github.com/foo/bar.git,main](https://github.com/foo/bar.git,main)']")
    confluence_pages: List[str] = Field(default_factory=list, description="List of Confluence page URLs")

class IngestResponse(BaseModel):
    message: str
    logical_name: str
    status_url: str

class RepositoryStatus(BaseModel):
    logical_name: str
    repo_url: str
    branch_name: str
    status: str
    test_status: str
    last_ingested_at: datetime | None
    code_coverage_score: float | None
    code_quality_score: float | None

class RepositoryList(BaseModel):
    repositories: List[RepositoryStatus]

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    code_context: str
    doc_context: str

class TestGenRequest(BaseModel):
    logical_name: str = Field(..., description="The unique name of the ingested repository")
    test_types: List[str] = Field(..., description="List of tests to generate, e.g., ['unit', 'acceptance', 'load']")

class TestGenResponse(BaseModel):
    message: str
    logical_name: str
    status_url: str

class MetricsUpdate(BaseModel):
    repo_url: str
    branch_name: str
    coverage_score: float = Field(..., ge=0, le=100, description="Code coverage percentage (0-100)")

class AutonomousTaskRequest(BaseModel):
    logical_name: str
    requirement: str

class AgentStatusResponse(BaseModel):
    logical_name: str
    status: str
    current_step: str
    logs: List[str]

def background_ingest_task(db_id: int, req: IngestRequest):
    """The wrapper function that BackgroundTasks will call."""
    
    db_gen = get_db_session_for_task()
    db = next(db_gen)
    
    try:
        # Get the DB object
        repo_branch = db.query(models.RepositoryBranch).filter(models.RepositoryBranch.id == db_id).first()
        if not repo_branch:
            print(f"ERROR: Background task for ID {db_id} found no object.")
            return

        # 1. Set status to "ingesting"
        repo_branch.status = "ingesting"
        db.commit()

        # 2. Run the heavy pipeline
        avg_score, static_analysis_score = run_ingestion_pipeline(
            logical_name=req.logical_name,
            git_repo_list=req.git_repos,
            confluence_page_list=req.confluence_pages
        )
        
        # 3. On success, update the DB
        repo_branch.status = "completed"
        repo_branch.last_ingested_at = datetime.now()
        if avg_score is not None:
            repo_branch.code_quality_score = avg_score
        if static_analysis_score is not None:
            repo_branch.code_static_analysis_score = static_analysis_score
        db.commit()
        
    except Exception as e:
        # 4. On failure, update the DB with the error
        print(f"ERROR: Background ingestion failed for {req.logical_name}: {e}")
        if db.is_active:
            repo_branch.status = "failed"
            # We could add an 'error_message' column to store str(e)
            db.commit()
    finally:
        db.close()

def background_test_task(db_id: int, req: TestGenRequest, repo_urls: list):
    """The wrapper function for the test generation background task."""
    
    db_gen = get_db_session_for_task()
    db = next(db_gen)
    
    try:
        # Get the DB object
        repo_branch = db.query(models.RepositoryBranch).filter(models.RepositoryBranch.id == db_id).first()
        if not repo_branch:
            print(f"ERROR: Background task for ID {db_id} found no object.")
            return
            
        # 1. Set status to "pending" (already set in endpoint, but good to be sure)
        repo_branch.test_status = "pending"
        db.commit()

        # 2. Run the heavy pipeline AND GET THE SCORE
        coverage_score = run_test_generation_pipeline(
            logical_name=req.logical_name,
            test_types=req.test_types,
            repo_urls=repo_urls
        )
        
        # 3. On success, update the DB
        repo_branch.test_status = "completed"
        if coverage_score is not None:
            repo_branch.code_coverage_score = coverage_score # <-- SAVE THE SCORE
        db.commit()
        
    except Exception as e:
        # 4. On failure, update the DB with the error
        print(f"ERROR: Background test generation failed for {req.logical_name}: {e}")
        if db.is_active:
            repo_branch.test_status = "failed"
            db.commit()
    finally:
        db.close()


# --- API Endpoints ---

@app.post("/ingest", response_model=IngestResponse, status_code=202)
def ingest_sources(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Queue a new ingestion task. Takes a logical name, git repos,
    and confluence pages. Runs in the background.
    """
    # We will create one DB entry PER REPO in the request
    # For this simple model, let's assume one repo per logical_name
    # A more complex model would handle multiple repos under one logical_name
    
    if not req.git_repos:
        raise HTTPException(status_code=400, detail="No git_repos provided.")

    # We'll use the first repo for this example
    try:
        repo_url, branch_name = req.git_repos[0].strip().split(',')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid git_repo format. Must be 'url,branch'.")

    # Check if this exact logical_name + repo + branch is already processing
    existing = db.query(models.RepositoryBranch).filter(
        models.RepositoryBranch.logical_name == req.logical_name,
        models.RepositoryBranch.repo_url == repo_url,
        models.RepositoryBranch.branch_name == branch_name
    ).first()

    if existing and existing.status in ["pending", "ingesting"]:
        raise HTTPException(status_code=409, detail=f"Ingestion for '{req.logical_name}' on branch '{branch_name}' is already in progress.")

    if existing and existing.status == "completed":
        # If it's already done, we'll re-run it
        print(f"Re-ingesting {req.logical_name} on branch {branch_name}...")
        db_object = existing
        db_object.status = "pending"
    else:
        # Create new DB entry
        db_object = models.RepositoryBranch(
            logical_name=req.logical_name,
            repo_url=repo_url,
            branch_name=branch_name,
            status="pending",
            test_status="n/a"
        )
        db.add(db_object)
    
    db.commit()
    db.refresh(db_object) # Get the ID of the new object

    # Add the heavy work to the background queue, passing the new DB ID
    task_ingest_repo.delay(db_object.id, req.logical_name, req.git_repos, req.confluence_pages)

    return IngestResponse(
        message="Ingestion task queued. Processing in background.",
        logical_name=req.logical_name,
        status_url="/repositories"
    )


@app.get("/repositories", response_model=RepositoryList)
def get_repositories(db: Session = Depends(get_db)):
    """
    Get the status of all ingested repositories from the database.
    """
    repos = db.query(models.RepositoryBranch).all()
    
    # Format the response
    repo_list = [
        RepositoryStatus(
            logical_name=repo.logical_name,
            repo_url=repo.repo_url,
            branch_name=repo.branch_name,
            status=repo.status,
            test_status=repo.test_status,
            last_ingested_at=repo.last_ingested_at,
            code_coverage_score=repo.code_coverage_score,
            code_quality_score=repo.code_quality_score
        ) for repo in repos
    ]
    
    return RepositoryList(repositories=repo_list)

@app.post("/generate-tests", response_model=TestGenResponse, status_code=202)
def generate_tests(
    req: TestGenRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Queue a new test generation task. Runs in the background.
    """
    # Find the repo branch to test
    # Note: This finds the *first* match for the logical name.
    repo_branch = db.query(models.RepositoryBranch).filter(
        models.RepositoryBranch.logical_name == req.logical_name
    ).first()

    if not repo_branch:
        raise HTTPException(status_code=44, detail=f"Logical name '{req.logical_name}' not found. Please ingest it first.")
    
    if repo_branch.status != "completed":
        raise HTTPException(status_code=400, detail=f"Ingestion for '{req.logical_name}' is not completed.")
    
    if repo_branch.test_status in ["pending", "generating"]:
        raise HTTPException(status_code=409, detail=f"Test generation for '{req.logical_name}' is already in progress.")

    # Set status to pending
    repo_branch.test_status = "pending"
    db.commit()

    # Get the repo URLs needed for the task
    repo_urls = [f"{repo_branch.repo_url},{repo_branch.branch_name}"]

    background_tasks.add_task(background_test_task, repo_branch.id, req, repo_urls)
    
    return TestGenResponse(
        message="Test generation task queued. Processing in background.",
        logical_name=req.logical_name,
        status_url="/repositories"
    )
    
 


# --- Q&A Module (Module 4) ---
# We initialize these clients *once* on startup for the /query endpoint
# The background tasks will initialize their own clients
print("Initializing Q&A clients...")
qdrant_client = setup_qdrant()
embedding_model, generative_model = setup_vertex_ai()

if not qdrant_client or not embedding_model or not generative_model:
    print("FATAL: Could not initialize Q&A clients. /query endpoint will fail.")
    # We don't exit, as ingestion might still work
else:
    print("Q&A clients initialized successfully. API is ready.")

@app.post("/query", response_model=QueryResponse)
def query_codebase(request: QueryRequest):
    """
    Takes a question, embeds it, searches Qdrant for context,
    and asks Gemini to generate an answer.
    """
    if not qdrant_client or not embedding_model or not generative_model:
        raise HTTPException(status_code=503, detail="Q&A service is not ready. Clients failed to initialize.")
        
    try:
        # 1. Embed question
        question_embedding = embedding_model.get_embeddings([request.question])[0].values

        # 2. Search collections
        code_results = qdrant_client.search(
            collection_name="code",
            query_vector=question_embedding,
            limit=request.top_k,
            with_payload=True
        )
        doc_results = qdrant_client.search(
            collection_name="docs",
            query_vector=question_embedding,
            limit=request.top_k,
            with_payload=True
        )

        # 3. Build context
        code_context = "\n--- RELEVANT CODE SNIPPETS ---\n"
        for result in code_results:
            payload = result.payload
            code_context += f"Snippet (from {payload['file_path']}, lines {payload['start_line']}-{payload['end_line']}):\n"
            code_context += "```\n" + payload['content'] + "\n```\n\n"

        doc_context = "\n--- RELEVANT DOCUMENTATION ---\n"
        for result in doc_results:
            payload = result.payload
            doc_context += f"Doc Snippet (from {payload.get('source_url') or payload.get('source_file_path')}):\n"
            doc_context += payload['content'] + "\n\n"

        # 4. Build prompt
        prompt = f"""
You are an expert pair programmer. Answer the user's question based on the code and documentation provided.
- Answer clearly and concisely.
- **Use the provided code snippets in your answer.**
- If context is missing, say "I could not find an answer in the provided context."

**QUESTION:**
{request.question}

{code_context}

{doc_context}

**ANSWER:**
"""
        
        # 5. Call Gemini
        response = generative_model.generate_content(prompt)

        return QueryResponse(
            answer=response.text,
            code_context=code_context,
            doc_context=doc_context
        )
    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/metrics/coverage", status_code=200)
def update_coverage_metrics(
    metric: MetricsUpdate,
    db: Session = Depends(get_db)
):
    """
    Endpoint for external CI/CD pipelines to report code coverage scores.
    """
    # Find the repo entry using the composite key
    repo_branch = db.query(models.RepositoryBranch).filter(
        models.RepositoryBranch.repo_url == metric.repo_url,
        models.RepositoryBranch.branch_name == metric.branch_name
    ).first()

    if not repo_branch:
        raise HTTPException(
            status_code=404, 
            detail=f"Repository not found for URL '{metric.repo_url}' and branch '{metric.branch_name}'"
        )

    # Update the score
    repo_branch.code_coverage_score = metric.coverage_score
    db.commit()

    return {"message": "Coverage score updated successfully", "logical_name": repo_branch.logical_name}


@app.post("/agent/run", status_code=202)
def implement_feature(req: AutonomousTaskRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Spins up an Autonomous Agent Team to implement a requirement.
    """
    # 1. Find the Repo
    repo_branch = db.query(models.RepositoryBranch).filter(
        models.RepositoryBranch.logical_name == req.logical_name
    ).first()
    
    if not repo_branch:
        raise HTTPException(status_code=404, detail="Repo not found.")

    # 2. Build the Agent Graph
    def _run_agent_team():
        # Setup Status
        AGENT_LIVE_STATUS[req.logical_name] = {
            "status": "starting",
            "logs": ["Initializing Agent Team..."],
            "current_step": "init"
        }

        try:
            work_dir = f"/tmp/ai_workspaces/{req.logical_name}"
            if os.path.exists(work_dir):
                import shutil
                shutil.rmtree(work_dir)
                
            from pipeline import clone_repo
            
            AGENT_LIVE_STATUS[req.logical_name]["logs"].append(f"Cloning {repo_branch.repo_url}...")
            repo_path = clone_repo(repo_branch.repo_url, repo_branch.branch_name) 
            
            # Initialize Graph
            app = build_agent_graph(req.logical_name, repo_path)
            
            initial_state = {
                "logical_name": req.logical_name,
                "repo_path": repo_path,
                "requirement": req.requirement,
                "history": []
            }
            
            AGENT_LIVE_STATUS[req.logical_name]["status"] = "running"
            
            # --- THE STREAMING MAGIC ---
            # app.stream yields the output of each node as it finishes
            for output in app.stream(initial_state):
                # 'output' looks like: {'planner': {'plan': '...', 'history': [...]}}
                
                for node_name, state_update in output.items():
                    # Update global status
                    AGENT_LIVE_STATUS[req.logical_name]["current_step"] = node_name
                    
                    # Log the history update
                    if "history" in state_update:
                        latest_log = state_update["history"][-1]
                        AGENT_LIVE_STATUS[req.logical_name]["logs"].append(f"[{node_name.upper()}]: {latest_log}")
                        print(f"[{req.logical_name}] {node_name}: {latest_log}")

            # Final Success Update
            AGENT_LIVE_STATUS[req.logical_name]["status"] = "completed"
            AGENT_LIVE_STATUS[req.logical_name]["logs"].append("Workflow Finished Successfully.")
            
        except Exception as e:
            AGENT_LIVE_STATUS[req.logical_name]["status"] = "failed"
            AGENT_LIVE_STATUS[req.logical_name]["logs"].append(f"FATAL ERROR: {str(e)}")
            print(f"Agent Error: {e}")

    # 3. Queue Task
    background_tasks.add_task(_run_agent_team)
    
    return {"message": "Autonomous agent team started.", "logical_name": req.logical_name}

@app.get("/agent/status/{logical_name}", response_model=AgentStatusResponse)
def get_agent_status(logical_name: str):
    """
    Returns the real-time logs and status of the autonomous agent.
    """
    if logical_name not in AGENT_LIVE_STATUS:
        # If not in memory, check if we have a persistent record (optional enhancement)
        # For now, return a generic "not found" or "idle"
        return AgentStatusResponse(
            logical_name=logical_name,
            status="idle",
            current_step="none",
            logs=[]
        )
    
    data = AGENT_LIVE_STATUS[logical_name]
    return AgentStatusResponse(
        logical_name=logical_name,
        status=data["status"],
        current_step=data.get("current_step", "unknown"),
        logs=data["logs"]
    )
# --- Run the API ---
if __name__ == "__main__":
    print("Running API server locally on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)