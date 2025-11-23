# Gryphon AI - Copilot Instructions

## Project Overview
Gryphon AI is a **modular developer productivity platform** with a Python/FastAPI backend and Angular 15 frontend. It provides AI-driven features for code analysis, documentation generation, testing, and autonomous code development.

**Key Value Proposition**: Six interconnected modules providing end-to-end AI assistance for code quality and developer velocity.

## Architecture

### Backend (Python FastAPI)
- **Location**: `backend/app/`
- **Core Entry Point**: `api.py` - FastAPI REST server (port 8000)
- **Database**: SQLite (`app.db`) via SQLAlchemy ORM; migrates to PostgreSQL on OpenShift
- **Key Models**: `models.py` defines `RepositoryBranch` - tracks ingested repos with composite key (logical_name, repo_url, branch_name)

### Frontend (Angular 15)
- **Location**: `ui/src/app/`
- **Routing**: Module-based navigation in `app.module.ts` (8 route-based components)
- **Data Layer**: `data.service.ts` - centralized HTTP client for backend API calls
- **Components**: Dashboard aggregates all modules; sidebar navigation persists across routes

### Data Flow
```
Frontend Component → DataService (HTTP) → FastAPI Endpoint → 
Background Task Queue → SQLAlchemy Session → DB Update
```

**Background Processing**: FastAPI `BackgroundTasks` handles long-running pipelines (ingestion, test generation) asynchronously; endpoints return 202 Accepted immediately.

## Critical Integration Points

### 1. **Ingestion Pipeline** (`POST /ingest`)
- **Request**: `IngestRequest` (logical_name, git_repos[], confluence_pages[])
- **Process**: Creates `RepositoryBranch` DB record → queues `background_ingest_task()`
- **Key File**: `pipeline.py` - orchestrates repo cloning, code parsing (tree-sitter for Python/Java/TypeScript), embedding generation (Vertex AI text-embedding-004), and Qdrant vector storage
- **Status Tracking**: DB columns `status` ("pending" → "ingesting" → "completed" | "failed") and `last_ingested_at`
- **Deduplication**: Queries existing repo entries to prevent duplicate concurrent ingestions

### 2. **Vector Database (Qdrant)**
- **Collections**: 
  - `"code"` - parsed code snippets with metadata (file_path, start_line, end_line)
  - `"docs"` - documentation from URLs and Confluence with source tracking
- **Embedding Model**: Vertex AI `text-embedding-004` (768-dim vectors)
- **Configuration**: Environment variables `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`

### 3. **Test Generation Pipeline** (`POST /generate-tests`)
- **Request**: `TestGenRequest` (logical_name, test_types[])
- **Types Supported**: "unit", "acceptance", "load" (extensible in pipeline.py)
- **Output**: Generated test files stored in `backend/app/generated_tests/`
- **Metrics**: Returns `code_coverage_score` that persists to DB

### 4. **Query Endpoint** (`POST /query`)
- **Q&A Workflow**: Embed question → search Qdrant code+docs collections (top-k results) → context-augmented Gemini prompt
- **Model**: Vertex AI `gemini-2.5-flash-lite`
- **Dependencies**: Requires both embedding_model and generative_model initialized at startup

## Module-Specific Patterns

### **Insight Engine** (Documentation Generation)
- Accepts multi-URL git repos and Confluence pages
- Generates documentation types: Functional, Technical
- **Frontend**: `insight-engine.component.ts` collects URLs and doc type checkboxes

### **Quality Guardian** (Test Execution)
- Generates test cases for multiple types
- Displays pass/fail rates and test suite details
- **Metrics**: Code coverage, quality scores, static analysis scores
- **Frontend**: Visualizes metrics and lists test results

### **Autonomous Co-Pilot** (Code Generation)
- Accepts feature requests in plain English
- Performs impact analysis and generates code
- Automates PR creation with reviewers
- **Frontend**: Multi-step workflow with diff visualization

### **Code Quality Guardian** (Static Analysis)
- Computes code complexity, linting errors, coverage
- Uses `radon` library for complexity metrics
- **Frontend**: Dashboard aggregates metrics across scans

### **Interactive Knowledge Base** (RAG)
- Query endpoint leveraging ingested code+docs
- Combines code snippets and documentation in responses

## Developer Workflow

### **Local Development**
```bash
# Backend
cd backend && pip install -r requirements.txt
cd app && uvicorn api:app --reload

# Frontend
cd ui && npm install && npm start
# Visit http://localhost:4200
```

### **Testing**
```bash
# Frontend unit tests
ng test

# Backend: pytest recommended (not yet configured in repo)
```

### **Key Environment Variables** (Required for Pipeline Features)
```
QDRANT_HOST=https://<instance>.gcp.cloud.qdrant.io
QDRANT_API_KEY=<key>
VERTEX_PROJECT_ID=<project-id>
VERTEX_LOCATION=<location>
```

## Code Patterns & Conventions

### **Backend**
- **DB Sessions**: Always use `get_db()` dependency injection; close in finally blocks
- **Background Tasks**: Wrap with `background_ingest_task()`/`background_test_task()` - they create independent DB sessions
- **Error Handling**: FastAPI `HTTPException` for API responses; print logs for background task failures
- **Status Fields**: DB model uses string enum patterns ("pending", "ingesting", "completed", "failed")

### **Frontend**
- **Component Structure**: Each module (e.g., `insight-engine/`) contains `.ts`, `.html`, `.scss`, `.spec.ts`
- **Service Layer**: `DataService` centralizes all HTTP calls; components don't directly import `HttpClient`
- **Data Binding**: Use two-way binding with `[(ngModel)]` in templates
- **Routing**: Lazy-load via route configuration; sidebar navigation updates `router.navigate()`

### **Language Support** (Pipeline)
- **Supported**: Python, Java, TypeScript (via tree-sitter)
- **Extensions**: `.py`, `.java`, `.ts` mapped to language configs in `pipeline.py`
- **Query Patterns**: Each language has specific tree-sitter queries for class/function extraction

## Common Tasks

### **Adding a New Module**
1. Create component folder in `ui/src/app/<module-name>/`
2. Generate via `ng generate component <module-name>` or scaffold manually
3. Add route to `app.module.ts` routes array
4. Add navigation button in `sidebar.component.html`
5. If backend integration needed: Add endpoint in `api.py`, call from service

### **Supporting a New Language**
1. Add language config to `LANGUAGE_CONFIG` dict in `pipeline.py`
2. Load tree-sitter language: `get_language('<lang>')` and `get_parser('<lang>')`
3. Define tree-sitter query for class/function extraction
4. Test with sample repo to verify parsing

### **Debugging Background Tasks**
- Monitor logs: Errors printed to stdout in background functions
- Check DB state: Query `RepositoryBranch` table for `status` and error messages
- Qdrant health: Verify connection and collections exist via Qdrant REST API

## Critical Files Reference
- `backend/app/api.py` - REST endpoints and background task orchestration
- `backend/app/pipeline.py` - Ingestion, embedding, Qdrant integration, test generation logic
- `backend/app/models.py` - SQLAlchemy ORM model for repository tracking
- `backend/app/database.py` - SQLAlchemy engine and session factory
- `ui/src/app/app.module.ts` - Route definitions and module imports
- `ui/src/app/data.service.ts` - HTTP API client for all components
- `ui/package.json` - Frontend dependencies and build scripts

## Known Constraints
- **SQLite → PostgreSQL Migration**: Production deployment requires updating `SQLALCHEMY_DATABASE_URL` in `database.py`
- **Test Generation Output**: Currently stored locally in `generated_tests/` directory; consider cloud storage for scalability
- **Rate Limiting**: No implemented; add before production to prevent Qdrant/Vertex AI quota exhaustion
- **Authentication**: Not yet implemented; add OAuth/JWT to endpoints before multi-user deployment
