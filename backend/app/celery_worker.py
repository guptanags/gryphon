import os
from celery import Celery
from pipeline import run_ingestion_pipeline, run_test_generation_pipeline
from database import SessionLocal
import models
from datetime import datetime

# Configure Celery to use Redis
celery_app = Celery(
    "ai_worker",
    broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0")
)

@celery_app.task(bind=True, max_retries=3)
def task_ingest_repo(self, db_id: int, logical_name: str, git_repos: list, confluence_pages: list):
    """
    Celery task for ingestion.
    """
    db = SessionLocal()
    try:
        repo_branch = db.query(models.RepositoryBranch).filter(models.RepositoryBranch.id == db_id).first()
        repo_branch.status = "ingesting"
        db.commit()

        # Run Pipeline
        score = run_ingestion_pipeline(logical_name, git_repos, confluence_pages)

        # Update Success
        repo_branch.status = "completed"
        repo_branch.code_quality_score = score
        repo_branch.last_ingested_at = datetime.now()
        db.commit()

    except Exception as e:
        repo_branch.status = "failed"
        db.commit()
        # Retry logic for transient errors (e.g., Git network issues)
        raise self.retry(exc=e, countdown=60)
    finally:
        db.close()