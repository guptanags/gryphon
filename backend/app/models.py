from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.sql import func
from database import Base

class RepositoryBranch(Base):
    """
    Database model for tracking each ingested repository branch.
    """
    __tablename__ = "repository_branches"

    # Core IDs
    id = Column(Integer, primary_key=True, index=True)
    logical_name = Column(String, index=True, nullable=False)
    
    # Composite key fields (indexed for fast lookups)
    repo_url = Column(String, index=True, nullable=False)
    branch_name = Column(String, index=True, nullable=False)

    # Status tracking
    status = Column(String, default="pending") # e.g., "pending", "ingesting", "completed", "failed"
    test_status = Column(String, default="n/a") # e.g., "n/a", "pending", "completed", "failed"

    # Timestamps
    last_ingested_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Scores for visualization
    code_quality_score = Column(Float, nullable=True)
    code_coverage_score = Column(Float, nullable=True)
    
    # AI-generated summary
    summary = Column(String, nullable=True) # Use Text for larger summaries if needed

    def __repr__(self):
        return f"<RepositoryBranch(logical_name='{self.logical_name}', branch='{self.branch_name}')>"
