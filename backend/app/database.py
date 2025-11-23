import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Define the Database URL
# We use a local SQLite file for development.
# When we move to OpenShift, we'll change this to a PostgreSQL connection string.
SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"

# 2. Create the SQLAlchemy "Engine"
# The engine is the core connection to the DB.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    # This setting is required only for SQLite to allow it to be used
    # by multiple threads (which FastAPI's background tasks will do).
    connect_args={"check_same_thread": False}
)

# 3. Create a SessionLocal class
# Each instance of a SessionLocal will be a single database session.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Create a Base class
# Our database model classes will inherit from this class.
Base = declarative_base()

# Helper function to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
