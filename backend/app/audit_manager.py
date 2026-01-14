# --- audit_manager.py ---
import uuid
from database import SessionLocal
import models
import json

class AuditLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id

    def log(self, component: str, event_type: str, summary: str, details: dict = None):
        """
        Writes a structured log entry to the database.
        """
        db = SessionLocal()
        try:
            # Ensure details are JSON serializable
            if details:
                try:
                    json.dumps(details)
                except (TypeError, OverflowError):
                    details = {"raw_content": str(details)}

            entry = models.AuditLog(
                run_id=self.run_id,
                component=component,
                event_type=event_type,
                summary=summary,
                details=details or {}
            )
            db.add(entry)
            db.commit()
            print(f"[{component}] {summary}") # Keep console output for dev
        except Exception as e:
            print(f"FAILED TO AUDIT LOG: {e}")
        finally:
            db.close()

# Factory to create a logger for a specific run
def get_logger(run_id: str = None):
    if not run_id:
        run_id = str(uuid.uuid4())
    return AuditLogger(run_id)