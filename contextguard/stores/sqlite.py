"""
ContextGuard SQLite Store

Default storage implementation using SQLite for zero-ops deployment.

Features:
- Single-file database (works in notebooks, CLI, tests)
- In-memory option for testing
- Automatic schema creation
- Thread-safe for basic use cases

This is the "Simon-ish" choice: simple, inspectable, works everywhere.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from ..core.specs import StateSpec, VerdictReport
from ..core.trace import TraceGraph
from .protocols import Store


class SQLiteStore(Store):
    """
    SQLite-backed storage for ContextGuard.
    
    Usage:
        store = SQLiteStore("contextguard.db")
        store.save_state("thread_1", state)
        loaded = store.load_state("thread_1")
    
    For in-memory (testing):
        store = SQLiteStore(":memory:")
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(
        self,
        db_path: str = "contextguard.db",
        create_tables: bool = True,
    ):
        """
        Initialize SQLite store.
        
        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory
            create_tables: Whether to create tables if they don't exist
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        
        if create_tables:
            self._create_tables()
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    @contextmanager
    def _cursor(self):
        """Context manager for cursor with commit."""
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        with self._cursor() as cursor:
            # States table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS states (
                    thread_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Facts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    fact_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    fact_text TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    scope_json TEXT,
                    entity_ids_json TEXT,
                    year INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create index for fact queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_thread 
                ON facts(thread_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_year 
                ON facts(year)
            """)
            
            # Runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    trace_json TEXT,
                    input_content TEXT,
                    overall_label TEXT,
                    overall_confidence REAL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create index for run queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_thread 
                ON runs(thread_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_created 
                ON runs(created_at DESC)
            """)
            
            # Metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Set schema version
            cursor.execute("""
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('schema_version', ?)
            """, (str(self.SCHEMA_VERSION),))
    
    # =========================================================================
    # STATE OPERATIONS
    # =========================================================================
    
    def load_state(self, thread_id: str) -> Optional[StateSpec]:
        """Load state for a thread."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT state_json FROM states WHERE thread_id = ?",
                (thread_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            data = json.loads(row["state_json"])
            return StateSpec.model_validate(data)
    
    def save_state(self, thread_id: str, state: StateSpec) -> None:
        """Save state for a thread."""
        now = datetime.now(timezone.utc).isoformat()
        state_json = state.model_dump_json()
        
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO states (thread_id, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
            """, (thread_id, state_json, now, now))
    
    def delete_state(self, thread_id: str) -> bool:
        """Delete state for a thread."""
        with self._cursor() as cursor:
            cursor.execute(
                "DELETE FROM states WHERE thread_id = ?",
                (thread_id,)
            )
            return cursor.rowcount > 0
    
    def list_threads(self) -> List[str]:
        """List all thread IDs with stored state."""
        with self._cursor() as cursor:
            cursor.execute("SELECT thread_id FROM states ORDER BY updated_at DESC")
            return [row["thread_id"] for row in cursor.fetchall()]
    
    # =========================================================================
    # FACT OPERATIONS
    # =========================================================================
    
    def add_fact(
        self,
        thread_id: str,
        fact_text: str,
        provenance: Dict[str, Any],
        confidence: float,
        scope: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a fact to the store."""
        fact_id = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()
        
        # Extract entity_ids and year from scope if present
        entity_ids = scope.get("entity_ids", []) if scope else []
        year = scope.get("year") if scope else None
        
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO facts (
                    fact_id, thread_id, fact_text, provenance_json,
                    confidence, scope_json, entity_ids_json, year, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact_id,
                thread_id,
                fact_text,
                json.dumps(provenance),
                confidence,
                json.dumps(scope) if scope else None,
                json.dumps(entity_ids),
                year,
                now,
            ))
        
        return fact_id
    
    def query_facts(
        self,
        thread_id: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        year: Optional[int] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Query facts by filters."""
        conditions = ["confidence >= ?"]
        params: List[Any] = [min_confidence]
        
        if thread_id is not None:
            conditions.append("thread_id = ?")
            params.append(thread_id)
        
        if year is not None:
            conditions.append("year = ?")
            params.append(year)
        
        where_clause = " AND ".join(conditions)
        
        with self._cursor() as cursor:
            cursor.execute(f"""
                SELECT * FROM facts
                WHERE {where_clause}
                ORDER BY created_at DESC
            """, params)
            
            rows = cursor.fetchall()
        
        # Post-filter by entity_ids if specified
        # (SQLite JSON querying is limited)
        results = []
        for row in rows:
            fact = {
                "fact_id": row["fact_id"],
                "thread_id": row["thread_id"],
                "fact_text": row["fact_text"],
                "provenance": json.loads(row["provenance_json"]),
                "confidence": row["confidence"],
                "scope": json.loads(row["scope_json"]) if row["scope_json"] else None,
                "entity_ids": json.loads(row["entity_ids_json"]) if row["entity_ids_json"] else [],
                "year": row["year"],
                "created_at": row["created_at"],
            }
            
            # Filter by entity_ids if specified
            if entity_ids is not None:
                fact_entities = set(fact["entity_ids"])
                if not fact_entities.intersection(entity_ids):
                    continue
            
            results.append(fact)
        
        return results
    
    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Get a fact by ID."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return {
                "fact_id": row["fact_id"],
                "thread_id": row["thread_id"],
                "fact_text": row["fact_text"],
                "provenance": json.loads(row["provenance_json"]),
                "confidence": row["confidence"],
                "scope": json.loads(row["scope_json"]) if row["scope_json"] else None,
                "entity_ids": json.loads(row["entity_ids_json"]) if row["entity_ids_json"] else [],
                "year": row["year"],
                "created_at": row["created_at"],
            }
    
    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact by ID."""
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            return cursor.rowcount > 0
    
    # =========================================================================
    # RUN OPERATIONS
    # =========================================================================
    
    def save_run(
        self,
        thread_id: str,
        report: VerdictReport,
        trace: Optional[TraceGraph] = None,
        input_content: Optional[str] = None,
    ) -> str:
        """Save a verification run."""
        run_id = report.report_id
        now = datetime.now(timezone.utc).isoformat()
        
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO runs (
                    run_id, thread_id, report_json, trace_json,
                    input_content, overall_label, overall_confidence, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                thread_id,
                report.model_dump_json(),
                trace.to_json() if trace else None,
                input_content,
                report.overall_label.value,
                report.overall_confidence,
                now,
            ))
        
        return run_id
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a run by ID."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return {
                "run_id": row["run_id"],
                "thread_id": row["thread_id"],
                "report": json.loads(row["report_json"]),
                "trace": json.loads(row["trace_json"]) if row["trace_json"] else None,
                "input_content": row["input_content"],
                "overall_label": row["overall_label"],
                "overall_confidence": row["overall_confidence"],
                "created_at": row["created_at"],
            }
    
    def list_runs(
        self,
        thread_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List runs, optionally filtered by thread."""
        with self._cursor() as cursor:
            if thread_id is not None:
                cursor.execute("""
                    SELECT run_id, thread_id, overall_label, overall_confidence, created_at
                    FROM runs
                    WHERE thread_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (thread_id, limit))
            else:
                cursor.execute("""
                    SELECT run_id, thread_id, overall_label, overall_confidence, created_at
                    FROM runs
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            return [
                {
                    "run_id": row["run_id"],
                    "thread_id": row["thread_id"],
                    "overall_label": row["overall_label"],
                    "overall_confidence": row["overall_confidence"],
                    "created_at": row["created_at"],
                }
                for row in cursor.fetchall()
            ]
    
    def get_trace(self, run_id: str) -> Optional[TraceGraph]:
        """Get the trace graph for a run."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT trace_json FROM runs WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            
            if row is None or row["trace_json"] is None:
                return None
            
            return TraceGraph.from_json(row["trace_json"])
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def vacuum(self) -> None:
        """Reclaim unused space in the database."""
        with self._cursor() as cursor:
            cursor.execute("VACUUM")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM states")
            state_count = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM facts")
            fact_count = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM runs")
            run_count = cursor.fetchone()["count"]
        
        # Get file size if not in-memory
        file_size = None
        if self.db_path != ":memory:":
            path = Path(self.db_path)
            if path.exists():
                file_size = path.stat().st_size
        
        return {
            "threads": state_count,
            "facts": fact_count,
            "runs": run_count,
            "file_size_bytes": file_size,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_store(
    path: str = "contextguard.db",
    in_memory: bool = False,
) -> SQLiteStore:
    """
    Create a SQLite store.
    
    Args:
        path: Path to database file
        in_memory: If True, use in-memory database (ignores path)
    """
    db_path = ":memory:" if in_memory else path
    return SQLiteStore(db_path)


def get_default_store() -> SQLiteStore:
    """Get the default store (contextguard.db in current directory)."""
    return SQLiteStore("contextguard.db")
