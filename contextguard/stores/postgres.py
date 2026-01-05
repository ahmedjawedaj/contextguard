"""
Postgres store (optional) implementing the Store protocol.

Design:
- Minimal implementation for state/facts/runs using a single connection string.
- Requires `psycopg2-binary` or `psycopg` (not installed by default).
- Schema is created on demand; for production, manage migrations explicitly.

Notes:
- This is a basic, synchronous implementation. For high throughput, consider
  pooling and async drivers. Durability and transactions are best-effort here.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from .protocols import Store
from ..core.specs import StateSpec, VerdictReport
from ..core.trace import TraceGraph


class PostgresStore(Store):
    """Postgres-backed store (optional dependency)."""

    SCHEMA_VERSION = 1

    def __init__(self, dsn: str, *, create_schema: bool = True):
        try:
            import psycopg2  # type: ignore  # pragma: no cover - optional dep
        except ImportError as e:  # pragma: no cover
            raise ImportError("PostgresStore requires psycopg2 or psycopg. Install with `pip install psycopg2-binary`.") from e

        self.psycopg2 = psycopg2
        self.dsn = dsn
        if create_schema:
            self._ensure_schema()

    def _conn(self):
        return self.psycopg2.connect(self.dsn)

    def _ensure_schema(self):
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS states (
                    thread_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS facts (
                    fact_id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    fact_text TEXT,
                    provenance_json TEXT,
                    confidence DOUBLE PRECISION,
                    scope_json TEXT,
                    entity_ids_json TEXT,
                    year INT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    report_json TEXT,
                    input_content TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS traces (
                    run_id TEXT PRIMARY KEY,
                    trace_json TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )
            cur.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', %s) ON CONFLICT (key) DO NOTHING",
                (str(self.SCHEMA_VERSION),),
            )

    # State
    def load_state(self, thread_id: str) -> Optional[StateSpec]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT state_json FROM states WHERE thread_id=%s", (thread_id,))
            row = cur.fetchone()
            if not row:
                return None
            return StateSpec.model_validate(json.loads(row[0]))

    def save_state(self, thread_id: str, state: StateSpec) -> None:
        state_json = state.model_dump_json()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO states (thread_id, state_json, created_at, updated_at)
                VALUES (%s, %s, NOW(), NOW())
                ON CONFLICT (thread_id) DO UPDATE SET
                    state_json=EXCLUDED.state_json,
                    updated_at=EXCLUDED.updated_at
                """,
                (thread_id, state_json),
            )

    def delete_state(self, thread_id: str) -> bool:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM states WHERE thread_id=%s", (thread_id,))
            return cur.rowcount > 0

    def list_threads(self) -> List[str]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT thread_id FROM states ORDER BY updated_at DESC")
            return [row[0] for row in cur.fetchall()]

    # Facts
    def add_fact(
        self,
        thread_id: str,
        fact_text: str,
        provenance: Dict[str, Any],
        confidence: float,
        scope: Optional[Dict[str, Any]] = None,
    ) -> str:
        fact_id = uuid.uuid4().hex[:16]
        entity_ids = (scope or {}).get("entity_ids", [])
        year = (scope or {}).get("year")
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO facts (
                    fact_id, thread_id, fact_text, provenance_json,
                    confidence, scope_json, entity_ids_json, year, created_at
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                """,
                (
                    fact_id,
                    thread_id,
                    fact_text,
                    json.dumps(provenance),
                    confidence,
                    json.dumps(scope) if scope else None,
                    json.dumps(entity_ids),
                    year,
                ),
            )
        return fact_id

    def query_facts(
        self,
        thread_id: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        year: Optional[int] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        clauses = ["confidence >= %s"]
        params: List[Any] = [min_confidence]
        if thread_id:
            clauses.append("thread_id = %s")
            params.append(thread_id)
        if year is not None:
            clauses.append("year = %s")
            params.append(year)
        where = " AND ".join(clauses)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT fact_id, fact_text, provenance_json, confidence, scope_json, entity_ids_json, year FROM facts WHERE {where} ORDER BY created_at DESC",
                params,
            )
            rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            rec = {
                "fact_id": row[0],
                "fact_text": row[1],
                "provenance": json.loads(row[2]) if row[2] else {},
                "confidence": row[3],
                "scope": json.loads(row[4]) if row[4] else {},
                "entity_ids": json.loads(row[5]) if row[5] else [],
                "year": row[6],
            }
            if entity_ids:
                if not any(eid in rec["entity_ids"] for eid in entity_ids):
                    continue
            out.append(rec)
        return out

    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT fact_text, provenance_json, confidence, scope_json, entity_ids_json, year FROM facts WHERE fact_id=%s",
                (fact_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "fact_text": row[0],
                "provenance": json.loads(row[1]) if row[1] else {},
                "confidence": row[2],
                "scope": json.loads(row[3]) if row[3] else {},
                "entity_ids": json.loads(row[4]) if row[4] else [],
                "year": row[5],
            }

    def delete_fact(self, fact_id: str) -> bool:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM facts WHERE fact_id=%s", (fact_id,))
            return cur.rowcount > 0

    # Runs
    def save_run(
        self,
        thread_id: str,
        report: VerdictReport,
        trace: Optional[TraceGraph] = None,
        input_content: Optional[str] = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:16]
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (run_id, thread_id, report_json, input_content, created_at)
                VALUES (%s,%s,%s,%s,NOW())
                """,
                (run_id, thread_id, json.dumps(report.model_dump()), input_content),
            )
            if trace:
                cur.execute(
                    "INSERT INTO traces (run_id, trace_json, created_at) VALUES (%s,%s,NOW())",
                    (run_id, json.dumps(trace.to_dict())),
                )
        return run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT thread_id, report_json, input_content FROM runs WHERE run_id=%s", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "thread_id": row[0],
                "report": json.loads(row[1]) if row[1] else {},
                "input_content": row[2],
            }

    def list_runs(
        self,
        thread_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if thread_id:
            clauses.append("thread_id=%s")
            params.append(thread_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT run_id, thread_id, report_json FROM runs {where} ORDER BY created_at DESC LIMIT %s", params + [limit])
            rows = cur.fetchall()
        out = []
        for row in rows:
            out.append({
                "run_id": row[0],
                "thread_id": row[1],
                "report": json.loads(row[2]) if row[2] else {},
            })
        return out

    def get_trace(self, run_id: str) -> Optional[TraceGraph]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT trace_json FROM traces WHERE run_id=%s", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            return TraceGraph.from_dict(json.loads(row[0]))

