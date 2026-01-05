"""
Cloud store adapter (S3-compatible) for ContextGuard.

Design:
- Implements the `Store` protocol using an S3-compatible bucket.
- Uses JSON blobs for state/fact/run data. Traces are stored as JSON.
- Minimal, dependency-light: requires `boto3` only when used.

Customization / extension:
- Override key templates (`state_key`, `fact_key`, `run_key`) to align with
  your org’s layout.
- Subclass to add encryption, compression, or metadata tagging.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from .protocols import Store
from ..core.specs import StateSpec, VerdictReport
from ..core.trace import TraceGraph


class S3Store(Store):
    """
    S3-backed store implementing the Store protocol.

    Note: This is a thin adapter; it assumes bucket-level permissions are
    already in place. Network and AWS credentials are outside this library’s
    scope.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "contextguard/",
        boto3_client: Any = None,
    ):
        try:
            import boto3  # type: ignore  # pragma: no cover - optional dependency
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("S3Store requires boto3. Install with `pip install boto3`.") from e

        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.s3 = boto3_client or boto3.client("s3")

    # ------------------------------------------------------------------
    # Key helpers (override to customize layout)
    # ------------------------------------------------------------------
    def state_key(self, thread_id: str) -> str:
        return f"{self.prefix}state/{thread_id}.json"

    def fact_key(self, fact_id: str) -> str:
        return f"{self.prefix}fact/{fact_id}.json"

    def run_key(self, run_id: str) -> str:
        return f"{self.prefix}run/{run_id}.json"

    def trace_key(self, run_id: str) -> str:
        return f"{self.prefix}trace/{run_id}.json"

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def load_state(self, thread_id: str) -> Optional[StateSpec]:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.state_key(thread_id))
        except self.s3.exceptions.NoSuchKey:
            return None
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return StateSpec.model_validate(data)

    def save_state(self, thread_id: str, state: StateSpec) -> None:
        body = state.model_dump_json().encode("utf-8")
        self.s3.put_object(Bucket=self.bucket, Key=self.state_key(thread_id), Body=body)

    def delete_state(self, thread_id: str) -> bool:
        self.s3.delete_object(Bucket=self.bucket, Key=self.state_key(thread_id))
        return True

    def list_threads(self) -> List[str]:
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=f"{self.prefix}state/")
        ids: List[str] = []
        for item in resp.get("Contents", []):
            key = item["Key"]
            if key.endswith(".json"):
                ids.append(key.rsplit("/", 1)[-1].replace(".json", ""))
        return ids

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------
    def add_fact(
        self,
        thread_id: str,
        fact_text: str,
        provenance: Dict[str, Any],
        confidence: float,
        scope: Optional[Dict[str, Any]] = None,
    ) -> str:
        fact_id = uuid.uuid4().hex[:16]
        record = {
            "fact_id": fact_id,
            "thread_id": thread_id,
            "fact_text": fact_text,
            "provenance": provenance,
            "confidence": confidence,
            "scope": scope,
        }
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.fact_key(fact_id),
            Body=json.dumps(record).encode("utf-8"),
        )
        return fact_id

    def query_facts(
        self,
        thread_id: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        year: Optional[int] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        # Simple scan; for large datasets use an indexable store (Dynamo/PG).
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=f"{self.prefix}fact/")
        out: List[Dict[str, Any]] = []
        for item in resp.get("Contents", []):
            obj = self.s3.get_object(Bucket=self.bucket, Key=item["Key"])
            rec = json.loads(obj["Body"].read().decode("utf-8"))
            if rec.get("confidence", 0) < min_confidence:
                continue
            if thread_id and rec.get("thread_id") != thread_id:
                continue
            scope = rec.get("scope") or {}
            if year and scope.get("year") and scope.get("year") != year:
                continue
            if entity_ids:
                rec_entities = scope.get("entity_ids") or []
                if not any(eid in rec_entities for eid in entity_ids):
                    continue
            out.append(rec)
        return out

    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.fact_key(fact_id))
        except self.s3.exceptions.NoSuchKey:
            return None
        return json.loads(obj["Body"].read().decode("utf-8"))

    def delete_fact(self, fact_id: str) -> bool:
        self.s3.delete_object(Bucket=self.bucket, Key=self.fact_key(fact_id))
        return True

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def save_run(
        self,
        thread_id: str,
        report: VerdictReport,
        trace: Optional[TraceGraph] = None,
        input_content: Optional[str] = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:16]
        run_record = {
            "run_id": run_id,
            "thread_id": thread_id,
            "report": report.model_dump(),
            "input_content": input_content,
        }
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.run_key(run_id),
            Body=json.dumps(run_record).encode("utf-8"),
        )
        if trace:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self.trace_key(run_id),
                Body=json.dumps(trace.to_dict()).encode("utf-8"),
            )
        return run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.run_key(run_id))
        except self.s3.exceptions.NoSuchKey:
            return None
        return json.loads(obj["Body"].read().decode("utf-8"))

    def list_runs(
        self,
        thread_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=f"{self.prefix}run/")
        runs: List[Dict[str, Any]] = []
        for item in resp.get("Contents", []):
            obj = self.s3.get_object(Bucket=self.bucket, Key=item["Key"])
            rec = json.loads(obj["Body"].read().decode("utf-8"))
            if thread_id and rec.get("thread_id") != thread_id:
                continue
            runs.append(rec)
            if len(runs) >= limit:
                break
        return runs

    def get_trace(self, run_id: str) -> Optional[TraceGraph]:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.trace_key(run_id))
        except self.s3.exceptions.NoSuchKey:
            return None
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return TraceGraph.from_dict(data)

