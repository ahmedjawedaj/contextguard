"""
ContextGuard Store Protocols

This module defines the storage interfaces for:
- StateStore: Persist StateSpec across turns
- FactStore: Store verified facts with provenance
- RunStore: Store verification run history and traces

All stores are designed for pluggable backends (SQLite, Postgres, Redis).
SQLite is the default for zero-ops deployment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..core.specs import StateSpec, VerdictReport
from ..core.trace import TraceGraph


# =============================================================================
# STATE STORE PROTOCOL
# =============================================================================


@runtime_checkable
class StateStore(Protocol):
    """
    Protocol for state persistence.
    
    Stores StateSpec objects keyed by thread_id.
    Used to maintain constraint continuity across turns.
    """
    
    def load(self, thread_id: str) -> Optional[StateSpec]:
        """
        Load state for a thread.
        
        Returns None if no state exists.
        """
        ...
    
    def save(self, thread_id: str, state: StateSpec) -> None:
        """
        Save state for a thread.
        
        Overwrites existing state.
        """
        ...
    
    def delete(self, thread_id: str) -> bool:
        """
        Delete state for a thread.
        
        Returns True if state existed and was deleted.
        """
        ...
    
    def list_threads(self) -> List[str]:
        """List all thread IDs with stored state."""
        ...


# =============================================================================
# FACT STORE PROTOCOL
# =============================================================================


@runtime_checkable
class FactStore(Protocol):
    """
    Protocol for fact persistence.
    
    Stores verified facts with provenance for reuse across queries.
    """
    
    def add(
        self,
        thread_id: str,
        fact_text: str,
        provenance: Dict[str, Any],
        confidence: float,
        scope: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a fact to the store.
        
        Returns the fact ID.
        """
        ...
    
    def query(
        self,
        thread_id: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        year: Optional[int] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Query facts by filters.
        
        Returns list of fact records.
        """
        ...
    
    def get(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Get a fact by ID."""
        ...
    
    def delete(self, fact_id: str) -> bool:
        """Delete a fact by ID."""
        ...


# =============================================================================
# RUN STORE PROTOCOL
# =============================================================================


@runtime_checkable
class RunStore(Protocol):
    """
    Protocol for run history persistence.
    
    Stores verification runs, including:
    - VerdictReport
    - TraceGraph
    - Input content
    """
    
    def save_run(
        self,
        thread_id: str,
        report: VerdictReport,
        trace: Optional[TraceGraph] = None,
        input_content: Optional[str] = None,
    ) -> str:
        """
        Save a verification run.
        
        Returns the run ID.
        """
        ...
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a run by ID."""
        ...
    
    def list_runs(
        self,
        thread_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List runs, optionally filtered by thread."""
        ...
    
    def get_trace(self, run_id: str) -> Optional[TraceGraph]:
        """Get the trace graph for a run."""
        ...


# =============================================================================
# COMBINED STORE
# =============================================================================


class Store(ABC):
    """
    Abstract base class combining all store protocols.
    
    Implementations should subclass this for complete storage.
    """
    
    @abstractmethod
    def load_state(self, thread_id: str) -> Optional[StateSpec]:
        ...
    
    @abstractmethod
    def save_state(self, thread_id: str, state: StateSpec) -> None:
        ...
    
    @abstractmethod
    def delete_state(self, thread_id: str) -> bool:
        ...
    
    @abstractmethod
    def list_threads(self) -> List[str]:
        ...
    
    @abstractmethod
    def add_fact(
        self,
        thread_id: str,
        fact_text: str,
        provenance: Dict[str, Any],
        confidence: float,
        scope: Optional[Dict[str, Any]] = None,
    ) -> str:
        ...
    
    @abstractmethod
    def query_facts(
        self,
        thread_id: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        year: Optional[int] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        ...
    
    @abstractmethod
    def save_run(
        self,
        thread_id: str,
        report: VerdictReport,
        trace: Optional[TraceGraph] = None,
        input_content: Optional[str] = None,
    ) -> str:
        ...
    
    @abstractmethod
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        ...
    
    @abstractmethod
    def list_runs(
        self,
        thread_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Close the store and release resources."""
        ...
