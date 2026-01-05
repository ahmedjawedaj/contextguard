"""
Generation utilities for ContextGuard.

Goal:
- Provide a thin, overrideable way to turn a `ContextPack` + user prompt into
  a guarded answer. This does not replace your main application generation
  stack; it is a reference implementation and an integration pattern.

Design:
- `Generator` protocol: strategy interface for generation.
- `LLMGenerator`: uses an `LLMProvider` (same protocol as `LLMJudge`) to
  produce a JSON answer, ensuring structured output and easy parsing.

Customization / extension points:
- Override `LLMGenerator.build_prompt` to change how context is formatted.
- Override `LLMGenerator.build_schema` to change required fields or add
  safety tags.
- Provide your own `Generator` implementation (e.g., retrieval-augmented
  streaming, guarded pipelines with red-team filters).
"""

from __future__ import annotations

from typing import Dict, Protocol

from ..core.specs import ContextPack
from ..verify.judges import LLMProvider


class Generator(Protocol):
    """Strategy interface for producing a response from a context pack."""

    def generate(self, prompt: str, context_pack: ContextPack, temperature: float = 0.2) -> Dict:
        ...


class LLMGenerator(Generator):
    """
    Reference generator that uses an `LLMProvider` to produce a JSON answer.

    Pattern:
    - Build a constrained prompt that reminds the model to stay within the context pack.
    - Request JSON with a small schema to simplify parsing and downstream validation.
    - Intended to be swapped out or subclassed for domain-specific generation.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def build_prompt(self, user_prompt: str, context_pack: ContextPack) -> str:
        """
        Build a guarded prompt:
        - Echo the user request.
        - Provide the curated facts-first context pack.
        - Remind the model to refuse answers that cannot be supported.
        """
        facts = []
        for fact in context_pack.facts:
            prov = fact.provenance
            src = prov.source_id if prov else "unknown"
            facts.append(f"- {fact.text} (src: {src})")
        facts_text = "\n".join(facts) or "- (no facts provided)"

        return (
            "You are a grounded assistant. Answer ONLY using the provided facts.\n"
            "If the facts are insufficient, reply with `insufficient`.\n\n"
            f"USER REQUEST:\n{user_prompt}\n\n"
            "FACTS (do not fabricate outside these):\n"
            f"{facts_text}\n"
        )

    def build_schema(self) -> Dict:
        """
        JSON schema to enforce structured, machine-readable output.
        Override to add more fields (e.g., citations array, confidence).
        """
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "status": {"type": "string", "enum": ["ok", "insufficient"]},
            },
            "required": ["answer", "status"],
        }

    def generate(self, prompt: str, context_pack: ContextPack, temperature: float = 0.2) -> Dict:
        guarded_prompt = self.build_prompt(prompt, context_pack)
        schema = self.build_schema()
        return self.llm.complete_json(guarded_prompt, schema=schema, temperature=temperature)

