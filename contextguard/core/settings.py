"""
Global budget and safety settings for ContextGuard.
Override via env or constructor params as needed in callers.
"""

# Retrieval budgets
MAX_CLAIMS = 20
MAX_TOTAL_K = 50
MAX_CHUNKS_PER_CLAIM = 12

# Judge budgets
MAX_JUDGE_CHUNKS_PER_CLAIM = 12
MAX_JUDGE_TEXT_LEN = 2000  # characters; rough guardrail for prompt size

