# LLM Providers

Patterns
- Interface: `LLMProviderBase` (abstract class) or `LLMProvider` protocol — implement `complete_json(prompt, schema, temperature) -> dict`.
- Decorators: `BudgetedProvider` (prompt/output limits), `RetryingProvider` (exponential backoff + jitter + logging). Stackable.
- Circuit-breaker: `CircuitBreakerProvider` adds trip/half-open/close with optional concurrency guard.
- Judge: `LLMJudge` consumes any provider.

Built-ins
- `OpenAIProvider`: Chat Completions with JSON mode, prompt/output budgets, timeout. Pass to `LLMJudge` directly or wrap with `BudgetedProvider` + `RetryingProvider`.

Implement your own
```python
from contextguard import LLMProviderBase, LLMJudge

class MyProvider(LLMProviderBase):
    def complete_json(self, prompt, schema, temperature=0.0):
        # Call your model and return parsed JSON
        return {"support": 0.6, "contradict": 0.1, "rationale": "...", "reasons": [], "confidence": 0.7}

judge = LLMJudge(MyProvider())
```

Budget + retry example
```python
from contextguard import OpenAIProvider, BudgetedProvider, RetryingProvider, LLMJudge
base = OpenAIProvider(model="gpt-4o-mini", max_prompt_chars=8000, max_output_tokens=300, timeout=30)
llm = RetryingProvider(BudgetedProvider(base, max_prompt_chars=8000, max_output_tokens=300), max_attempts=3)
judge = LLMJudge(llm)
```

Notes
- Enforce budgets to avoid runaway cost/latency.
- Add your own logging/metrics by injecting a logger or subclassing the wrappers.

