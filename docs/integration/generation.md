# Generation

Patterns
- Generation is optional and separate from verification.
- Interface: `Generator` protocol (`generate(prompt, context_pack, temperature) -> dict`).
- Reference: `LLMGenerator` uses an `LLMProvider` to produce a JSON answer, prompting with a facts-first context pack and “answer only if supported” instructions.

Customize
- Override `LLMGenerator.build_prompt` to change formatting or add domain instructions.
- Override `LLMGenerator.build_schema` to add fields (citations, confidence).
- Implement your own `Generator` to add streaming, guardrails, or different output formats.

Example
```python
from contextguard import LLMGenerator, OpenAIProvider

llm = OpenAIProvider(model="gpt-4o-mini")
gen = LLMGenerator(llm)
result = gen.generate("Summarize ACME revenue.", context_pack)
```

