# Integration Cookbook

Purpose: show concrete adapters and patterns to plug ContextGuard into real stacks.

## LLM providers

**OpenAI with budget + retries**
```python
from contextguard import OpenAIProvider, BudgetedProvider, RetryingProvider, LLMJudge

base = OpenAIProvider(
    model="gpt-4o-mini",
    max_output_tokens=300,
    max_prompt_chars=8000,
    timeout=30.0,
)
budgeted = BudgetedProvider(base, max_prompt_chars=8000, max_output_tokens=300)
llm = RetryingProvider(budgeted, max_attempts=3, base_delay=0.5, max_delay=4.0)
judge = LLMJudge(llm)
```

**Local NLI (no LLM call)**
```python
from contextguard import create_judge
judge = create_judge("nli", model_name="cross-encoder/nli-deberta-v3-base")
```

## Vector DB adapters

**LangChain retriever**
```python
from contextguard import LangChainRetrieverAdapter, CanonicalFilters
lc_ret = your_langchain_retriever  # e.g., from LC vectorstore.as_retriever()
adapter = LangChainRetrieverAdapter(lc_ret, source_type=SourceType.SECONDARY)
filters = CanonicalFilters.from_state_spec(state)
chunks = adapter.search("acme 2024 revenue", filters=filters, k=5)
```

**LlamaIndex retriever**
```python
from contextguard import LlamaIndexRetrieverAdapter
li_ret = index.as_retriever()
adapter = LlamaIndexRetrieverAdapter(li_ret, source_type=SourceType.PRIMARY)
chunks = adapter.search("acme 2024 revenue", filters=filters, k=5)
```

**Chroma**
```python
from contextguard import ChromaRetrieverAdapter, CanonicalFilters
import chromadb
client = chromadb.Client()
collection = client.get_or_create_collection("docs")
def embed(text: str): ...
adapter = ChromaRetrieverAdapter(collection, embed_fn=embed, source_type=SourceType.SECONDARY)
chunks = adapter.search("acme 2024 revenue", filters=filters, k=5)
```

**Qdrant**
```python
from contextguard import QdrantRetrieverAdapter, CanonicalFilters
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
def embed(text: str): ...
adapter = QdrantRetrieverAdapter(client, collection="docs", embed_fn=embed, source_type=SourceType.SECONDARY)
chunks = adapter.search("acme 2024 revenue", filters=filters, k=5)
```

## Async verification
```python
import asyncio
from contextguard import async_run_verification, RuleBasedJudge, MockRetriever

retriever = MockRetriever()
# add_chunk(...) as needed
judge = RuleBasedJudge()
overall_label, overall_conf, claim_verdicts = asyncio.run(
    async_run_verification(claims, state, retriever, judge=judge)
)
```

## Storage
**S3 Store**
```python
from contextguard import S3Store
store = S3Store(bucket="my-bucket", prefix="contextguard/")
store.save_state("thread1", state)
```

## Patterns
- Providers: use `BudgetedProvider` + `RetryingProvider` around your `LLMProvider`.
- Retrievers: use adapters or implement `Retriever.search(query, filters, k)` returning `Chunk` with `provenance.source_id/source_type`, `entity_ids`, `year`, `metadata.doc_type`.
- Async: `async_run_verification` for concurrent retrieval/judge.

