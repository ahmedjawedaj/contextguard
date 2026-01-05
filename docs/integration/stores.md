# Stores

Interface
- Store protocols cover state, facts, and runs/traces. Implement the `Store` abstract base (load/save/delete state, add/query facts, save/list runs, get trace).

Provided stores
- `SQLiteStore`: zero-ops local store.
- `S3Store`: S3-compatible bucket (keys customizable); optional dependency `boto3`.

Use S3 store
```python
from contextguard import S3Store
store = S3Store(bucket="my-bucket", prefix="contextguard/")
store.save_state("thread1", state)
state_loaded = store.load_state("thread1")
```

Implement your own
- Subclass the Store protocol; use your DB of choice (Postgres/Redis/Firestore).
- Keep the contract: store/retrieve `StateSpec`, facts (text + provenance + confidence), runs (VerdictReport + optional TraceGraph).

