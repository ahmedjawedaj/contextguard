from contextguard import (
    LLMJudge,
    JudgeResult,
    LLMProviderBase,
    Claim,
    Chunk,
    Provenance,
    SourceType,
)


class DummyProvider(LLMProviderBase):
    def complete_json(self, prompt, schema, temperature: float = 0.0):
        return {
            "schema_version": "v0.1",
            "support": 0.6,
            "contradict": 0.2,
            "rationale": "dummy",
            "reasons": [],
            "confidence": 0.7,
            "evidence_quality": {
                "entity_match": True,
                "time_match": True,
                "metric_match": False,
                "unit_match": False,
            },
        }


def test_llm_judge_with_dummy_provider():
    judge = LLMJudge(DummyProvider())
    claim = Claim(claim_id="c1", text="test claim", entities=[], time=None)
    evidence = Chunk(
        text="test evidence",
        score=1.0,
        provenance=Provenance(source_id="s1", source_type=SourceType.SECONDARY),
    )
    result = judge.score(
        claim=claim,
        evidence=evidence,
        state=None,
    )
    assert isinstance(result, JudgeResult)
    assert result.support_score == 0.6
    assert result.contradict_score == 0.2

