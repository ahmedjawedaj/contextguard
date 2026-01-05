from contextguard.core.trace import TraceGraph
from contextguard.verify.report import ReportBuilder
from contextguard.core.specs import StateSpec, VerdictLabel


def test_trace_deterministic_run_id():
    g1 = TraceGraph(seed="seed123")
    g2 = TraceGraph(seed="seed123")
    assert g1.run_id == g2.run_id


def test_report_deterministic_ids():
    state = StateSpec(thread_id="t1")
    rb = ReportBuilder(thread_id="t1", state=state)
    report1 = rb.build(VerdictLabel.SUPPORTED, 0.9, report_id="fixed", created_at="2025-01-01T00:00:00")
    report2 = rb.build(VerdictLabel.SUPPORTED, 0.9, report_id="fixed", created_at="2025-01-01T00:00:00")
    assert report1.report_id == report2.report_id == "fixed"
    assert report1.created_at == report2.created_at == "2025-01-01T00:00:00"

