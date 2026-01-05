from examples._support_minimal import run_minimal_proof  # see note below


def test_supported_claim():
    report = run_minimal_proof("ACME 2024 revenue was $200M.")
    assert report.overall_label == "SUPPORTED"


def test_contradicted_claim():
    report = run_minimal_proof("ACME 2024 revenue was $500M.")
    assert report.overall_label == "CONTRADICTED"


def test_insufficient_claim():
    report = run_minimal_proof("ACME 2025 revenue was $999M.")
    assert report.overall_label in ("INSUFFICIENT", "MIXED")
