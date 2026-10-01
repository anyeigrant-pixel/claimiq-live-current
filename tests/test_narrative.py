from src.data.generate_data import generate
from src.nlp.narrative import analyze_narrative, extract_entities


def test_narrative_analysis_extracts_expected_signals():
    text = "On 2026-01-02, an SUV was reported stolen in Austin, TX for $12,500. Total loss is urgent."
    result = analyze_narrative(text)
    assert "stolen" in result["risk_terms"]
    assert result["urgency_score"] > 0
    assert extract_entities(text)["dollar_amounts"] == ["$12,500"]


def test_synthetic_narratives_are_created():
    data = generate(10, seed=9)
    assert data["claim_narrative"].str.len().gt(30).all()
