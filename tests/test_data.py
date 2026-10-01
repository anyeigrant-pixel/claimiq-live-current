from src.data.generate_data import generate

def test_generate_shape():
    df = generate(250, seed=1)
    assert len(df) == 250
    assert {"actual_loss", "investigation_flag", "claim_type", "claim_narrative"}.issubset(df.columns)
    assert (df.actual_loss > 0).all()
