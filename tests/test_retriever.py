from src.rag.retriever import retrieve

def test_collision_retrieval():
    out = retrieve("collision damage vehicle")
    assert len(out) > 0
    assert any("Collision" in x.text for x in out)
