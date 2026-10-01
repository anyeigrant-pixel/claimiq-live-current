from src.rag.evaluate import evaluate


def test_rag_evaluation_covers_25_questions():
    report = evaluate()
    assert report["question_count"] >= 25
    assert report["citation_accuracy"] >= 0.8
    assert report["groundedness"] == 1.0
