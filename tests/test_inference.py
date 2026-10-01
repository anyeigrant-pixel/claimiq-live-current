from src.data.generate_data import generate
from src.models.inference import ClaimModels
from src.models.train_models import FEATURES


def test_model_batch_predictions_are_probability_bounded():
    data = generate(5, seed=8)
    predictions = ClaimModels().predict_batch(data[FEATURES])
    assert len(predictions) == 5
    assert predictions["predicted_loss"].gt(0).all()
    assert predictions["investigation_probability"].between(0, 1).all()
