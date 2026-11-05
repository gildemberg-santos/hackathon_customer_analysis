"""Testes do treinamento e da persistência do modelo."""
import numpy as np
import pandas as pd

from src.predict import Predict


def _dataset(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    income = rng.normal(5000, 1500, rows)
    score = rng.integers(300, 900, rows)
    sold = ((income > 5000) & (score > 600)).astype(int)
    return pd.DataFrame({"lead_income": income, "lead_bank_score": score, "sale_sold": sold})


FEATURES = ["lead_income", "lead_bank_score"]


def test_save_trains_and_persists_model(tmp_path):
    """save() treina o modelo, grava o .pkl e preenche predições e acurácia."""
    model = Predict(str(tmp_path / "modelo"), _dataset(), FEATURES, "sale_sold")

    model.save()

    assert (tmp_path / "modelo.pkl").exists()
    assert len(model.y_pred) == len(model.y_test)
    assert 0.0 <= model.accuracy <= 1.0


def test_load_reuses_saved_model(tmp_path):
    """load() reaproveita o modelo salvo sem treinar de novo."""
    path = str(tmp_path / "modelo")
    data = _dataset()
    Predict(path, data, FEATURES, "sale_sold").save()

    loaded = Predict(path, data, FEATURES, "sale_sold")
    loaded.load()

    assert len(loaded.y_pred) == len(loaded.y_test)
    assert loaded.accuracy > 0.7
