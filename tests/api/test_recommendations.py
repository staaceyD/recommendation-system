import pytest

from app.services import collaborative


@pytest.fixture(autouse=True)
def _small_thresholds(monkeypatch):
    monkeypatch.setattr(collaborative, "MIN_RATINGS", 1)
    monkeypatch.setattr(collaborative, "PRIOR_STRENGTH", 5)


def test_user_id_is_required(client):
    assert client.get("/recommendations").status_code == 400
    assert client.get("/recommendations?user_id=abc").status_code == 400


def test_falls_back_to_collaborative_without_a_model(client, preferences, model_dir):
    body = client.get("/recommendations?user_id=999999").get_json()

    assert body["strategy"] == "collaborative"
    assert len(body["recommendations"]) > 0
    assert all("title" in movie for movie in body["recommendations"])


def test_uses_rating_based_when_the_model_knows_the_user(client, preferences, train_model):
    train_model()

    body = client.get(f"/recommendations?user_id={preferences['target'].id}").get_json()

    assert body["strategy"] == "rating_based"
    assert len(body["recommendations"]) > 0


def test_limit_is_capped(client, preferences):
    body = client.get("/recommendations?user_id=999999&limit=9999").get_json()

    assert len(body["recommendations"]) <= 100
