from app.services import rating_based


def test_no_model_returns_empty(make_user, model_dir):
    assert rating_based.recommend(make_user().id) == []


def test_unknown_user_returns_empty(preferences, train_model, make_user):
    train_model()
    stranger = make_user()  # created after training, so absent from the vocab
    assert rating_based.recommend(stranger.id) == []


def test_excludes_seen_and_respects_limit(preferences, train_model):
    train_model()
    seen = {m.id for m in preferences["action"][:4] + preferences["romance"][:4]}

    recs = rating_based.recommend(preferences["target"].id, limit=3)

    assert len(recs) == 3
    assert seen.isdisjoint(recs)


def test_ranks_the_liked_genre_higher(preferences, train_model):
    train_model(epochs=40)
    unseen_action = {preferences["action"][4].id, preferences["action"][5].id}

    recs = rating_based.recommend(preferences["target"].id, limit=4)

    assert set(recs[:2]) == unseen_action
