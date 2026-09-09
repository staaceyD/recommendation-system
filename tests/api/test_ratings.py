from app.models import Rating


def test_rate_new_movie_creates_row(client, make_movie, make_user):
    make_user(1)
    movie = make_movie("Heat (1995)")

    resp = client.post(f"/movies/{movie.id}/rate", json={"user_id": 1, "rating": 4.5})

    assert resp.status_code == 201
    body = resp.get_json()
    assert body["user_id"] == 1
    assert body["movie_id"] == movie.id
    assert body["rating"] == 4.5
    assert body["rated_at"] is not None


def test_rerating_updates_in_place(client, make_movie, make_user):
    make_user(1)
    movie = make_movie("Heat (1995)")

    client.post(f"/movies/{movie.id}/rate", json={"user_id": 1, "rating": 3.0})
    resp = client.post(f"/movies/{movie.id}/rate", json={"user_id": 1, "rating": 5.0})

    assert resp.status_code == 200
    assert resp.get_json()["rating"] == 5.0
    assert Rating.query.count() == 1


def test_unknown_movie_is_404(client, make_user):
    make_user(1)

    resp = client.post("/movies/999999/rate", json={"user_id": 1, "rating": 4.0})

    assert resp.status_code == 404


def test_unknown_user_is_404(client, make_movie):
    movie = make_movie("Heat (1995)")

    resp = client.post(f"/movies/{movie.id}/rate", json={"user_id": 999999, "rating": 4.0})

    assert resp.status_code == 404


def test_missing_fields_are_400(client, make_movie, make_user):
    make_user(1)
    movie = make_movie("Heat (1995)")

    assert client.post(f"/movies/{movie.id}/rate", json={"user_id": 1}).status_code == 400
    assert client.post(f"/movies/{movie.id}/rate", json={"rating": 4.0}).status_code == 400


def test_out_of_range_or_off_step_rating_is_400(client, make_movie, make_user):
    make_user(1)
    movie = make_movie("Heat (1995)")

    for bad in (0, -1, 5.5, 3.3):
        resp = client.post(f"/movies/{movie.id}/rate", json={"user_id": 1, "rating": bad})
        assert resp.status_code == 400, bad


def test_non_json_body_is_400(client, make_movie, make_user):
    make_user(1)
    movie = make_movie("Heat (1995)")

    resp = client.post(f"/movies/{movie.id}/rate", data="nope", content_type="text/plain")

    assert resp.status_code == 400


def test_boolean_is_not_a_valid_user_id(client, make_movie):
    movie = make_movie("Heat (1995)")

    resp = client.post(f"/movies/{movie.id}/rate", json={"user_id": True, "rating": 4.0})

    assert resp.status_code == 400
