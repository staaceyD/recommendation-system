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


def test_list_ratings_empty(client):
    resp = client.get("/ratings")

    assert resp.status_code == 200
    assert resp.get_json() == {"ratings": [], "page": 1, "per_page": 20, "total": 0}


def test_list_ratings_returns_movie_and_score(client, make_movie, make_user, make_rating):
    user = make_user(1)
    movie = make_movie("Toy Story (1995)", ["Animation", "Adventure"])
    make_rating(user, movie, 4.5)

    body = client.get("/ratings").get_json()

    assert body["total"] == 1
    item = body["ratings"][0]
    assert item["user_id"] == 1
    assert item["rating"] == 4.5
    assert item["movie"] == {
        "id": movie.id,
        "title": "Toy Story (1995)",
        "genres": ["Adventure", "Animation"],
    }


def test_list_ratings_is_ordered_by_rating_desc(client, make_movie, make_user, make_rating):
    user = make_user(1)
    for title, score in (("Low", 1.0), ("High", 5.0), ("Mid", 3.0)):
        make_rating(user, make_movie(title), score)

    body = client.get("/ratings").get_json()

    assert [r["movie"]["title"] for r in body["ratings"]] == ["High", "Mid", "Low"]


def test_filter_by_user(client, make_movie, make_user, make_rating):
    movie = make_movie("Heat (1995)")
    make_rating(make_user(1), movie, 4.0)
    make_rating(make_user(2), movie, 2.0)

    body = client.get("/ratings?user_id=1").get_json()

    assert body["total"] == 1
    assert body["ratings"][0]["user_id"] == 1


def test_filter_by_rating_range(client, make_movie, make_user, make_rating):
    user = make_user(1)
    for title, score in (("Low", 1.0), ("Mid", 3.0), ("High", 5.0)):
        make_rating(user, make_movie(title), score)

    body = client.get("/ratings?min_rating=2.5&max_rating=4").get_json()

    assert [r["movie"]["title"] for r in body["ratings"]] == ["Mid"]


def test_rating_bounds_are_inclusive(client, make_movie, make_user, make_rating):
    user = make_user(1)
    for title, score in (("Low", 1.0), ("High", 5.0)):
        make_rating(user, make_movie(title), score)

    body = client.get("/ratings?min_rating=1&max_rating=5").get_json()

    assert body["total"] == 2


def test_ratings_pagination_slices_results(client, make_movie, make_user, make_rating):
    user = make_user(1)
    for i in range(5):
        make_rating(user, make_movie(f"Movie {i}"), 4.0)

    body = client.get("/ratings?per_page=2&page=2").get_json()

    assert body["total"] == 5
    assert body["page"] == 2
    assert [r["movie"]["title"] for r in body["ratings"]] == ["Movie 2", "Movie 3"]


def test_list_ratings_unknown_user_is_404(client):
    assert client.get("/ratings?user_id=999999").status_code == 404


def test_list_ratings_rejects_bad_filters(client, make_user):
    make_user(1)

    assert client.get("/ratings?user_id=abc").status_code == 400
    assert client.get("/ratings?min_rating=abc").status_code == 400
    assert client.get("/ratings?min_rating=0").status_code == 400
    assert client.get("/ratings?max_rating=6").status_code == 400
    assert client.get("/ratings?min_rating=4&max_rating=2").status_code == 400
