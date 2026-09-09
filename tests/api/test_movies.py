def test_list_movies_empty(client):
    resp = client.get("/movies")
    assert resp.status_code == 200
    assert resp.get_json() == {"movies": [], "page": 1, "per_page": 20, "total": 0}


def test_list_movies_returns_titles_and_sorted_genres(client, make_movie):
    make_movie("Toy Story (1995)", ["Animation", "Adventure"])
    make_movie("Heat (1995)", ["Action"])

    body = client.get("/movies").get_json()

    assert body["total"] == 2
    assert [m["title"] for m in body["movies"]] == ["Toy Story (1995)", "Heat (1995)"]
    assert body["movies"][0]["genres"] == ["Adventure", "Animation"]


def test_search_by_title_is_case_insensitive(client, make_movie):
    make_movie("Toy Story (1995)")
    make_movie("Heat (1995)")

    body = client.get("/movies?q=toy").get_json()

    assert [m["title"] for m in body["movies"]] == ["Toy Story (1995)"]


def test_filter_by_genre(client, make_movie):
    make_movie("Toy Story (1995)", ["Animation"])
    make_movie("Heat (1995)", ["Action", "Crime"])

    body = client.get("/movies?genre=Action").get_json()

    assert [m["title"] for m in body["movies"]] == ["Heat (1995)"]


def test_pagination_slices_results(client, make_movie):
    for i in range(5):
        make_movie(f"Movie {i}")

    body = client.get("/movies?per_page=2&page=2").get_json()

    assert body["total"] == 5
    assert body["page"] == 2
    assert [m["title"] for m in body["movies"]] == ["Movie 2", "Movie 3"]


def test_per_page_is_capped(client, make_movie):
    make_movie("Toy Story (1995)")

    body = client.get("/movies?per_page=9999").get_json()

    assert body["per_page"] == 100


def test_unparseable_page_falls_back_to_first(client, make_movie):
    make_movie("Toy Story (1995)")

    body = client.get("/movies?page=abc").get_json()

    assert body["page"] == 1
    assert body["total"] == 1
