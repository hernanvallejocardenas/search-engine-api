"""Tests for the search engine API.

The fixture builds a fresh SQLite index from the real corpus into a
temporary directory and points the app at it, so the tests exercise the
full stack (SQLite index -> FastAPI endpoints) with real data.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

import build_db
import main

# Resolved against this file, so the tests pass from any working directory.
CORPUS_DIR = str(Path(__file__).parent / "corpus")


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "search.db"
    build_db.build_database(CORPUS_DIR, str(db_path))
    main.DB_PATH = str(db_path)
    return TestClient(main.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_single_term_search(client):
    response = client.get("/search", params={"q": "alice"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "alice"
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["filename"] == "alices-adventures-in-wonderland.txt"
    assert result["title"] == "Alice's Adventures in Wonderland"
    assert result["score"] == pytest.approx(2.30, abs=0.01)
    assert "Alice" in result["snippet"]


def test_multi_term_and_search(client):
    response = client.get("/search", params={"q": "sleepy hollow"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    # The book about Sleepy Hollow ranks first; Alice only mentions
    # "sleepy" once, so it ranks a distant second.
    assert results[0]["filename"] == "the-legend-of-sleepy-hollow.txt"
    assert results[0]["score"] == pytest.approx(0.38, abs=0.01)
    assert results[1]["filename"] == "alices-adventures-in-wonderland.txt"
    assert results[1]["score"] == pytest.approx(0.03, abs=0.01)
    assert results[0]["score"] > results[1]["score"]


def test_phrase_search(client):
    response = client.get("/search",
                          params={"q": '"once upon a midnight dreary"'})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["filename"] == "the-raven.txt"
    assert "once upon a midnight dreary" in results[0]["snippet"].lower()


def test_missing_term(client):
    response = client.get("/search", params={"q": "quantum"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_empty_query(client):
    response = client.get("/search", params={"q": ""})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_case_insensitive(client):
    upper = client.get("/search", params={"q": "ALICE"}).json()["results"]
    lower = client.get("/search", params={"q": "alice"}).json()["results"]
    assert upper == lower


def test_documents_list(client):
    response = client.get("/documents")
    assert response.status_code == 200
    documents = response.json()
    assert len(documents) == 5
    filenames = [doc["filename"] for doc in documents]
    assert filenames == [
        "alices-adventures-in-wonderland.txt",
        "the-gift-of-the-magi.txt",
        "the-legend-of-sleepy-hollow.txt",
        "the-raven.txt",
        "the-yellow-wallpaper.txt",
    ]
    titles = {doc["filename"]: doc["title"] for doc in documents}
    assert titles["the-raven.txt"] == "The Raven"
    assert titles["the-yellow-wallpaper.txt"] == "The Yellow Wallpaper"
