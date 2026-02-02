import pytest
from fastapi.testclient import TestClient
from serving.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint_structure(client):
    """Test the recommend endpoint returns the correct structure."""
    payload = {"user_id": 1, "k": 5, "explain": False, "include_history": False}
    response = client.post("/recommend", json=payload)

    # 200 OK
    assert response.status_code == 200

    data = response.json()
    assert "movie_ids" in data
    assert "scores" in data
    assert "explanations" in data

    # Check types
    assert isinstance(data["movie_ids"], list)
    assert isinstance(data["scores"], list)

    # Check lengths match
    assert len(data["movie_ids"]) == len(data["scores"])

    # Check k (might be less if not enough candidates, but usually should be k)
    # Since we fallback to popular items, we expect exactly k usually.
    assert len(data["movie_ids"]) <= payload["k"]


def test_recommend_endpoint_unknown_user(client):
    """Test the recommend endpoint handles unknown users gracefully."""
    # Use a very large user ID that definitely doesn't exist
    payload = {"user_id": 999999999, "k": 5}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Should get fallback items
    assert len(data["movie_ids"]) > 0


def test_recommend_with_explanation(client):
    """Test that requesting explanations returns them."""
    payload = {"user_id": 1, "k": 3, "explain": True, "include_history": False}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["explanations"] is not None
    assert isinstance(data["explanations"], list)
    assert len(data["explanations"]) == len(data["movie_ids"])
