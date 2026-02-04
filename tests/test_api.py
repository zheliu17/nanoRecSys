# Copyright (c) 2026 Zhe Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import os
from fastapi.testclient import TestClient
from serving.app.main import app


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("NANORECSYS_STUB", "1")
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
