from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_positive_danish():
    response = client.post("/v1/sentiment", json={"text": "Det var en god lærer."})
    assert response.status_code == 200
    assert response.json()["score"] == 3

def test_negative_english():
    response = client.post("/v1/sentiment", json={"text": "It was a bad course"})
    assert response.status_code == 200
    assert response.json()["score"] == -3

def test_negative_compound():
    response = client.post("/v1/sentiment", json={"text": "It was a very dry course and I did not learn much."})
    assert response.status_code == 200
    assert response.json()["score"] == -3