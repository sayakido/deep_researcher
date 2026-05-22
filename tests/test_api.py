"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "llm" in data


class TestResearchEndpoint:
    def test_research_missing_topic(self):
        resp = client.post("/research", json={})
        assert resp.status_code == 422  # validation error

    def test_research_empty_topic(self):
        resp = client.post("/research", json={"topic": ""})
        assert resp.status_code == 422


class TestResearchStreamEndpoint:
    def test_stream_missing_topic(self):
        resp = client.post("/research/stream", json={})
        assert resp.status_code == 422

    def test_stream_returns_sse(self):
        resp = client.post("/research/stream", json={"topic": "test"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
