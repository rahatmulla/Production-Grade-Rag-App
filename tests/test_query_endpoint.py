
from fastapi.testclient import TestClient
import app.main as main


def test_blocked_query_skips_rag(monkeypatch):
    """Blocked queries must never reach the RAG agent."""
    client = TestClient(main.app)

    monkeypatch.setattr(
        main,
        "guard",
        lambda q: (True, "Blocked test query."),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("RAG ran for a blocked query.")

    monkeypatch.setattr(main.rag_agent, "invoke", fail_if_called)

    response = client.post(
        "/query",
        json={"q": "How do I make coffee?", "thread_id": "test-blocked"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "Blocked by scope check."
    assert response.json()["answer"] == "Blocked test query."
    assert response.json()["sources"] == []


def test_allowed_query_reaches_rag(monkeypatch):
    """Allowed queries must reach the RAG agent."""
    client = TestClient(main.app)
    called = {"value": False}

    monkeypatch.setattr(main, "guard", lambda q: (False, None))

    def fake_invoke(state, config):
        called["value"] = True
        return {
            "final_answer": "Test Kubernetes answer.",
            "plan": ["Test completed"],
            "status": "success",
            "documents": [],
        }

    monkeypatch.setattr(main.rag_agent, "invoke", fake_invoke)

    response = client.post(
        "/query",
        json={
            "q": "Explain Kubernetes ConfigMaps.",
            "thread_id": "test-allowed",
        },
    )

    assert response.status_code == 200
    assert called["value"] is True
    assert response.json()["answer"] == "Test Kubernetes answer."


def test_guard_failure_does_not_run_rag(monkeypatch):
    """A guard exception must not allow the query through to RAG."""
    client = TestClient(main.app)

    def fail_guard(q):
        raise RuntimeError("Simulated classifier failure")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("RAG ran after guard failure.")

    monkeypatch.setattr(main, "guard", fail_guard)
    monkeypatch.setattr(main.rag_agent, "invoke", fail_if_called)

    response = client.post(
        "/query",
        json={"q": "Explain Kubernetes.", "thread_id": "test-error"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "error"


def test_startup_initializes_guardrails(monkeypatch):
    """Verify that application startup initializes guardrails."""
    from fastapi.testclient import TestClient
    import app.main as main

    called = {"value": False}

    def fake_initialize_rails():
        called["value"] = True

    monkeypatch.setattr(main, "initialize_rails", fake_initialize_rails)

    with TestClient(main.app) as client:
        assert client.get("/").status_code == 200

    assert called["value"] is True


def test_empty_query_is_rejected():
    """Reject an empty query before the endpoint runs."""
    client = TestClient(main.app)

    response = client.post(
        "/query",
        json={"q": "", "thread_id": "test-empty"},
    )

    assert response.status_code == 422


def test_whitespace_query_is_rejected():
    """Reject a query containing only whitespace."""
    client = TestClient(main.app)

    response = client.post(
        "/query",
        json={"q": "   ", "thread_id": "test-whitespace"},
    )

    assert response.status_code == 422


def test_query_over_5000_characters_is_rejected():
    """Reject queries exceeding the maximum length."""
    client = TestClient(main.app)

    response = client.post(
        "/query",
        json={"q": "a" * 5001, "thread_id": "test-long"},
    )

    assert response.status_code == 422
