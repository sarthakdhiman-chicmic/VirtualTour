

def test_the_routes_also_answer_under_the_service_prefixes(tmp_path):
    """SERVICE_PREFIXES: the backend's RUNPOD_ENDPOINT_URL + RUNPOD_TOUR_RENDER_PATH (e.g.
    /tour/render) reaches the same routes; /ping is a liveness check."""
    from fastapi.testclient import TestClient
    from tour_worker.config import Settings
    from tour_worker.service.http import create_app
    s = Settings.from_env({"WORKER_API_KEY": "k", "SERVICE_PREFIXES": "/tour/render, tour/segment/"})
    assert s.service_prefixes == ("/tour/render", "/tour/segment")
    c = TestClient(create_app(s))
    for base in ("", "/tour/render", "/tour/segment"):
        assert c.get(base + "/ping").json() == {"status": "healthy"}
        assert c.get(base + "/health").status_code == 200
        assert c.get(base + "/status/nope", headers={"authorization": "Bearer k"}).status_code == 404
        assert c.post(base + "/run", json={"input": {}}).status_code == 401
