"""tour_worker.service.redact: signed URLs and secrets never reach a log line."""
from tour_worker.service.redact import REDACTED, redact, redact_text, redact_url

SIGNED = "https://b.s3.amazonaws.com/l/a.jpg?X-Amz-Credential=AKIA&X-Amz-Signature=abc123"


def test_redact_text_strips_query_strings_and_userinfo():
    assert redact_text(f"fetch failed for {SIGNED}: 403") == \
        f"fetch failed for https://b.s3.amazonaws.com/l/a.jpg?{REDACTED}: 403"
    assert redact_text("s3://bucket/key.png and nothing else") == "s3://bucket/key.png and nothing else"
    assert redact_text("https://user:pw@host/x?t=1") == f"https://{REDACTED}@host/x?{REDACTED}"
    assert redact_text("'https://h/p#frag'") == f"'https://h/p?{REDACTED}'"


def test_redact_url():
    assert redact_url("https://user:pw@hooks.example.com/tour/done?token=abc") == "https://hooks.example.com/tour/done"


def test_redact_hides_secret_keys_and_urls_in_values():
    out = redact({"job_type": "tour_render", "image_urls": {"a.jpg": SIGNED},
                  "plan": {"shots": [{"image_uri": SIGNED}]}, "webhook": "https://h/?t=1"})
    assert out["image_urls"] == REDACTED and out["webhook"] == REDACTED
    assert out["plan"]["shots"][0]["image_uri"] == f"https://b.s3.amazonaws.com/l/a.jpg?{REDACTED}"
    assert "abc123" not in str(out)
