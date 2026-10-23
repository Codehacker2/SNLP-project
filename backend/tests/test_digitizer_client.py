"""Upload bridge checks without starting OCR models or network services."""

import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import digitizer_client


class FakeResponse:
    def __init__(self, result):
        self.result = result

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.result).encode("utf-8")


def test_page_upload_forwards_pdf_or_image_bytes_and_roster(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse({"schema_version": "1.0", "booklet_id": "B1", "answers": []})

    monkeypatch.setattr(digitizer_client, "urlopen", fake_urlopen)
    request = digitizer_client.DigitizePagesRequest(
        pages=["JVBERi0=", "aW1hZ2U="], booklet_id="B1", roster=["23D001"]
    )
    result = digitizer_client.digitize_pages(request)
    assert captured["url"] == "http://127.0.0.1:8002/digitize-base64"
    assert captured["body"] == {
        "pages": ["JVBERi0=", "aW1hZ2U="], "booklet_id": "B1", "roster": ["23D001"]
    }
    assert captured["timeout"] == 600
    assert result["booklet_id"] == "B1"


def test_digitizer_unavailable_has_actionable_error(monkeypatch):
    from urllib.error import URLError

    monkeypatch.setattr(digitizer_client, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")))
    with pytest.raises(HTTPException) as exc:
        digitizer_client.digitize_pages(digitizer_client.DigitizePagesRequest(pages=["aW1hZ2U="]))
    assert exc.value.status_code == 502
    assert "Start it on port 8002" in exc.value.detail
