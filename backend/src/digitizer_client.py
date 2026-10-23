"""HTTP client for the bundled digitizer service."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

class DigitizePagesRequest(BaseModel):
    pages: list[str] = Field(min_length=1, max_length=40)
    booklet_id: str = ""
    roster: list[str] | None = None

    @model_validator(mode="after")
    def check_total_size(self):
        # Base64 adds about one third to the original file size.
        if sum(len(page) for page in self.pages) > 80_000_000:
            raise ValueError("Page files exceed the digitizer's 60 MB upload limit.")
        return self


def digitize_pages(req: DigitizePagesRequest) -> dict:
    url = os.environ.get("DIGITIZER_URL", "http://127.0.0.1:8002").rstrip("/")
    body = json.dumps({
        "pages": req.pages,
        "booklet_id": req.booklet_id,
        "roster": req.roster,
    }).encode("utf-8")
    request = Request(
        f"{url}/digitize-base64",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = f"Digitizer rejected the upload (HTTP {exc.code})."
        try:
            error = json.loads(exc.read().decode("utf-8"))
            if isinstance(error, dict) and isinstance(error.get("detail"), str):
                detail = error["detail"]
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        raise HTTPException(exc.code if exc.code in {400, 413, 422} else 502, detail) from exc
    except TimeoutError as exc:
        raise HTTPException(504, "Digitizer did not finish within 10 minutes.") from exc
    except (URLError, OSError) as exc:
        raise HTTPException(502, "Digitizer is unreachable. Start it on port 8002 or set DIGITIZER_URL.") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(502, "Digitizer returned invalid JSON.") from exc
    if not isinstance(result, dict) or result.get("schema_version") != "1.0":
        raise HTTPException(502, "Digitizer returned an unsupported response.")
    return result
