"""The Go chunker's golden fixture must match the current Python chunker."""

import json

from scripts.gen_ingestion_golden import CHUNKER_GOLDEN, chunker_cases


def test_go_chunker_golden_is_current():
    committed = json.loads(CHUNKER_GOLDEN.read_text())
    assert committed == chunker_cases(), (
        "chunker changed: run `python -m scripts.gen_ingestion_golden` and the Go tests"
    )
