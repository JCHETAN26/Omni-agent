"""Ingest a directory of Markdown docs into the configured stores, then exit.

    python -m backend.ingestion.seed samples/

Used by the Helm post-install Job: one process seeds Qdrant + Elasticsearch,
instead of every API replica racing to create the same collection. Chunk ids
are deterministic, so re-running is idempotent.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from backend.core.config import get_settings
from backend.core.services import Services
from backend.ingestion.loader import load_markdown_dir


async def main(path: str) -> None:
    services = await Services.build(get_settings())
    try:
        docs = load_markdown_dir(path)
        report = await services.processor.ingest(docs) if docs else None
        print(
            json.dumps(
                {
                    "path": path,
                    "documents": len(docs),
                    "report": report.model_dump() if report else None,
                }
            )
        )
    finally:
        await services.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path")
    asyncio.run(main(p.parse_args().path))
