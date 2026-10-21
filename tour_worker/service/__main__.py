"""python -m tour_worker.service: serve the HTTP API with uvicorn on HOST:PORT (tour_worker.config).

One process only: jobs and their statuses live in this process (JOBS_DIR keeps the records), so
do not run it with several uvicorn workers - scale with more containers, each its own endpoint."""
from __future__ import annotations

import logging


def main() -> None:
    import uvicorn

    from tour_worker.config import get_settings
    from tour_worker.service.http import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    app = create_app(settings)                 # refuses to start without a key (or the opt-out)
    uvicorn.run(app, host=settings.host, port=settings.port, workers=1, log_level="info",
                proxy_headers=False, server_header=False)


if __name__ == "__main__":
    main()
