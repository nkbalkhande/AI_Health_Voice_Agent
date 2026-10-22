"""Run the API: py -3.13 -m backend [--port 8000] [--host 127.0.0.1]

On Windows uvicorn creates a Proactor event loop, which psycopg async (the
LangGraph Postgres checkpointer) cannot use, so the server runs on a
selector loop here instead of through the `uvicorn` command.
"""

import argparse
import asyncio
import sys

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the voice agent API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = uvicorn.Server(uvicorn.Config("backend.main:app", host=args.host, port=args.port))
    if sys.platform == "win32":
        asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
    else:
        server.run()


if __name__ == "__main__":
    main()
