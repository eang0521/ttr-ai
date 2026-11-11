"""Start the game: python run.py  (add --no-browser to skip opening a tab, --port N for another port)."""

import sys

from ttr.server import serve

if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8765
    serve(port=port, open_browser="--no-browser" not in sys.argv)
