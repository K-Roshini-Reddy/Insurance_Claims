import threading
import time

import requests
import uvicorn


def run_server():
    # Run uvicorn server in the background (test-only)
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, log_level="warning")


def main():
    # Start server in a daemon thread so the script can exit cleanly
    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    # Wait briefly for server startup
    time.sleep(1.5)

    r = requests.get("http://127.0.0.1:8000/health", timeout=5)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    body = r.json()
    assert body.get("status") == "ok", f"Unexpected body: {body}"
    print("health_ok")


if __name__ == "__main__":
    main()
