"""Start the local forecasting workspace on an available port."""

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Steel Energy on this computer.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("backend.main:app", host="127.0.0.1", port=args.port, workers=1)
