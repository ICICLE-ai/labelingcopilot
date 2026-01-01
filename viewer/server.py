#!/usr/bin/env python3
"""
Annotation Viewer — lightweight web server for browsing curated datasets.

Serves the viewer UI and proxies images from both:
  - agent_output/images/ (curated dataset)
  - The retrieval service (full image pool via MinIO)

Usage:
    python viewer/server.py [--port 8501] [--dataset agent_output/dataset.json]
"""

import argparse
import json
import mimetypes
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen


RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8000")
ANNOTATOR_URL = os.environ.get("ANNOTATOR_URL", "http://localhost:8080")
SYNTHESIS_URL = os.environ.get("SYNTHESIS_URL", "http://localhost:8090")

VIEWER_DIR = Path(__file__).parent
PROJECT_DIR = VIEWER_DIR.parent


class ViewerHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # --- API routes ---
        if path == "/api/dataset":
            self._serve_json_file(self.server.dataset_path)
        elif path == "/api/state":
            state_path = self.server.dataset_path.parent / "curation_state.json"
            self._serve_json_file(state_path)
        elif path == "/api/pool-status":
            self._proxy_get(f"{RETRIEVAL_URL}/status")
        elif path == "/api/annotator-health":
            self._proxy_get(f"{ANNOTATOR_URL}/health")
        elif path == "/api/synthesis-health":
            self._proxy_get(f"{SYNTHESIS_URL}/health")

        # --- Image routes ---
        elif path.startswith("/img/curated/"):
            # Serve from agent_output/images/
            filename = path[len("/img/curated/"):]
            img_path = self.server.dataset_path.parent / "images" / filename
            self._serve_file(img_path)
        elif path.startswith("/img/pool/"):
            # Proxy from retrieval service
            key = path[len("/img/pool/"):]
            self._proxy_get(f"{RETRIEVAL_URL}/images/{key}")

        # --- Static files ---
        elif path == "/" or path == "/index.html":
            self._serve_file(VIEWER_DIR / "index.html")
        else:
            self.send_error(404)

    def _serve_json_file(self, filepath):
        try:
            data = Path(filepath).read_text()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode())
        except FileNotFoundError:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

    def _serve_file(self, filepath):
        try:
            data = Path(filepath).read_bytes()
            mime = mimetypes.guess_type(str(filepath))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404)

    def _proxy_get(self, url):
        try:
            resp = urlopen(url, timeout=10)
            data = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/octet-stream"))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, format, *args):
        # Quieter logging
        if "/api/" not in self.path and "/img/" not in self.path:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Annotation Viewer")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--dataset", default="agent_output/dataset.json")
    args = parser.parse_args()

    dataset_path = (PROJECT_DIR / args.dataset).resolve()
    print(f"Dataset: {dataset_path}")
    print(f"Images:  {dataset_path.parent / 'images'}")

    server = HTTPServer(("0.0.0.0", args.port), ViewerHandler)
    server.dataset_path = dataset_path

    print(f"Viewer running at http://0.0.0.0:{args.port}")
    print(f"Open: http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
