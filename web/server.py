#!/usr/bin/env python3
import argparse
import http.server
import json
import os
import socketserver
import time

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(WEB_DIR, "results.json")


class Handler(http.server.SimpleHTTPRequestHandler):
    results_path = DEFAULT_RESULTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/api/results", "/api/results/"):
            self.serve_results()
        elif path in ("/api/status", "/api/status/"):
            self.serve_status()
        else:
            super().do_GET()

    def serve_results(self):
        if not os.path.exists(self.results_path):
            self.send_json({"status": "waiting"})
            return

        try:
            with open(self.results_path, "r", encoding="utf-8") as f:
                text = f.read()
            json.loads(text)
        except Exception as exc:
            self.send_json({"error": str(exc)}, code=500)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def serve_status(self):
        exists = os.path.exists(self.results_path)
        mtime = os.path.getmtime(self.results_path) if exists else 0
        self.send_json({
            "ready": exists,
            "mtime": mtime,
            "mtime_str": time.ctime(mtime) if exists else None,
        })

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):
            return
        super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="TSMM benchmark dashboard server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    args = parser.parse_args()

    Handler.results_path = os.path.abspath(args.results)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), Handler) as httpd:
        print(f"TSMM dashboard: http://localhost:{args.port}")
        print(f"Results file: {Handler.results_path}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
