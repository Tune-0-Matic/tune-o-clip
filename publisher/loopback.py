"""One-time desktop OAuth login, shared by the TikTok and YouTube publishers: open the consent page in the user's
browser and catch the redirect on a fixed loopback port (http://127.0.0.1:<port>/callback, with or without the
trailing slash). Stdlib only."""
import http.server
import time
import urllib.parse
import webbrowser


def catch_redirect(url, port, timeout=180, open_browser=webbrowser.open, clock=time.time):
    """-> the redirect's query parameters ({"code": ..., "state": ...} or {"error": ...}); {} on timeout."""
    got = {}

    class Callback(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path.rstrip("/") == "/callback":  # the browser may also ask for /favicon.ico: ignore that
                got.update({k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<h2>Tune-O-Clip: login received. You can close this tab.</h2>")

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), Callback)
    server.timeout = 1
    try:
        open_browser(url)
        deadline = clock() + timeout
        while "code" not in got and "error" not in got and clock() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    return got
