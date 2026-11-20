"""
TruthLens — Flask Application Factory
Security: Content-Security-Policy, X-Frame-Options, X-Content-Type-Options headers.
Unified: Image forensics (Sapthagiri) + Text forensics (Rakshita).
"""
import os
from flask import Flask
from flask_cors import CORS

from truthlens_forensics.config import (
    MAX_FILE_SIZE_BYTES,
    ARTIFACT_DIR,
    TEMP_DIR,
    EVIDENCE_DIR,
)


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
        template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    )

    # ── Security config ───────────────────────────────────────────
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES  # 10 MB
    app.config["SECRET_KEY"]         = os.environ.get("TRUTHLENS_SECRET", os.urandom(32))

    # ── CORS ─────────────────────────────────────────────────────
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── Security response headers ─────────────────────────────────
    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]        = "DENY"
        response.headers["Referrer-Policy"]        = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"]      = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' https://generativelanguage.googleapis.com;"
        )
        return response

    # ── Error handlers (OWASP & Fault Tolerance) ───────────────────
    @app.errorhandler(413)
    def file_too_large(e):
        from flask import jsonify, request
        if request.path.startswith("/api/"):
            return jsonify({
                "error": "File size exceeds the 10 MB limit.",
                "max_allowed_mb": MAX_FILE_SIZE_BYTES // (1024 * 1024),
            }), 413
        return "File too large. Maximum allowed size is 10 MB.", 413

    @app.errorhandler(404)
    def not_found(e):
        from flask import jsonify, request
        if request.path.startswith("/api/"):
            return jsonify({"error": "Resource not found", "path": request.path}), 404
        return "404 Not Found", 404

    @app.errorhandler(500)
    def server_error(e):
        from flask import jsonify, request
        if request.path.startswith("/api/"):
            return jsonify({"error": "Internal forensic processing exception occurred"}), 500
        return "500 Internal Server Error", 500

    # ── Ensure directories exist ──────────────────────────────────
    for directory in [ARTIFACT_DIR, TEMP_DIR, EVIDENCE_DIR]:
        os.makedirs(directory, exist_ok=True)

    # ── Register blueprints ───────────────────────────────────────
    from truthlens_forensics.routes import main_bp
    app.register_blueprint(main_bp)

    from truthlens_forensics.text_routes import text_bp
    app.register_blueprint(text_bp)

    return app

