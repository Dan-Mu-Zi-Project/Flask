from flask import current_app

def build_external_url(path: str) -> str:
    base = current_app.config["EXTERNAL_SERVER_URL"].rstrip("/")
    return f"{base}/{path}"
