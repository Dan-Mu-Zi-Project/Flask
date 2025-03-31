from .routes.register import register_bp
from .routes.identify import identify_bp
from .routes.upload_photo import upload_photo_bp

blueprints = [
    (register_bp, "/face"),
    (identify_bp, "/face"),
    (upload_photo_bp, "/face")
]
