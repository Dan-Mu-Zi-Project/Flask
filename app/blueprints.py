from .routes.face.register import register_bp
from .routes.face.identify import identify_bp
from .routes.photo.upload import upload_bp
from .routes.photo.feedback import feedback_bp
from .routes.test.prompt import prompt_bp

blueprints = [
    (register_bp, "/face"),
    (identify_bp, "/face"),
    (upload_bp, "/photo"),
    (feedback_bp, "/photo"),
    (prompt_bp, "/test")
]
