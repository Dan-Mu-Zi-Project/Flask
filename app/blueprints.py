from .routes.register import register_bp
from .routes.identify import identify_bp

blueprints = [
    (register_bp, "/face"),
    (identify_bp, "/face")
]
