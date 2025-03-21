from flask import Flask
from flasgger import Swagger
from .routes.register import model1_bp
from .config import Config

def create_app():
    app = Flask(__name__)

    app.config.from_object(Config)

    # Swagger 설정
    swagger = Swagger(app)

    # 블루프린트 등록
    app.register_blueprint(model1_bp, url_prefix='')

    return app
