from flask import Flask
from flasgger import Swagger
from .config import Config
from .swagger_config import SWAGGER_TEMPLATE, SWAGGER_CONFIG
from .blueprints import blueprints

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    app.config['SWAGGER'] = SWAGGER_CONFIG
    Swagger(app, template=SWAGGER_TEMPLATE)

    for bp, prefix in blueprints:
        app.register_blueprint(bp, url_prefix=prefix)

    return app
