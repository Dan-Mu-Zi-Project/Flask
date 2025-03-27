from flask import Flask
from flasgger import Swagger
from .config import DevelopmentConfig, ProductionConfig, Config
from .swagger_config import SWAGGER_TEMPLATE, SWAGGER_CONFIG
from .blueprints import blueprints

def create_app(config_name="production"):
    app = Flask(__name__)

    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
    }
    app_config = config_map.get(config_name, Config)
    app.config.from_object(app_config)

    if app.config.get("SWAGGER_ENABLED", False):
        app.config["SWAGGER"] = SWAGGER_CONFIG
        Swagger(app, template=SWAGGER_TEMPLATE)

    for bp, prefix in blueprints:
        app.register_blueprint(bp, url_prefix=prefix)

    return app
