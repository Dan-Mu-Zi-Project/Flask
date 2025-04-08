import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    EXTERNAL_SERVER_URL = os.getenv("EXTERNAL_SERVER_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SWAGGER_ENABLED = False
    DEBUG = False
    TESTING = False
    ENV = "production"

class DevelopmentConfig(Config):
    DEBUG = True
    ENV = "development"
    SWAGGER_ENABLED = True

class ProductionConfig(Config):
    DEBUG = False
    ENV = "production"
    SWAGGER_ENABLED = True