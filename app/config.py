import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    EXTERNAL_SERVER_URL = os.getenv("EXTERNAL_SERVER_URL", "http://localhost:8081")