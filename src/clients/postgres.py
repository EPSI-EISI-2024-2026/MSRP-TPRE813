import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

def get_postgres_engine():
    uri = os.environ["POSTGRES_URI"]
    return create_engine(uri)
