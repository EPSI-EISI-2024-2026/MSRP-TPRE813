import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

def get_warehouse_engine():
    uri = os.environ["WAREHOUSE_URI"]
    return create_engine(uri)

def get_datamart_engine():
    uri = os.environ["DATAMART_URI"]
    return create_engine(uri)
