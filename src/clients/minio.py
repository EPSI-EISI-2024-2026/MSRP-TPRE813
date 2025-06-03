import os
from dotenv import load_dotenv
from minio import Minio

# Load environment variables from .env file
load_dotenv()

def get_minio_client():
    MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
    MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
    MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
