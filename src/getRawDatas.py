import requests
from clients.minio import get_minio_client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MinIO bucket name from .env
BUCKET_NAME = os.environ["MINIO_BUCKET"]

# List of files to download: (url, filename)
FILES = [
    ("https://www.data.gouv.fr/fr/datasets/r/18847484-f622-4ccc-baa9-e6b12f749514", "elections-2022-depts-t1.xlsx"),
]

def main():
    # Connect to MinIO
    client = get_minio_client()

    # Create bucket if it doesn't exist
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)

    for url, filename in FILES:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', 'application/octet-stream')
        # Stream upload to MinIO without saving to disk, with content-type
        client.put_object(
            BUCKET_NAME,
            filename,
            data=response.raw,
            length=int(response.headers.get('content-length', 0)) if response.headers.get('content-length') else -1,
            part_size=10*1024*1024,  # 10MB
            content_type=content_type
        )
        print(f"Uploaded {filename} to bucket {BUCKET_NAME} in MinIO.")

if __name__ == "__main__":
    main()
