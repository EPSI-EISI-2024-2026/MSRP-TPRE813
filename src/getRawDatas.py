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
    ("https://www.observatoire-des-territoires.gouv.fr/outils/cartographie-interactive/api/v1/functions/GC_API_download.php?type=stat&nivgeo=dep&dataset=indic_sex_rp&indic=tx_chom1564", "chomage_2009-2021.xlsx"),
    ("https://www.insee.fr/fr/statistiques/fichier/7456887/ECRT2023-F12.xlsx", "chomage_2022.xlsx"),
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
