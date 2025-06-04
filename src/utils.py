from minio import Minio

def debugDF(df):
    print(f"Columns and types: {[f'{col}: {dtype}' for col, dtype in zip(df.columns, df.dtypes)]}")
    print(f"Number of rows: {df.shape[0]}")
    print("First 20 rows:")
    print(df.head(20))
    print("Done printing head")

# MinIO file loader
def loadFile(client: Minio, bucket: str, filename: str) -> bytes:
    """
    Load a file from MinIO bucket.
    """
    try:
        data = client.get_object(bucket, filename)
        return data.read()
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None
